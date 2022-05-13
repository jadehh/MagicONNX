from itertools import chain
import time
import os
import warnings
from importlib import import_module

import numpy as np
import onnx
from onnx import (helper, GraphProto, ModelProto, OperatorSetIdProto)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from .. import (BaseGraph, PLACEHOLDER, INITIALIZER)
from .node import BaseNode, PlaceHolder, Initializer, Node
from .assistant import Assistant
from ..utils.log import typeassert


class OnnxGraph(BaseGraph):
    def __init__(self, nodes=[], inputs=[], outputs=[], inits=[], name=None, **kwargs):
        super(OnnxGraph, self).__init__()
        self._inputs = inputs
        self._outputs = outputs
        self._inits = inits
        self._name = name
        self._idx = 0
        Assistant.update_maps(nodes, inputs, outputs, inits)
        for k, v in Assistant.name2node.items():
            print(f'k = {k}')
            print(f'v.inputs = {v.inputs}\n'
                  f'v.prev() = {v.prev()}\n'
                  f'v.outputs = {v.outputs}\n'
                  f'v.next() = {v.next()}')
        self._meta = {'ir_version': kwargs.get('ir_version', 4),
                      # TODO:从version.py中动态读取更好
                      'producer_name': kwargs.get('producer_name', 'MagicONNX'),
                      'producer_version': kwargs.get('producer_version', 'beta'),
                      'domain': kwargs.get('domain', ''),
                      'model_version': kwargs.get('model_version', 0),
                      'opset_import': kwargs.get('opset_import', None)}

    @classmethod
    @typeassert(path_or_bytes=(str, ModelProto, GraphProto))
    def parse(cls, path_or_bytes):
        meta = {}
        if isinstance(path_or_bytes, str):
            path_or_bytes = onnx.load(path_or_bytes)
        if isinstance(path_or_bytes, ModelProto):
            graph = path_or_bytes.graph
            meta = {'ir_version': path_or_bytes.ir_version,
                    'domain': path_or_bytes.domain,
                    'model_version': path_or_bytes.model_version,
                    'doc_string': path_or_bytes.doc_string,
                    'opset_import': path_or_bytes.opset_import}
        else:
            graph = path_or_bytes

        inits = [BaseNode.create_node(init) for init in graph.initializer]
        init_names = {init.name for init in graph.initializer}
        inputs = [BaseNode.create_node(input)
                  for input in graph.input if input.name not in init_names]
        # TODO:这里关于constant结点的处理不够优雅
        nodes = []
        for node in graph.node:
            if node.op_type == 'Constant':
                const = BaseNode.create_node(node)
                const.name = node.output[0]
                inits.append(const)
            else:
                nodes.append(BaseNode.create_node(node))
        outputs = [BaseNode.create_node(output) for output in graph.output]

        return cls(nodes, inputs, outputs, inits, graph.name, **meta)

    ###############################################
    #######              Create             #######
    ###############################################

    @typeassert(name=str, shape=(tuple, list))
    def add_placeholder(self, name, dtype, shape, is_input=True):
        try:
            dtype = np.dtype(dtype)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f'{dtype} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')
        ph = PlaceHolder(name, dtype, shape)
        self._inputs.append(ph)
        if is_input:
            Assistant.update_maps(inputs=[ph])
        else:
            Assistant.update_maps(outputs=[ph])
        return ph

    @typeassert(name=str, value=np.ndarray)
    def add_initializer(self, name, value):
        init = Initializer(name, value)
        self._inits.append(init)
        Assistant.update_maps(inits=[init])
        return init

    @typeassert(name=str, op_type=str, attrs=dict, inputs=list, outputs=list, domain=str)
    def add_node(self, name, op_type, attrs={}, inputs=[], outputs=[], domain=None):
        node = Node(name, op_type, inputs=inputs,
                    outputs=outputs, attrs=attrs, domain=domain)
        Assistant.update_maps(nodes=[node])
        return node

    @typeassert(anchor=str, dst=BaseNode, src_idx=int, mode=str)
    def insert_node(self, anchor, dst, src_idx=0, mode='after'):
        src = Assistant.name2node.get(anchor)
        assert src is None, f'The anchor({anchor}) does not exist in graph, please check it!'

        if mode == 'after':
            if len(dst.inputs) > 1:
                raise RuntimeError(
                    'Only support single input Node, maybe you can use graph.connection')
            dst.outputs = [src.outputs[src_idx]]
            dst.inputs = [f'{anchor}/{dst.name}']
            src.outputs[src_idx] = f'{anchor}/{dst.name}'

        elif mode == 'before':
            if len(dst.outputs) > 1:
                raise RuntimeError(
                    'Only support single output Node, maybe you can use graph.connection')
            dst.inputs = [src.inputs[src_idx]]
            dst.outputs = [f'{dst.name}/{anchor}']
            src.inputs[src_idx] = f'{dst.name}/{anchor}'
        else:
            raise ValueError(
                f'The mode should be equal to "after" or "before", but got {mode}')

        Assistant.update_maps(nodes=[dst])
        return self

    ###############################################
    #######            Retrieve             #######
    ###############################################
    @typeassert(op_type=str)
    def get_nodes(self, op_type):
        ret = []
        seen = set()
        for name, node in Assistant.name2node.items():
            if name not in seen and node.op_type == op_type:
                ret.append(node)
                seen.add(name)
        return ret

    @typeassert(key=str)
    def __getitem__(self, key):
        ret = Assistant.name2node.get(key)
        if ret is None:
            raise ValueError(f'{key} dose not exist in graph')
        return ret

    ###############################################
    #######             Update              #######
    ###############################################
    @typeassert(key=str, value=BaseNode)
    def __setitem__(self, key, value):
        if value.op_type in (INITIALIZER, PLACEHOLDER):
            raise ValueError(
                f'Only supports replacing NodeProto, but value({value.name}) is not')
        assert key in Assistant.name2node, f'The ({key}) is not exists in graph, please check it!'
        src = Assistant.name2node.pop(key)
        value.inputs = src.inputs
        value.outputs = src.outputs
        Assistant.update_maps(nodes=[value])

    ###############################################
    #######             Delete              #######
    ###############################################
    @typeassert(name=str, maps=dict, auto_connection=bool)
    def del_node(self, name, maps={0: 0}, auto_connection=True):
        assert name in Assistant.name2node, f'The Node({name}) is not exists in graph, please check it!'
        src = Assistant.name2node.pop(name)
        if auto_connection:
            Assistant.remove_node(src)

    def keep_default_domain(self):
        self._meta['opset_import'] = None

    ###############################################
    #######         graph operation         #######
    ###############################################
    @typeassert(previous=str, behind=str, maps=dict)
    def connection(self, previous, behind, maps={0: 0}):
        if previous not in Assistant.name2node or behind not in Assistant.name2node:
            raise ValueError(f'{previous} or {behind} is not exists in graph')
        for idx, odx in maps.items():
            Assistant.name2node[behind].inputs[idx] = Assistant.name2node[previous].outputs[odx]

    def __str__(self):
        return helper.printable_graph(self.graph)

    @property
    def graph(self):
        return helper.make_graph(nodes=[node.node for node in self.toposort()],
                                 name=self._name,
                                 inputs=[inp.node for inp in self.inputs],
                                 outputs=[out.node for out in self.outputs],
                                 initializer=[init.node for init in self.inits])

    def toposort(self):
        ret = []
        seen = set()

        def dfs(start, targets):
            for node in start.next():
                if node.op_type in (PLACEHOLDER, INITIALIZER) or node.name in seen:
                    return
                targets.append(node)
                seen.add(node.name)
                dfs(node, targets)
        for gen in chain(self.inputs, self.inits):
            dfs(gen, ret)
        return ret

    @property
    def model(self):
        return helper.make_model(self.graph, **self._meta)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    @property
    def inits(self):
        return self._inits

    def save(self, path):
        onnx.save(self.model, path)

    @typeassert(datas=(np.ndarray, list))
    def run(self, datas):
        self._run(self.model, datas)

    def _run(self, model, datas):
        ort = import_module('onnxruntime')

        if isinstance(datas, np.ndarray):
            datas = [datas]
        sess = ort.InferenceSession(model)
        inputs = [inode.name for inode in sess.get_inputs()]
        outputs = [out.name for out in sess.get_outputs()]
        ret = sess.run(outputs, {name: data
                                 for name, data in zip(inputs, datas)})
        return ret

    @typeassert(data=(np.ndarray, list), path=str, outputs=(tuple, list))
    def dump(self, data, path='dump', outputs=[]):
        select_model_inputs_outputs = import_module(
            'skl2onnx.helpers.onnx_helper.select_model_inputs_outputs')
        enumerate_model_node_outputs = import_module(
            'skl2onnx.helpers.onnx_helper.enumerate_model_node_outputs')

        if len(outputs) == 0:
            outputs = [
                name for name in enumerate_model_node_outputs(self.model)]
        new_model = select_model_inputs_outputs(self.model, outputs)
        new_model_byte = new_model.SerializeToString()
        arrs = self._run(new_model_byte, data)
        idx = 0
        if not os.path.exists(path):
            os.makedirs(path, mode=0o700)
        for node in self.model.graph.node:
            for i, output in enumerate(node.output):
                fname = f'{node.name}_output{i}({output})_{round(time.time() * 1000000)}.npy'
                np.save(os.path.join(path, fname), arrs[idx])
                idx += 1

    @typeassert(new_model_save_path=str, input_tensor_name_list=list, output_tensor_name_list=list, enable_model_check=bool)
    def extract(self, new_model_save_path, input_tensor_name_list, output_tensor_name_list, enable_model_check=True):
        def check_model(model):
            pass
        if not enable_model_check:
            # TODO: Avoid permanent modification
            onnx.checker.check_model = check_model
        print('[INFO] Begin to extract the model.')
        onnx.utils.extract_model(
            self._model_path, new_model_save_path, input_tensor_name_list, output_tensor_name_list)
        print('[INFO] Extract the model completed, model saved in {}.'.format(
            new_model_save_path))

    def optimizer(self, blacklist=[]):
        print('optimizer')
        pass