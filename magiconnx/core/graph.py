import time
import os
import warnings

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
                inits.append(BaseNode.create_node(node))
            else:
                nodes.append(BaseNode.create_node(node))
        outputs = [BaseNode.create_node(output) for output in graph.output]

        return cls(nodes, inputs, outputs, inits, graph.name, **meta)

    ###############################################
    #######              Create             #######
    ###############################################

    @typeassert(name=str, shape=(tuple, list))
    def add_placeholder(self, name, dtype, shape):
        try:
            dtype = np.dtype(dtype)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f'{dtype} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')
        ph = PlaceHolder(name, dtype, shape)
        self._update_ops_map(ph)
        return ph

    @typeassert(name=str, value=np.ndarray)
    def add_initializer(self, name, value):
        init = Initializer(name, value)
        self._update_ops_map(init)
        return init

    @typeassert(name=str, op_type=str, attrs=dict, inputs=list, outputs=list, domain=str)
    def add_node(self, name, op_type, attrs={}, inputs=[], outputs=[], domain=None):
        node = Node(name, op_type, inputs=inputs,
                    outputs=outputs, attrs=attrs, domain=domain)
        self._update_ops_map(node)
        return node

    @typeassert(anchor=str, dst=BaseNode, index=int, mode=str)
    def insert_node(self, anchor, dst, index=0, mode='after'):
        assert dst.name not in self._all_ops_name, f'The insert node (dst.name={dst.name}) has been existed in graph, it is illegal.'
        assert anchor in self._all_ops_name, f'The anchor node ({anchor}) is not exists in graph, please check it!'
        src = self._all_ops_map.get(anchor)

        if mode == 'after':
            if len(dst.inputs) > 1:
                raise RuntimeError(
                    'Only support single input Node, maybe you can use graph.connection')
            if len(src.outputs) > 1:
                print(
                    '[WARNING] Results may be not correct when the anchor node has multi outputs.')
            while dst.outputs:
                dst.outputs.pop()
            dst.outputs.append(src.outputs[index])
            dst.inputs[0] = f'{anchor}/{dst.name}'

            self._all_edges_map[dst.name] = self._all_edges_map[src.name]
            self._all_edges_map[src.name] = [dst.name]
            self._all_ops_map[src.outputs[index]] = dst
            src.outputs[index] = f'{anchor}/{dst.name}'
            self._all_ops_map[f'{anchor}/{dst.name}'] = src

        elif mode == 'before':
            if len(dst.outputs) > 1:
                raise RuntimeError(
                    'Only support single output Node, maybe you can use graph.connection')
            while dst.inputs:
                dst.inputs.pop()
            dst.inputs.append(src.inputs[index])

            input_name = self._all_ops_map[src.inputs[index]].name
            input_index = self._all_edges_map[input_name].index(src.name)
            self._all_edges_map[input_name][input_index] = dst.name
            self._all_edges_map[dst.name] = [src.name]
            dst.outputs[0] = f'{dst.name}/{anchor}'
            self._all_ops_map[f'{dst.name}/{anchor}'] = dst
            src.inputs[index] = f'{dst.name}/{anchor}'
        else:
            raise ValueError(
                f'The mode should be equal to "after" or "before", but got {mode}')

        self._update_ops_map(dst)
        return self

    ###############################################
    #######            Retrieve             #######
    ###############################################
    @typeassert(op_type=str)
    def get_nodes(self, op_type):
        ret = []
        seen = set()
        for node in self._all_ops_map.values():
            if node.name not in seen and node.op_type == op_type:
                ret.append(node)
                seen.add(node.name)
        return ret

    @typeassert(key=str)
    def __getitem__(self, key):
        ret = self._all_ops_map.get(key)
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
        assert key in self._all_ops_name, f'The ({key}) is not exists in graph, please check it!'
        src = self._all_ops_map.pop(key)
        self._del_node(src)
        value.inputs = src.inputs
        value.outputs = src.outputs
        self._all_ops_map[key] = value
        for out in value.outputs:
            self._all_ops_map[out] = value

    ###############################################
    #######             Delete              #######
    ###############################################
    @typeassert(name=str, maps=dict, auto_connection=bool)
    def del_node(self, name, maps={0: 0}, auto_connection=True):
        assert name in self._all_ops_name, f'The ({name}) is not exists in graph, please check it!'
        src = self._all_ops_map.pop(name)
        if not auto_connection:
            self._del_node(src)
            return

        for appendix_name in self._all_edges_map[name]:
            appendix = self._all_ops_map[appendix_name]
            for src_idx, dst_idx in maps.items():
                appendix.set_input(dst_idx, src.inputs[src_idx])
                input_name = self._all_ops_map[src.inputs[src_idx]].name
                edge_idx = self._all_edges_map[input_name].index(name)
                self._all_edges_map[input_name][edge_idx] = appendix_name

        self._del_node(src)
        del self._all_edges_map[name]
        self._all_ops_name.remove(name)

    def _del_node(self, node):
        if node.op_type == INITIALIZER:
            self._model.graph.initializer.remove(node.node)
            # case by keep_initializers_as_inputs=True
            if node.name in self.inputs:
                self._model.graph.input.remove(node.node)
        elif node.op_type == PLACEHOLDER:
            self._model.graph.input.remove(node.node)
        else:
            self._model.graph.node.remove(node.node)

    def keep_default_domain(self):
        while len(self._model.opset_import) > 1:
            self._model.opset_import.pop()
        for name, node in self._all_ops_map:
            if isinstance(node, Node):
                self._all_ops_map[name].clear_domain()

    ###############################################
    #######         graph operation         #######
    ###############################################
    @typeassert(previous=str, out_idx=(int, list, tuple), behind=str, in_idx=(int, list, tuple))
    def connection(self, previous, out_idx, behind, in_idx):
        if previous not in self._all_ops_name or behind not in self._all_ops_name:
            raise ValueError(f'{previous} or {behind} is not exists in graph')
        prev = self._all_ops_map[previous]
        beh = self._all_ops_map[behind]
        if isinstance(out_idx, int):
            out_idx = [out_idx]
        if isinstance(in_idx, int):
            in_idx = [in_idx]
        out_len, in_len = len(out_idx), len(in_idx)
        if (0 in (out_len, in_len)) or \
                ((out_len != in_len) and (1 not in (out_len, in_len))):
            raise RuntimeError(
                f'It is fuzzy to connect between {out_idx} and {in_idx}')
        elif out_len > in_len:
            in_idx = in_idx * out_len
        elif out_len < in_len:
            out_idx = out_idx * in_len
        for idx, odx in zip(in_idx, out_idx):
            beh.inputs[idx] = prev.outputs[odx]

    def __str__(self):
        return helper.printable_graph(self._model.graph)

    @property
    def graph(self):
        return self._model.graph

    @property
    def inputs(self):
        return [in_node.name for in_node in self._model.graph.input]

    @property
    def outputs(self):
        return [out.name for out in self._model.graph.output]

    def save(self, path):
        onnx.save(self._model, path)

    @typeassert(data=(np.ndarray, list))
    def run(self, data):
        model = self._model.SerializeToString()
        return self._run(model, data)

    def _run(self, model, datas):
        try:
            import onnxruntime as rt
        except ImportError:
            raise RuntimeError(
                "\033[45;1m onnxruntime模块导入失败，请检查环境或pip install onnxruntime \033[0m")

        if isinstance(datas, np.ndarray):
            datas = [datas]
        sess = rt.InferenceSession(model)
        inputs = [inode.name for inode in sess.get_inputs()]
        outputs = [out.name for out in sess.get_outputs()]
        ret = sess.run(outputs, {name: data for name,
                                 data in zip(inputs, datas)})
        return ret

    @typeassert(data=(np.ndarray, list), path=str, outputs=(tuple, list))
    def dump(self, data, path='dump', outputs=[]):
        try:
            from skl2onnx.helpers.onnx_helper import (select_model_inputs_outputs,
                                                      enumerate_model_node_outputs)
        except ImportError:
            raise RuntimeError(
                "\033[45;1m skl2onnx模块导入失败，请检查环境或pip install skl2onnx \033[0m")

        if len(outputs) == 0:
            outputs = [
                name for name in enumerate_model_node_outputs(self._model)]
        new_model = select_model_inputs_outputs(self._model, outputs)
        new_model_byte = new_model.SerializeToString()
        arrs = self._run(new_model_byte, data)
        idx = 0
        if not os.path.exists(path):
            os.makedirs(path, mode=0o700)
        for node in self._model.graph.node:
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

    ###############################################
    #######       assistant operation       #######
    ###############################################
    def _update_ops_map(self, name, node):
        exist_node = self._all_ops_map.get(name)
        if exist_node is not None:
            if exist_node.op_type == PLACEHOLDER and node.op_type == INITIALIZER:
                warnings.warn(f'{name} belongs to both {PLACEHOLDER} and {INITIALIZER}, we only keep it as {INITIALIZER}'
                              f'This may be caused by setting keep_initializers_as_inputs=True in torch.onnx.export')
            else:
                raise RuntimeError(
                    f"This is an invalid model. Error: two nodes with same node name ({name})")

        self._all_ops_map[name] = node

    def _update_edges_map(self, node):
        if node.name in self._all_edges_map:
            raise RuntimeError(
                f"This is an invalid model. Error: two nodes with same node name ({node.name})")
        for in_idx in node.inputs:
            if in_idx not in self._all_ops_map:
                continue
            in_name = self._all_ops_map[in_idx].name
            self._all_edges_map.setdefault(in_name, []).append(node.name)

    def optimizer(self, blacklist=[]):
        print('optimizer')
        pass
