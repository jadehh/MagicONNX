import time
import os
import warnings
from itertools import chain

import numpy as np
import onnx
from onnx import (NodeProto, TensorProto, ValueInfoProto, TensorShapeProto, AttributeProto)
from onnx import (helper, numpy_helper)
from onnx.mapping import (TENSOR_TYPE_TO_NP_TYPE, NP_TYPE_TO_TENSOR_TYPE)

from skl2onnx.helpers.onnx_helper import (select_model_inputs_outputs,
                                          enumerate_model_node_outputs,
                                          save_onnx_model)
import onnxruntime as rt
from onnxsim import simplify
from node import OnnxNode


class OnnxGraph():
    def __init__(self, model):
        #TODO:support filename, serialtostring, graphproto
        self._model = onnx.load(model)
        graph = self._model.graph
        self._all_ops_map = {}
        self.all_edges_map = {}
        out_names = [out.name for out in graph.output]
        #TODO:optimizer
        for node in chain(graph.input, graph.initializer, graph.node):
            node = OnnxNode(node)
            self._update_ops_map(node.name, node, False)
            if node.op_type in ['Initializer', 'Placeholder']:
                continue
            for out in node.outputs:
                if out not in out_names:
                    self._update_ops_map(out, node, False)
        for node in graph.node:
            node = OnnxNode(node)
            self._update_edges_map(node, False)
    ###############################################
    #######              Create             #######
    ###############################################
    def add_placeholder(self, name, dtype, shape):
        #TODO:这里要使用self.graph.node.add()添加node再copyfrom
        try:
            dtype = np.dtype(dtype)
        except Exception as e:
            print(e)
            raise RuntimeError(f'{dtype} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')
        elem_type = NP_TYPE_TO_TENSOR_TYPE[dtype]
        node = self._model.graph.input.add()
        node.CopyFrom(helper.make_tensor_value_info(name, elem_type, shape))
        ph = OnnxNode(node)
        self._update_ops_map(ph.name, ph, False)
        return ph

    def add_initializer(self, name, value):
        node = self._model.graph.initializer.add()
        node.CopyFrom(helper.make_tensor(name,
                                        NP_TYPE_TO_TENSOR_TYPE[value.dtype],
                                        value.shape,
                                        value.flatten().tolist()))
        init = OnnxNode(node)
        self._update_ops_map(init.name, init, False)
        return init

    def add_node(self, name, op_type, attrs={}):
        node = self._model.graph.node.add()
        node.CopyFrom(helper.make_node(op_type = op_type,
                                    inputs = ['Null'],
                                    outputs = ['Null'],
                                    name=name,
                                    **attrs))
        node = OnnxNode(node)
        self._update_ops_map(node.name, node, False)
        return node

    def insert_node(self, anchor, dst, index=0, mode='after'):
        src = self._all_ops_map.get(anchor)
        assert src != None, f'There is no node.name={anchor} in graph, please check it by yourself.'

        if mode == 'after':
            if len(dst.inputs) > 1:
                raise RuntimeError('Only support single input Node, maybe you can use graph.connection')
            while dst.outputs:
                dst.outputs.pop()
            dst.outputs.append(src.outputs[index])
            dst.inputs[0] = f'{src.name}_{dst.name}'
            src.outputs[index] = f'{src.name}_{dst.name}'
        elif mode == 'before':
            if len(dst.outputs) > 1:
                raise RuntimeError('Only support single output Node, maybe you can use graph.connection')
            while dst.inputs:
                dst.inputs.pop()
            dst.inputs.append(src.inputs[index])
            dst.outputs[0] = f'{dst.name}/{src.name}'
            src.inputs[index] = f'{dst.name}/{src.name}'
        else:
            raise ValueError(f'Only support mode="after" or mode="before", but got {mode}')

        return self

    ###############################################
    #######            Retrieve             #######
    ###############################################
    def get_nodes(self, op_type):
        return {node for node in self._all_ops_map.values() if node.op_type == op_type}

    def __getitem__(self, key):
        ret = self._all_ops_map.get(key)
        if ret is None:
            raise ValueError(f'{key} dose not exist in graph')
        return ret

    ###############################################
    #######             Update              #######
    ###############################################
    def __setitem__(self, key, value):
        # TODO: 仅nodeproto的替换，且要求替换前后的输入输出数量必须一致
        # 对应init和ph的修改不支持，可以先获取node再用node方法修改
        # try:
        #     node = OnnxNode(value)
        # except Exception as e:
        #     print(e)
        #     raise RuntimeError(f'{value} is wrong')
        # if not isinstance(value, NodeProto):
        #     raise RuntimeError(f'Only support change NodeProto, but {key} is exclude')
        src = self._all_ops_map.pop(key)
        self._del_node(src)
        value.inputs = src.inputs
        value.outputs = src.outputs
        self._all_ops_map[key] = value

    ###############################################
    #######             Delete              #######
    ###############################################
    def del_node(self, name, maps=None, auto_connection=True):
        src = self._all_ops_map.pop(name)
        if len(src.inputs) != 1 and len(src.outputs) != 1:
            raise RuntimeError('fuzz action')
        appendix = self._all_ops_map[self.all_edges_map[name][0]]
        for idx, in_name in enumerate(appendix.inputs):
            if in_name == src.outputs[0]:
                break
        appendix.set_input(idx, src.inputs[0])
        self._del_node(src)

    def _del_node(self, node):
        if node.op_type == 'Initializer':
            self._model.graph.initializer.remove(node.node)
        elif node.op_type == 'Placeholder':
            self._model.graph.input.remove(node.node)
        else:
            self._model.graph.node.remove(node.node)
    ###############################################
    #######         graph operation         #######
    ###############################################
    def connection(self, previous, out_idx, behind, in_idx):
        if previous not in self._all_ops_map or behind not in self._all_ops_map:
            raise ValueError(f'{previous} or {behind} not in graph')
        prev = self._all_ops_map[previous]
        beh = self._all_ops_map[behind]
        out_len, in_len = len(out_idx), len(in_idx)
        if (0 in (out_len, in_len)) or \
            ((out_len != in_len) and (1 not in (out_len, in_len))):
            raise RuntimeError(f'It is fuzzy to connect between {out_idx} and {in_idx}')
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

    def run(self, data):
        model = self._model.SerializeToString()
        return self._run(model, data)

    def _run(self, model, datas):
        sess = rt.InferenceSession(model)
        inputs = [inode.name for inode in sess.get_inputs()]
        outputs = [out.name for out in sess.get_outputs()]
        ret = sess.run(outputs, {name: data for name, data in zip(inputs, datas)})
        return ret

    def dump(self, data, path='dump', outputs=None):
        outs = [name for name in enumerate_model_node_outputs(self._model)]
        new_model = select_model_inputs_outputs(self._model, outs)
        new_model_byte = new_model.SerializeToString()
        arrs = self._run(new_model_byte, data)
        idx = 0
        for node in self._model.graph.node:
            for i, output in enumerate(node.output):
                fname = f'{node.op_type}_{node.name}_output{i}_{round(time.time() * 1000000)}.npy'
                np.save(os.path.join(path, fname), arrs[idx])
                idx += 1

    def simplify(self, inplace, kwargs={}):
        model_sim, check = simplify(self._model, **kwargs)
        assert check, "Simplified ONNX model could not be validated"
        if inplace:
            self._model = model_sim
            return self
        else:
            return model_sim 

    ###############################################
    #######       assistant operation       #######
    ###############################################
    #TODO: 接口设计需要更合理，主要是name和rewrite的设计
    def _update_ops_map(self, name, node, rewrite=True):
        if (name in self._all_ops_map) and (not rewrite):
            raise RuntimeError(f'{name} already exists in the NodeProto')
        self._all_ops_map[name] = node

    def _update_edges_map(self, node, rewrite=True):
        if (node.name in self.all_edges_map) and (not rewrite):
            raise RuntimeError(f'{node.name} already exists in the {node.op_type}')
        for in_idx in node.inputs:
            in_name = self._all_ops_map[in_idx].name
            self.all_edges_map.setdefault(in_name, []).append(node.name)
