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

from node import OnnxNode

class OnnxGraph():
    def __init__(self, model):
        #TODO:support filename, serialtostring, graphproto
        self._model = onnx.load(model)
        graph = self._model.graph
        self._all_ops_map = {}
        self.all_edges_map = {}
        #TODO:optimizer
        for node in chain(graph.input, graph.initializer, graph.node, graph.output):
            node = OnnxNode(node)
            self._update_ops_map(node.name, node, False)
            if node.op_type in ['Initializer', 'Placeholder']:
                continue
            for out in node.outputs:
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
        placeholder = helper.make_tensor_value_info(name, elem_type, shape)
        ph = OnnxNode(placeholder)
        self._update_ops_map(ph.name, ph, False)
        return ph

    def add_initializer(self, name, value):
        initializer = helper.make_tensor(name,
                                        NP_TYPE_TO_TENSOR_TYPE[value.dtype],
                                        value.shape,
                                        value.flatten().tolist())
        init = OnnxNode(initializer)
        self._update_ops_map(init.name, init, False)
        return init

    def add_node(self, name, op_type, attrs):
        node_proto = helper.make_node(op_type = op_type,
                                    inputs = [],
                                    outputs = [],
                                    name=name,
                                    **attrs)
        node = OnnxNode(node_proto)
        self._update_ops_map(node.name, node, False)
        return node

    def insert_node(self, anchor, target, index=0, mode='after'):
        src = self._all_ops_map.get(anchor)
        assert src != None, f'There is no node.name={anchor} in graph, please check it by yourself.'

        dst = OnnxNode(target)
 
        if mode == 'after':
            if len(target.input) > 1:
                raise RuntimeError('Only support single input Node, maybe you can use graph.connection')
            while dst.output:
                dst.output.pop()
            target.output.append(src.output[index])
            dst.input[0] = f'{src.name}_{dst.name}'
            src.output[index] = f'{src.name}_{dst.name}'
        elif mode == 'before':
            if len(target.output) > 1:
                raise RuntimeError('Only support single output Node, maybe you can use graph.connection')
            while dst.input:
                dst.input.pop()
            dst.input.append(src.input[index])
            dst.output[0] = f'{dst.name}/{src.name}'
            src.input[index] = f'{dst.name}/{src.name}'
        else:
            raise ValueError(f'Only support mode="after" or mode="before", but got {mode}')

        #TODO:这里可能要改，应该是insert，而不是append
        self._graph.node.append(target)
        self._update_ops_map(dst.name, dst, False)
        return self

    ###############################################
    #######            Retrieve             #######
    ###############################################
    def get_nodes(self, op_type):
        return {node for node in self._all_ops_map.values() if node.op_type == op_type}

    @property
    def graph(self):
        return self._graph

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
        try:
            node = OnnxNode(value)
        except Exception as e:
            print(e)
            raise RuntimeError(f'{value} is wrong')
        if not isinstance(value, NodeProto):
            raise RuntimeError(f'Only support change NodeProto, but {key} is exclude')
        self._all_ops_map[key] = value

    ###############################################
    #######             Delete              #######
    ###############################################
    def del_node(self, name, maps=None, auto_connection=True):
        pass

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
        return helper.printable_graph(self._graph)

    @property
    def inputs(self):
        return self._graph.input

    @property
    def outputs(self):
        return self._graph.output
    
    def save(self, path):
        onnx.save(self._model, path)

    def run(self, data):
        model = self._model.Seri()
        return self._run(model, data)

    def _run(self, model, data):
        pass

    def dump(self, data, path='dump', outputs=None):
        outs = [name for name in enumerate_model_node_outputs(self._model)]
        new_model = select_model_inputs_outputs(self._model, outs)
        new_model_byte = new_model.Seri()
        arrs = self._run(new_model_byte, data)
        idx = 0
        for node in self._model.graph.node:
            for i, output in enumerate(node.output):
                fname = f'{node.op_type}_{node.name}_output{i}_{round(time.time() * 1000000)}.npy'
                np.save(os.path.join(path, fname), arrs[idx])
                idx += 1

    def simplify(self, inplace, **kwargs):
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

if __name__ == '__main__':
    graph = OnnxGraph('layernorm.onnx')
    ph = graph.add_placeholder('dummy_input', 'int32', [2, 3, 4])
    init = graph.add_initializer('dummy_init', np.array([[2, 3, 4]]))
    node = graph.add_node('dummy_ArgMax',
                          'ArgMax',
                          {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
    adds = graph.get_nodes("Add")
    add_6 = graph['Add_6']

