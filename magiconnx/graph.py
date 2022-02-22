import time
import os
from itertools import chain
import warnings

import numpy as np
import onnx
from onnx import (helper, GraphProto)
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE

from . import (BaseGraph, OnnxNode, PLACEHOLDER, INITIALIZER)
from .node import OnnxNode
from .utils.log import typeassert


class OnnxGraph(BaseGraph):
    @typeassert(model=str)
    def __init__(self, model):
        # TODO:support filename, serialtostring, graphproto
        self._model_path = model
        self._model = onnx.load(model)
        graph = self._model.graph
        # key = node.name, value = OnnxNode
        self._all_ops_map = {}
        # key = node.name, value = node后继节点name组成的列表
        self._all_edges_map = {}
        self._all_ops_name = set()
        for node in chain(graph.input, graph.initializer, graph.node):
            node = OnnxNode(node)
            self._all_ops_name.add(node.name)
            self._update_ops_map(node.name, node)
            if node.op_type not in [INITIALIZER, PLACEHOLDER]:
                for out in node.outputs:
                    self._update_ops_map(out, node)
        for node in graph.node:
            node = OnnxNode(node)
            self._update_edges_map(node)

    ###############################################
    #######              Create             #######
    ###############################################
    @typeassert(name=str, shape=(tuple, list))
    def add_placeholder(self, name, dtype, shape):
        assert name not in self._all_ops_name, f'The ({name}) has been existed in graph, please change the node.name.'
        try:
            dtype = np.dtype(dtype)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f'{dtype} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')
        elem_type = NP_TYPE_TO_TENSOR_TYPE[dtype]
        node = self._model.graph.input.add()
        node.CopyFrom(helper.make_tensor_value_info(name, elem_type, shape))
        ph = OnnxNode(node)
        self._update_ops_map(ph.name, ph)
        return ph

    @typeassert(name=str, value=np.ndarray)
    def add_initializer(self, name, value):
        assert name not in self._all_ops_name, f'The ({name}) has been existed in graph, please change the node.name.'
        node = self._model.graph.initializer.add()
        node.CopyFrom(helper.make_tensor(name,
                                         NP_TYPE_TO_TENSOR_TYPE[value.dtype],
                                         value.shape,
                                         value.flatten().tolist()))
        init = OnnxNode(node)
        self._update_ops_map(init.name, init)
        return init

    @typeassert(name=str, op_type=str, attrs=dict, inputs=list, outputs=list)
    def add_node(self, name, op_type, attrs={}, inputs=['Null'], outputs=['Null']):
        assert name not in self._all_ops_name, f'The ({name}) has been existed in graph, please change the node.name.'
        node = self._model.graph.node.add()
        node.CopyFrom(helper.make_node(op_type=op_type,
                                       inputs=inputs,
                                       outputs=outputs,
                                       name=name,
                                       **attrs))
        node = OnnxNode(node)
        self._update_ops_map(node.name, node)
        return node

    @typeassert(anchor=str, dst=OnnxNode, index=int, mode=str)
    def insert_node(self, anchor, dst, index=0, mode='after'):
        assert dst.name not in self._all_ops_name, f'The insert node (dst.name={dst.name}) has been existed in graph, it is illegal.'
        assert anchor in self._all_ops_name, f'The anchor node ({anchor}) is not exists in graph, please check it!'
        src = self._all_ops_map.get(anchor)

        if mode == 'after':
            if len(dst.inputs) > 1:
                raise RuntimeError(
                    'Only support single input Node, maybe you can use graph.connection')
            while dst.outputs:
                dst.outputs.pop()
            dst.outputs.append(src.outputs[index])
            dst.inputs[0] = f'{anchor}/{dst.name}'
            src.outputs[index] = f'{anchor}/{dst.name}'
        elif mode == 'before':
            if len(dst.outputs) > 1:
                raise RuntimeError(
                    'Only support single output Node, maybe you can use graph.connection')
            while dst.inputs:
                dst.inputs.pop()
            dst.inputs.append(src.inputs[index])
            dst.outputs[0] = f'{dst.name}/{anchor}'
            src.inputs[index] = f'{dst.name}/{anchor}'
        else:
            raise ValueError(
                f'The mode should be equal to "after" or "before", but got {mode}')

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
    @typeassert(key=str, value=OnnxNode)
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
        self._del_node(src)

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
        for name in self._all_ops_name:
            if self._all_ops_map[name].op_type not in (INITIALIZER, PLACEHOLDER):
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

    def simplify(self, inplace, **kwargs):
        try:
            from onnxsim import simplify
        except ImportError:
            raise RuntimeError(
                "\033[45;1m onnxsim模块导入失败，请检查环境或pip install onnx-simplifier \033[0m")

        model_sim, check = simplify(self._model, **kwargs)
        assert check, "Simplified ONNX model could not be validated"
        if inplace:
            self._model = model_sim
            return self
        else:
            return model_sim

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


if __name__ == '__main__':
    graph = OnnxGraph('doublename.onnx')
    for k, v in graph._all_ops_map.items():
        print(k, v.name)
    # print(graph._all_ops_map)
    print('='*40)
    print(graph['Sub_1'].outputs)
    print('='*40)
    print(graph._all_edges_map)
