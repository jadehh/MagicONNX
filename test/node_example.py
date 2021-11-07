import numpy as np
import onnx
from onnx import helper
import sys
sys.path.append('..')
from core import OnnxNode


def create_all_nodes():
    value_info = helper.make_tensor_value_info(
                    "myvalue",
                    onnx.TensorProto.FLOAT,
                    [2, 2],
                    doc_string='test for ValueInfoProto')
    tensor = helper.make_tensor(
                    "mytensor",
                    onnx.TensorProto.INT64,
                    [2, 3],
                    [1, 2, 3, 4, 5, 6])

    node = helper.make_node(
        'ArgMin',
        inputs=['data'],
        outputs=['result'],
        name='mynode',
        axis=1,
        keepdims=1,
    )
    return node, value_info, tensor


def test_common(node):
    print(node)                         # 打印 node 节点详细信息
    node.node                           # 获取 node 对应onnx的原始节点
    print(f'name = {node.name}\top_type = {node.op_type}\tdoc_string = {node.doc_string}')
    node.name = 'Arbitrary string'      # 修改 node 节点名称
    try:
        node.op_type = 'op_type'        # 修改 node 节点op_type
    except Exception as e:
        print(e)
    node.doc_string = 'doc_string'
    print(f'after change\nname = {node.name}\top_type = {node.op_type}\tdoc_string = {node.doc_string}')


def test_for_placeholder(node):
    print(f'placeholder\ndtype = {node.dtype}\tshape = {node.shape}')
    node.dtype = 'int32'
    node.shape = [1,5,10,10]
    print(f'after change\ndtype = {node.dtype}\tshape = {node.shape}')


def test_for_initializer(node):
    print(f'initializer\nvalue = {node.value}')
    node.value = np.array([2, 3])
    print(f'after change\nvalue = {node.value}')


def test_for_nodeproto(node):
    print(f'nodeproto\ninputs = {node.inputs}\toutputs = {node.outputs}\tattrs = {node.attrs}')
    node.inputs = ['in_1', 'in_2']      # 修改输入name列表
    node.set_input(0, 'in_x')           # 修改第0个输入name
    node.outputs = ['in_1', 'in_2']
    node.set_output(0, 'in_y')
    node['attr_1'] = 'fake attribute'
    print(f'after change\ninputs = {node.inputs}\toutputs = {node.outputs}\tattrs = {node.attrs}')


if __name__ == '__main__':
    model = onnx.load('layernorm.onnx')
    model = onnx.load('dynamic.onnx')
    placeholder = model.graph.input[0]
    initializer = model.graph.initializer[0]
    node_proto = model.graph.node[0]
    # node_proto, placeholder, initializer = create_all_nodes()
    ph, init, node = OnnxNode(placeholder), OnnxNode(initializer), OnnxNode(node_proto)
    for node in (ph, init, node):
        test_common(node)
    test_for_placeholder(ph)
    test_for_initializer(init)
    test_for_nodeproto(node)
