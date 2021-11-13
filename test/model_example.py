
import onnx
from onnx import (helper, TensorProto)
from onnx.onnx_ml_pb2 import ModelProto
from magiconnx import OnnxGraph
from copy import deepcopy
import numpy as np

def create_graph():
    # create graph according to layernorm
    x = helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [20,5,10,10])
    y = helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [20,5,10,10])
    pow_const = helper.make_tensor('16', onnx.TensorProto.FLOAT, [], [2])
    add_const = helper.make_tensor('10', onnx.TensorProto.FLOAT, [], [1e-5])

    ReduceMean_0 = helper.make_node(
        'ReduceMean',
        inputs=['x'],
        outputs=['4'],
        name='ReduceMean_0',
        axes=[-3, -2, -1]
    )

    sub_1 = helper.make_node(
        'Sub',
        inputs=['x', '4'],
        outputs=['5'],
        name='sub_1'
    )

    Cast_2 = helper.make_node(
        'Cast',
        inputs=['5'],
        outputs=['6'],
        name='Cast_2',
        to=1
    )

    Pow_3 = helper.make_node(
        'Pow',
        inputs=['6', '16'],
        outputs=['8'],
        name='Pow_3'
    )

    ReduceMean_4 = helper.make_node(
        'ReduceMean',
        inputs=['8'],
        outputs=['9'],
        name='ReduceMean_4',
        axes=[-3, -2, -1]
    )

    Add_6 = helper.make_node(
        'Add',
        inputs=['9', '10'],
        outputs=['11'],
        name='Add_6'
    )

    Sqrt_7 = helper.make_node(
        'Sqrt',
        inputs=['11'],
        outputs=['12'],
        name='Sqrt_7'
    )

    Div_8 = helper.make_node(
        'Div',
        inputs=['5', '12'],
        outputs=['y'],
        name='Div_8'
    )

    graph = helper.make_graph(
        nodes=[ReduceMean_0, sub_1, Cast_2, Pow_3, ReduceMean_4, Add_6, Sqrt_7, Div_8],
        name='my-model',
        inputs=[x],
        outputs=[y],
        initializer=[pow_const, add_const]
    )
    model_def = onnx.helper.make_model(graph, producer_name='HJ-ArgMin-onnx')
    model_def.opset_import[0].version = 11
    onnx.save(model_def, "./test_argmin_case.onnx")
    return graph


def create(path):
    graph = OnnxGraph(path)
    ph = graph.add_placeholder('dummy_input', 'int32', [2, 3, 4])
    init = graph.add_initializer('dummy_init', np.array([[2, 3, 4]]))
    add = graph.add_node('dummy_add', 'Add')
    add.inputs = ['dummy_input', 'dummy_init']
    add.outputs = ['add_out']
    graph.save('case1.onnx')
    argmax = graph.add_node('dummy_ArgMax',
                          'ArgMax',
                          {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
    graph.insert_node('dummy_add', argmax, mode='before')
    graph.save('case2.onnx')

def retrieve(path):
    graph = OnnxGraph(path)
    adds = graph.get_nodes("Add")
    for add in adds:
        print(f'add.name = {add.name}')
    inits = graph.get_nodes("Initializer")
    phs = graph.get_nodes("Placeholder")
    add_6 = graph['Add_6']
    print(f'add_6.inputs = {add_6.inputs}')

def update(path):
    graph = OnnxGraph(path)
    argmax = graph.add_node('dummy_ArgMax',
                      'ArgMax',
                      {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
    graph['Cast_2'] = argmax
    graph.save('case3.onnx')

def delete(path):
    graph = OnnxGraph(path)
    graph.del_node('Cast_2')
    graph.save('case4.onnx')

def test_connection(path):
    graph = OnnxGraph(path)
    graph.connection('Div_8', 0, 'Add_10', 1)
    graph.save('case5.onnx')

def test_run_dump(path, data):
    graph = OnnxGraph(path)
    graph.dump([data])
    ret = graph.run([data])
    for output in ret:
        print(output.shape)

if __name__ == '__main__':
    # graph = create_graph()

    # test for create
    create('layernorm.onnx')
    # test for Retrieve
    retrieve('layernorm.onnx')
    # test for Update
    update('layernorm.onnx')
    # test for Delete
    delete('layernorm.onnx')

    # test for graph operation
    test_connection('layernorm.onnx')
    graph = OnnxGraph('layernorm.onnx')
    print(graph)
    print(graph.inputs)
    print(graph.outputs)
    print(graph.graph)

    data = np.random.randn(20, 5, 10, 10).astype(np.float32)
    test_run_dump('layernorm.onnx', data)

    graph.simplify(True).save('case6.onnx')
