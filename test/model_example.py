
import onnx
from onnx import (helper, TensorProto)
from onnx.onnx_ml_pb2 import ModelProto
from core import OnnxModel
import unittest

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


def print_message(model):
    message = (f'model.ir_version = {model.ir_version}\n'
               f'model.opset_import = {model.opset_import}\n'
               f'model.producer_name = {model.producer_name}\n'
               f'model.producer_version = {model.producer_version}\n'
               f'model.domain = {model.domain}\n'
               f'model.model_version = {model.model_version}\n'
               f'model.graph = {model.graph}\n')
    print(message)


if __name__ == '__main__':
    graph = create_graph()
    # model = OnnxModel('./test_resize.onnx')

    # print_message(model)
