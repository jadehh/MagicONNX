import sys
import os
from magiconnx import OnnxGraph


def replace_matmul(onnx_node):
    onnx_node.name = onnx_node.name.replace("Einsum", "MatMul")
    onnx_node.op_type = "MatMul"


def replace_transpose_matmul(graph, onnx_node):
    # Einsum -> MatMul
    replace_matmul(onnx_node)
    # insert transpose(2, 3)
    first_input_node = graph[onnx_node.inputs[0]]
    inserted_node_name = "{}_transpose".format(first_input_node.name)
    inserted_node = graph.add_node(
        inserted_node_name,
        "Transpose",
        attrs={
            "perm": [0, 1, 3, 2]
        }
    )
    graph.insert_node(
        first_input_node.name,
        inserted_node
    )


def replace_einsum(graph):
    for onnx_node in graph.get_nodes(op_type="Einsum"):
        equation_value = onnx_node["equation"].decode("UTF-8")
        if equation_value == "b h n k, b h k v -> b h n v":
            # matmul
            replace_matmul(onnx_node)
        elif equation_value == "b h n k, b h n v -> b h k v":
            # transpose(2, 3) + matmul
            replace_transpose_matmul(graph, onnx_node)
        else:
            raise ValueError("Not supporte: {}".format(equation_value))


if __name__ == '__main__':
    model_path = sys.argv[1]
    save_path = sys.argv[2]
    onnx_graph = OnnxGraph(model_path)
    replace_einsum(onnx_graph)
    onnx_graph.save(save_path)
