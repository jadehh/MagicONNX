import sys
import numpy as np
from magiconnx import OnnxGraph


def fold_exp(graph):
    for exp_node in graph.get_nodes(op_type="Exp"):
        # cal constant vlaue for clip-exp nodes
        clip_node = graph[exp_node.inputs[0]]
        mul_node = graph.get_next_nodes(exp_node.name)[0]
        add_node = graph.get_next_nodes(mul_node.name)[0]
        input_value = graph[clip_node.inputs[0]].value
        min_value = graph[clip_node.inputs[1]].value if clip_node.inputs[1] else None
        max_value = graph[clip_node.inputs[2]].value
        input_value = np.clip(input_value, min_value, max_value)
        input_value = np.expand_dims(np.exp(input_value), axis=0)

        # insert constant node
        inserted_init_name = "{}_weight".format(exp_node.name)
        inserted_init = graph.add_initializer(inserted_init_name, input_value)
        mul_node.inputs[1] = inserted_init_name

        # del clip-exp nodes
        graph.del_node(exp_node.name, auto_connection=False)
        graph.del_node(clip_node.name, auto_connection=False)


def merge_add(graph):
    def check_single_pattern(start_node, op_type_list):
        try:
            # get node list: suppose single input single output
            cur_node = start_node
            node_list = []
            for op_type in range(op_type_list):
                cur_node = graph.get_next_nodes(cur_node.name)[0]
                node_list.append(cur_node)
                if cur_node.op_type != op_type:
                    return None
        except:
            return None
        return node_list

    pattern_op_list = ["Add", "Reshape", "Add", "Reshape"]
    for mul_node in graph.get_nodes(op_type="Mul"):
        # check pattern: mul->add->reshape->add->reshape
        node_list = check_single_pattern(mul_node, pattern_op_list)
        if node_list is None:
            continue

        add_node1, reshape_node1, add_node2, reshape_node2 = node_list
        # merge add value
        add_value1 = graph[add_node1.inputs[1]].value
        add_value2 = graph[add_node2.inputs[1]].value
        broad_dim1 = add_value2.shape[1]
        broad_dim2 = add_value1.shape[1]
        add_value1 = np.tile(add_value1, (broad_dim1, 1, 1, 1))
        add_value2 = np.tile(add_value2.squeeze(0), (1, broad_dim2, 1, 1))
        graph[add_node1.inputs[1]].value = add_value1 + add_value2

        # del reshape->add->reshape
        graph.del_node(reshape_node1.name)
        graph.del_node(add_node2.name)
        graph.del_node(reshape_node2.name)


def reconnect_mul_add(graph):
    for softmax_node in graph.get_nodes(op_type="Softmax"):
        pre_node = softmax_node
        while pre_node.op_type != "Matmul":
            pre_node = graph[pre_node.inputs[0]]
        matmul_node = pre_node
        mul_node = graph.get_next_nodes(matmul_node.name)[0]
        add_node = graph.get_next_nodes(mul_node.name)[0]

        # reconnect: matmul->mul->add ==> mul->matmul->add
        add_node.inputs[0] = matmul_node.outputs[0]
        pre_input = matmul_node.inputs[1]
        mul_node.inputs[0] = pre_input
        matmul_node.inputs[1] = mul_node.outputs[0]


if __name__ == '__main__':
    onnx_graph = OnnxGraph(sys.argv[1])
    fold_exp(onnx_graph)
    merge_add(onnx_graph)
    reconnect_mul_add(onnx_graph)
    onnx_graph.save(sys.argv[2])
