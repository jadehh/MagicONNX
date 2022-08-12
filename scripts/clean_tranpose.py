import sys
import numpy as np
import copy
from magiconnx import OnnxGraph, OnnxNode
from magiconnx.optimize.optimizer_manager import OptimizerManager


def clean_softmax_transpose(graph):
    for softmax_node in graph.get_nodes(op_type="Softmax"):
        # del next two transpose node
        next_node = graph.get_next_nodes(softmax_node.name)[0]
        next_next_node = graph.get_next_nodes(next_node.name)[0]
        if next_node.op_type == "Transpose" and next_next_node.op_type == "Transpose":
            if next_node["perm"] == next_next_node["perm"]:
                graph.del_node(next_node.name)
                graph.del_node(next_next_node.name)


def split_three_parts(graph, start_matmul):
    # build three branchs
    matmul_node_base = graph[start_matmul]
    add_node_base = graph.get_next_nodes(matmul_node_base.name)[0]
    reshape_node_base = graph.get_next_nodes(add_node_base.name)[0]
    transpose_node_base = graph.get_next_nodes(reshape_node_base.name)[0]

    matmul_node_base_weight = graph[matmul_node_base.inputs[1]].value
    add_node_base_weight = graph[add_node_base.inputs[0]].value
    reshape_node_base_weight = graph[reshape_node_base.inputs[1]].value
    transpose_node_base_weight = graph.get_next_nodes(reshape_node_base.name)[0]
    last_node_lists = graph.get_next_nodes(transpose_node_base.name)

    # build branch
    def build_branch_nodes(base_node, base_node_weight, split_axixs, is_shape_node=False, num_branch=3):
        node_list = []
        op_type = base_node.op_type
        if not is_shape_node:
            target_shape = list(base_node_weight.shape)
            target_shape[split_axixs] = target_shape[split_axixs] // num_branch
            target_shape.insert(split_axixs, num_branch)
            base_node_weight = base_node_weight.reshape(target_shape)
            splited_node_weights = np.split(base_node_weight, num_branch, axis=split_axixs)
        for idx in range(num_branch):
            node_name = "{}_{}".format(base_node.name, idx)
            init_name = "{}_init".format(node_name)
            if not is_shape_node:
                node_weight = splited_node_weights[idx].squeeze(split_axixs)
            else:
                node_weight = base_node_weight.copy()
                node_weight = np.delete(node_weight, split_axixs)

            added_init = graph.add_initializer(
                init_name,
                node_weight
            )
            added_node = graph.add_node(
                node_name,
                op_type,
                outputs=["{}_output".format(node_name)]
            )
            added_node.inputs.append(added_init.name)
            node_list.append(added_node)
        return node_list

    def duplicate_transpose_nodes(base_node, perm=[0, 2, 1, 3], num=3):
        node_list = []
        for idx in range(num):
            node = graph._model.graph.node.add()
            node.CopyFrom(base_node._node)
            node = OnnxNode(node)
            node.name = "{}_{}".format(base_node.name, idx)
            node.inputs = ["Null"]
            node.outputs = ["{}_output".format(node.name)]
            node['perm'] = perm
            node_list.append(node)
        return node_list
    matmul_node_lists = build_branch_nodes(matmul_node_base, matmul_node_base_weight, split_axixs=1)
    add_node_lists = build_branch_nodes(add_node_base, add_node_base_weight, split_axixs=0)
    reshape_node_lists = build_branch_nodes(reshape_node_base, reshape_node_base_weight, split_axixs=2, is_shape_node=True)
    transpose_node_lists = duplicate_transpose_nodes(transpose_node_base)

    # connect branch
    input_node = graph[matmul_node_base.inputs[0]]
    for idx in range(3):
        matmul_node_lists[idx].inputs[0] = input_node.outputs[0]
        add_node_lists[idx].inputs[0] = matmul_node_lists[idx].outputs[0]
        reshape_node_lists[idx].inputs[0] = add_node_lists[idx].outputs[0]
        transpose_node_lists[idx].inputs[0] = reshape_node_lists[idx].outputs[0]

    for idx, last_node in enumerate(last_node_lists):
        for next_node in graph.get_next_nodes(last_node.name):
            # merge continuous tranpose nodes
            if next_node.op_type == "Transpose":
                assert transpose_node_lists[idx]["perm"] == [0, 2, 1, 3] and \
                    next_node["perm"] == [0, 1, 3, 2]
                transpose_node_lists[idx]["perm"] = [0, 2, 3, 1]
                next_next_node = graph.get_next_nodes(next_node.name)[0]
                next_next_node.inputs[0] = transpose_node_lists[idx].outputs[0]
                onnx_graph.del_node(next_node.name, auto_connection=False)
                continue

            if len(next_node.inputs) == 1:
                next_node.inputs[0] = transpose_node_lists[idx].outputs[0]
            else:
                for input_idx, next_node_inputs in enumerate(next_node.inputs):
                    if graph[next_node_inputs].name == last_node.name:
                        next_node.inputs[input_idx] = transpose_node_lists[idx].outputs[0]

    # clean ori nodes
    onnx_graph.del_node(matmul_node_base.name, auto_connection=False)
    onnx_graph.del_node(add_node_base.name, auto_connection=False)
    onnx_graph.del_node(reshape_node_base.name, auto_connection=False)
    onnx_graph.del_node(transpose_node_base.name, auto_connection=False)
    for last_node in last_node_lists:
        onnx_graph.del_node(last_node.name, auto_connection=False)


def split_model(graph):
    for div_node in graph.get_nodes(op_type="Div"):
        # check patter: Div->Mul->Add->MatMul->Add->Reshape
        try:
            next_node1 = graph.get_next_nodes(div_node.name)[0]
            next_node2 = graph.get_next_nodes(next_node1.name)[0]
            next_node3 = graph.get_next_nodes(next_node2.name)[0]
            next_node4 = graph.get_next_nodes(next_node3.name)[0]
            next_node5 = graph.get_next_nodes(next_node4.name)[0]
        except:
            continue
        if next_node1.op_type == "Mul" and \
           next_node2.op_type == "Add" and \
           next_node3.op_type == "MatMul" and \
           next_node4.op_type == "Add" and \
           next_node5.op_type == "Reshape":
            split_three_parts(graph, start_matmul=next_node3.name)


if __name__ == '__main__':
    onnx_graph = OnnxGraph(sys.argv[1])
    save_path = sys.argv[2]
    deep_opt = False
    if len(sys.argv) > 3:
        deep_opt = bool(sys.argv[3])
    print("Remove continuous transpose nodes after softmax node...")
    clean_softmax_transpose(onnx_graph)
    if deep_opt:
        print("Split block into three parts...")
        split_model(onnx_graph)
    onnx_graph.save(save_path)
    print("Succeed!")
