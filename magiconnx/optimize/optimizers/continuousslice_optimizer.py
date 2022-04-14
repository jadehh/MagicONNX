import numpy as np
from ..base_optimizer import BaseOptimizer


def get_continuousop(graph, op_type='Slice'):
    """
    @des        get continuous same type nodes
    @param      graph: input onnx graph
                op_type: the same op type
    @return     continuous op list
    """
    all_ops = graph.get_nodes(op_type)
    res = []
    # init mark list, -1 means not marked
    flags = [-1] * len(all_ops)
    for idx, node in enumerate(all_ops):
        # TODO: multi outputs scenes not inclued here
        pre_node = graph[node.inputs[0]]
        if pre_node in all_ops:
            pre_idx = all_ops.index(pre_node)
            if flags[idx] == -1 and flags[pre_idx] == -1:
                # both two nodes are not in list: add new node sequence
                res.append([node, pre_node])
                flags[idx] =  flags[pre_idx] = len(res) - 1
            elif flags[idx] != -1 and flags[pre_idx] == -1:
                # only cur_node in list: append pre node in list
                res[flags[idx]].append(pre_node)
                flags[pre_idx] = flags[idx]
            elif flags[idx] == -1 and flags[pre_idx] != -1:
                # only pre node in list: insert cur node in list
                res_idx = res[flags[pre_idx]].index(pre_node)
                res[flags[pre_idx]].insert(res_idx, node)
                flags[idx] = flags[pre_idx]
            else:
                # both pre node and cur node in list: concat two sequence
                res[flags[idx]] = res[flags[idx]] + res[flags[pre_idx]]
                flags[pre_idx] = flags[idx]
    flags = list(filter(lambda x: x != -1, flags))
    uniq_flags = []
    for f in flags:
        if f not in uniq_flags:
            uniq_flags.append(f)
    # inverted front and back
    return [res[idx][::-1] for idx in uniq_flags]


class ContinuousSliceOptimizer(BaseOptimizer):
    """
    @des        merge two continuous slice ops to one op
                slice1->slice2 ==> merged_slice
    """

    def optimize(self, graph):
        continuous_slice_nodes = get_continuousop(graph)
        flag = False
        for nodes in continuous_slice_nodes:
            if len(nodes) > 2:
                print('[WARNING] Skip: only support for two continuous slice nodes.')
                continue
            slice_node1, slice_node2 = nodes
            graph = ContinuousSliceOptimizer.merge_slicedop(
                graph, slice_node1, slice_node2)
            flag = True
        return graph, flag

    @staticmethod
    def merge_intializers(graph, initializer1, initializer2, merged_name):
        """
        @des        merge two initializers to one initializer
        @param      graph: input onnx graph
                    initializer1: initializer need to be merged
                    initializer2: initializer need to be merged
                    merged_name: name for merged node
        @return     merged initializer
        """

        merged_data = np.append(
            initializer1.value,
            initializer2.value,
        )
        merged_node = graph.add_initializer(name=merged_name, value=merged_data)
        return merged_node

    @staticmethod
    def merge_slicedop(graph, node1, node2):
        """
        @des        merge two node to one node
        @param      graph: input onnx graph
                    slice_node1: slice node1 need to be merged
                    slice_node2: slice node2 need to be merged
        @return     merged graph
        """
        # modify slice_node1 -> merge_node
        node2.inputs[1] = ContinuousSliceOptimizer.merge_intializers(
            graph,
            graph[node1.inputs[1]],
            graph[node2.inputs[1]],
            '{}_1'.format(node1.name)).name
        node2.inputs[2] = ContinuousSliceOptimizer.merge_intializers(
            graph,
            graph[node1.inputs[2]],
            graph[node2.inputs[2]],
            '{}_2'.format(node1.name)).name
        node2.inputs[3] = ContinuousSliceOptimizer.merge_intializers(
            graph,
            graph[node1.inputs[3]],
            graph[node2.inputs[3]],
            '{}_3'.format(node1.name)).name
        node2.inputs[4] = ContinuousSliceOptimizer.merge_intializers(
            graph,
            graph[node1.inputs[4]],
            graph[node2.inputs[4]],
            '{}_4'.format(node1.name)).name
        graph.del_node(node1.name)
        return graph
