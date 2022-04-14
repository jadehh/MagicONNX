import numpy as np
from ..base_optimizer import BaseOptimizer


INT64 = 7
INT32 = 6
MAX_VALUE_INT32 = 2147483647
MIN_VALUE_INT32 = -2147483648


class Int64ToInt32Optimizer(BaseOptimizer):
    """
    @des        convert all values with int64 type to int32, inclued:
                  1. convert all cast nodes from int64 to int32
                  2. convert all constant shape values from int64 to int32
                  3. convert all constant values from int64 to int32
                  4. convert all initializer values from int64 to int32
                  5. insert cast node after shape nodes(default value output type is int64)
    """

    def optimize(self, graph):
        flag = False
        flag |= self._convert_cast_nodes(graph)
        flag |= self._transfer_constantofshape(graph)
        flag |= self._convert_all_constants(graph)
        flag |= self._convert_all_initializers(graph)
        flag |= self._insert_cast_after_shape(graph)
        return graph, flag

    @staticmethod
    def _convert_cast_nodes(graph):
        flag = False
        cast_nodes = graph.get_nodes(op_type='Cast')
        for node in cast_nodes:
            if node['to'] == INT64:
                node._node.attribute[0].i = INT32
                flag = True
        return flag

    @staticmethod
    def _transfer_constantofshape(graph):
        flag = False
        constant_nodes = graph.get_nodes("ConstantOfShape")
        for node in constant_nodes:
            if node.attrs['value'].t.data_type == INT64:
                node.attrs['value'].t.data_type = INT32
                flag = True
        return flag

    @staticmethod
    def _value_to_int32(node):
        node_value = node.value.copy()
        if (node_value > MAX_VALUE_INT32).any():
            node_value[node_value > MAX_VALUE_INT32] = MAX_VALUE_INT32
        if (node_value < MIN_VALUE_INT32).any():
            node_value[node_value < MIN_VALUE_INT32] = MIN_VALUE_INT32
        node.value = node_value.astype(np.int32)
        return node

    @staticmethod
    def _convert_all_constants(graph):
        flag = False
        constant_nodes = graph.get_nodes('Constant')
        for node in constant_nodes:
            if np.issubdtype(node.value.dtype, np.int64):
                node = Int64ToInt32Optimizer._value_to_int32(node)
                flag = True
        return flag

    @staticmethod
    def _convert_all_initializers(graph):
        flag = False
        initializer_nodes = graph.get_nodes('Initializer')
        for node in initializer_nodes:
            if np.issubdtype(node.value.dtype, np.int64):
                node = Int64ToInt32Optimizer._value_to_int32(node)
                flag = True
        return flag

    @staticmethod
    def _insert_cast_after_shape(graph):
        flag = False

        def insert_cast_node(graph, before_node, node_name, dtype=INT32):
            cast_node = graph.add_node(
                node_name,
                'Cast',
                {'to': dtype}
            )
            graph.insert_node(before_node, cast_node, mode='after')

        shape_nodes = graph.get_nodes("Shape")
        for node in shape_nodes:
            node_name = node.name
            insert_name = 'expand_after_{}'.format(node_name)
            insert_cast_node(graph, node_name, insert_name)
            flag = True
        return flag
