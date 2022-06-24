import numpy as np
import re
from ..base_optimizer import BaseOptimizer


HIDDEN_NUM = 768
DIV_CONST = 8


class BertBigKernelOptimizer(BaseOptimizer):
    """
    @des        modify bert onnx model to adapt bigkernel fusion pass:
                  1. insert reshape node
                  2. fix output shape for specified reshape node
                  3. fix transpose perm att && insert transpose node
                  4. fix div node
                  5. broadcast input value for input mask node
    """
    def insert_reshape_node(self, node_list):
        for idx, node_name in enumerate(node_list):
            init_name = 'Reshape_1_{}_shape'.format(idx)
            init = self._graph.add_initializer(init_name,
                                               np.array([-1, HIDDEN_NUM]))
            inserted_node = self._graph.add_node(
                'Reshape_1_{}'.format(idx),
                'Reshape'
            )
            self._graph.insert_node(node_name, inserted_node)
            inserted_node.inputs.append(init_name)

    def insert_last_reshape(self, seq):
        # get last mul node
        mul_num = -1
        last_mul_name = ''
        for mul_node in self._graph.get_nodes(op_type="Mul"):
            mul_name = mul_node.name
            temp_num = int(mul_name.split('_')[-1])
            if temp_num > mul_num:
                mul_num = temp_num
                last_mul_name = mul_name

        init_name = 'Reshape_last_shape'
        init = self._graph.add_initializer(init_name, np.array([-1, seq, HIDDEN_NUM]))
        inserted_node = self._graph.add_node(
            'Reshape_last',
            'Reshape'
        )
        self._graph.insert_node(last_mul_name, inserted_node)
        inserted_node.inputs.append(init_name)

    def fix_reshape_node(self, node_list):
        for idx, node_name in enumerate(node_list):
            shape_value = self._graph[self._graph[node_name].inputs[1]].value
            if sum(shape_value.shape) == 3:
                self._graph[self._graph[node_name].inputs[1]].value = np.array(
                    [shape_value[0] * shape_value[1], shape_value[2]])

    def modify_transpose_node(self, node_list):
        for idx, node_name in enumerate(node_list):
            node = self._graph[node_name]
            node['perm'] = np.array([0, 2, 1, 3])
            inserted_node = self._graph.add_node(
                'Transpose_x_{}'.format(idx),
                'Transpose',
                {"perm": [0, 1, 3, 2]}
            )
            self._graph.insert_node(node_name, inserted_node)

    def modify_div_node(self, node_list):
        for idx, node_name in enumerate(node_list):
            node = self._graph[node_name]
            node_num = node_name.split("_")[-1]
            node.op_type = "Mul"
            node.name = "bert_Mul_" + node_num
            for input_name in node.inputs:
                input_node = self._graph[input_name]
                if input_node.op_type == "Initializer":
                    val = input_node.value
                    if idx == 0:
                        # first init global value
                        DIV_CONST = val
                    if val == DIV_CONST:
                        input_node.value = np.array(1/val, dtype="float32")

    def find_transpose_node(self, start_node_name):
        # pattern: matmul->add->reshape->transpose
        next_node = self._graph[start_node_name]
        for idx in range(3):
            next_node = self._graph.get_next_nodes(next_node.name)[0]
        assert next_node.op_type == "Transpose", \
            "Pattern not match: matmul->add->reshape->transpose"
        return next_node.name

    def find_reshape_node(self, start_node_name):
        # pattern: transpose->matmul->div->add->softmax->matmul->transpose->reshape
        next_node = self._graph[start_node_name]
        for idx in range(7):
            next_node = self._graph.get_next_nodes(next_node.name)[0]
        assert next_node.op_type == "Reshape", \
            "Pattern not match: transpose->matmul->div->add->softmax->matmul->transpose->reshape"
        return next_node.name

    def find_div_node(self, start_node_name):
        # pattern: transpose->matmul->div
        next_node = self._graph[start_node_name]
        for idx in range(2):
            next_node = self._graph.get_next_nodes(next_node.name)[0]
        assert next_node.op_type == "Div", \
            "Pattern not match: transpose->matmul->div"
        return next_node.name

    def get_bigkernel_part(self):
        with_kernel = False
        nodes_info = {
            "Mul": [],
            "Reshape": [],
            "Transpose": [],
            "Div": []
        }
        mul_nodes = self._graph.get_nodes(op_type="MatMul") + self._graph.get_nodes(op_type="Mul")
        for mul_node in mul_nodes:
            mul_name = mul_node.name
            # judge whether the mul_node match the big kernel
            next_add_node = self._graph.get_next_nodes(mul_name)[0]
            kernel_starts = self._graph.get_next_nodes(next_add_node.name)
            if next_add_node.op_type != "Add" or len(kernel_starts) != 4:
                continue
            kernel_start_mul1 = kernel_starts[0]
            kernel_start_mul2 = kernel_starts[1]
            kernel_start_mul3 = kernel_starts[2]
            kernel_add = kernel_starts[3]
            if kernel_start_mul1.op_type != "MatMul" or \
               kernel_start_mul2.op_type != "MatMul" or \
                   kernel_start_mul3.op_type != "MatMul" or \
                       kernel_add.op_type != "Add":
                continue

            nodes_info['Mul'].append(mul_name)
            transpose_node_name = self.find_transpose_node(kernel_start_mul2.name)
            nodes_info['Transpose'].append(transpose_node_name)
            nodes_info['Reshape'].append(self.find_reshape_node(transpose_node_name))
            nodes_info['Div'].append(self.find_div_node(transpose_node_name))

            with_kernel = True
        return with_kernel, nodes_info

    def broadcast_input_mask(self):
        def get_next_mul_node(input_node, max_depth=20):
            next_node = input_node
            cur_depth = 1
            while next_node.op_type != "Mul" and cur_depth < max_depth:
                next_node = self._graph.get_next_nodes(next_node.name)[0]
                cur_depth += 1
            return next_node

        for input_name in self._graph.inputs:
            input_node = self._graph[input_name]
            next_mul_node = get_next_mul_node(input_node)
            if next_mul_node.op_type != "Mul":
                continue
            if len(self._graph.get_next_nodes(next_mul_node.name)) > 2:
                # broadcast dim2 for input mask node
                inserted_node = self._graph.add_node(
                    'Expand_mask_input',
                    'Expand'
                )
                self._graph.insert_node(next_mul_node.name, inserted_node, mode='before')
                input_shape = BertBigKernelOptimizer.convert_str2numlist(input_node.shape)
                assert len(input_shape) == 2, 'input mask shape: [batch_size, seq]'
                expand_value = [input_shape[0], 1, input_shape[1], input_shape[1]]
                expand_init = self._graph.add_initializer(
                    'Expand_mask_expand_shape',
                    np.array(expand_value))
                inserted_node.inputs.append('Expand_mask_expand_shape')
                break

    def optimize(self, graph):
        self._graph = graph
        try:
            with_kernel, nodes_info = self.get_bigkernel_part()
        except:
            return graph, False
        if not with_kernel:
            return self._graph, with_kernel
        input_shape = self._graph[self._graph.inputs[0]].shape
        input_seq = BertBigKernelOptimizer.convert_str2numlist(input_shape)
        self.insert_reshape_node(nodes_info.get('Mul'))
        self.fix_reshape_node(nodes_info.get('Reshape'))
        self.modify_transpose_node(nodes_info.get('Transpose'))
        self.modify_div_node(nodes_info.get('Div'))
        self.insert_last_reshape(input_seq[-1])
        self.broadcast_input_mask()
        return self._graph, with_kernel

    @staticmethod
    def convert_str2numlist(input_str):
        # sample: '[1, 2, 3]'-->[1, 2, 3]
        input_str = input_str[1:-1].replace(' ', '')
        input_seq = [int(v) for v in re.split(',', input_str)]
        return input_seq
