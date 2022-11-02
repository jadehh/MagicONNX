import numpy as np
import re
from ..base_optimizer import BaseOptimizer

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
    def insert_reshape_node(self, node_list, hidden_num):
        for idx, node_name in enumerate(node_list):
            init_name = 'Reshape_1_{}_shape'.format(idx)
            init = self._graph.add_initializer(init_name,
                                               np.array([-1, hidden_num]))
            inserted_node = self._graph.add_node(
                'Reshape_1_{}'.format(idx),
                'Reshape'
            )
            self._graph.insert_node(node_name, inserted_node)
            inserted_node.inputs.append(init_name)

    def insert_last_reshape(self, seq, hidden_num):
        # get last mul node
        mul_num = -1
        last_mul_name = ''
        for mul_node in self._graph.get_nodes(op_type="Mul"):
            mul_name = mul_node.name
            temp_num = int(mul_name.split('_')[-1])
            if temp_num > mul_num:
                mul_num = temp_num
                last_mul_name = mul_name

        # check pattern: div->mul->add
        last_mul_node = self._graph[last_mul_name]
        pre_div_node = self._graph[last_mul_node.inputs[0]]
        next_node = self._graph.get_next_nodes(last_mul_name)[0]
        if pre_div_node.op_type != "Div" or \
           next_node.op_type != "Add":
            print("[Error] Pattern not match for {}: div->mul->add".format(last_mul_name))
            return False

        init_name = 'Reshape_last_shape'
        init = self._graph.add_initializer(init_name, np.array([-1, seq, hidden_num]))
        inserted_node = self._graph.add_node(
            'Reshape_last',
            'Reshape'
        )
        self._graph.insert_node(last_mul_name, inserted_node)
        inserted_node.inputs.append(init_name)
        return True

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
            "[Warning] Pattern not match for {}: matmul->add->reshape->transpose".format(start_node_name)
        return next_node.name

    def find_reshape_node(self, start_node_name):
        # pattern: transpose->matmul->div->add->softmax->matmul->transpose->reshape
        next_node = self._graph[start_node_name]
        for idx in range(7):
            next_node = self._graph.get_next_nodes(next_node.name)[0]
        assert next_node.op_type == "Reshape", \
            "[Warning] Pattern not match for {}: transpose->matmul->div->add->softmax->matmul->transpose->reshape".format(start_node_name)
        return next_node.name

    def find_div_node(self, start_node_name):
        # pattern: transpose->matmul->div
        next_node = self._graph[start_node_name]
        for idx in range(2):
            next_node = self._graph.get_next_nodes(next_node.name)[0]
        assert next_node.op_type == "Div", \
            "[Warning] Pattern not match for {}: transpose->matmul->div".format(start_node_name)
        return next_node.name

    def find_add_node(self, start_node_name):
        # pattern: div->add
        next_node = self._graph.get_next_nodes(start_node_name)[0]
        assert next_node.op_type == "Add", \
            "[Warning] Pattern not match for {}: div->add".format(start_node_name)
        return next_node.name

    def get_bigkernel_part(self):
        with_kernel = False
        hidden_num = -1
        nodes_info = {
            "Mul": [],
            "Reshape": [],
            "Add": [],
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
            try:
                transpose_node_name = self.find_transpose_node(kernel_start_mul2.name)
                nodes_info['Transpose'].append(transpose_node_name)
                nodes_info['Reshape'].append(self.find_reshape_node(transpose_node_name))
                nodes_info['Div'].append(self.find_div_node(transpose_node_name))
                nodes_info['Add'].append(self.find_add_node(nodes_info.get('Div')[-1]))
            except Exception as e:
                print(e)
                continue
            with_kernel = True
            hidden_num = self._graph[mul_node.inputs[1]].value.shape[-1]
        return with_kernel, nodes_info, hidden_num

    def broadcast_input_mask(self, start_node_names):
        pre_node_dic = dict()
        for start_node_name in start_node_names:
            pre_node = None
            start_node = self._graph[start_node_name]
            # get specific pre node along attention mask input branch
            for input_name in start_node.inputs:
                input_node = self._graph[input_name]
                if input_node.op_type == "Mul":
                    input1 = self._graph[input_node.inputs[0]]
                    input2 = self._graph[input_node.inputs[1]]
                    input_init = input1 if input1.op_type == "Initializer" else input2
                    if input_init.value < 0:
                        pre_node = input_node
            if pre_node is None:
                return False
            if pre_node.name not in pre_node_dic:
                pre_node_dic[pre_node.name] = {}
            pre_node_dic[pre_node.name][start_node_name] = 1

        # build expand value initializer
        input_node = self._graph[self._graph.inputs[0]]
        input_shape = BertBigKernelOptimizer.convert_str2numlist(input_node.shape)
        assert len(input_shape) == 2, 'input shape: [batch_size, seq]'
        expand_value = [input_shape[0], 1, input_shape[1], input_shape[1]]
        expand_init = self._graph.add_initializer(
            "Expand_mask_expand_shape",
            np.array(expand_value)
        )

        # insert expand node before start node
        for pre_node_name in pre_node_dic:
            pre_node = self._graph[pre_node_name]
            assert len(pre_node.outputs) == 1
            for start_node_name in pre_node_dic[pre_node_name]:
                inserted_node = self._graph.add_node(
                    "Expand_before_{}".format(start_node_name),
                    "Expand",
                    inputs=[pre_node.outputs[0]],
                    outputs=["Expand_before_{}_output".format(start_node_name)]
                )
                start_node = self._graph[start_node_name]
                input_idx = pre_node_dic[pre_node_name][start_node_name]
                start_node.inputs[input_idx] = "Expand_before_{}_output".format(start_node_name)
                inserted_node.inputs.append("Expand_mask_expand_shape")
        return True

    def optimize(self, graph):
        self._graph = graph
        try:
            with_kernel, nodes_info, hidden_num = self.get_bigkernel_part()
        except:
            return graph, False
        if not with_kernel:
            return self._graph, with_kernel
        input_shape = self._graph[self._graph.inputs[0]].shape
        input_seq = BertBigKernelOptimizer.convert_str2numlist(input_shape)
        self.insert_reshape_node(nodes_info.get('Mul'), hidden_num)
        self.fix_reshape_node(nodes_info.get('Reshape'))
        self.modify_transpose_node(nodes_info.get('Transpose'))
        self.modify_div_node(nodes_info.get('Div'))
        with_kernel &= self.insert_last_reshape(input_seq[-1], hidden_num)
        with_kernel &= self.broadcast_input_mask(nodes_info.get('Add'))
        if not with_kernel:
            return graph, False
        return self._graph, with_kernel

    @staticmethod
    def convert_str2numlist(input_str):
        # sample: '[1, 2, 3]'-->[1, 2, 3]
        input_str = input_str[1:-1].replace(' ', '')
        input_seq = [int(v) for v in re.split(',', input_str)]
        return input_seq