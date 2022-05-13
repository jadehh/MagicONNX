from magiconnx import OnnxGraph


if __name__ == '__main__':
    import numpy as np
    graph = OnnxGraph.parse('resnet50-v1-12-int8.onnx')
    ph = graph.add_placeholder('dummy_input', 'int32', [2, 3, 4])
    init = graph.add_initializer('dummy_init', np.array([[2, 3, 4]]))
    # TODO:没有更新prev2node和next2node导致保存的模型不对
    add = graph.add_node('dummy_add', 'Add',
                         inputs=['dummy_input', 'dummy_init'],
                         outputs=['add_out'])

    graph.save('case1.onnx')
    argmax = graph.add_node('dummy_ArgMax',
                            'ArgMax',
                            {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
    graph.insert_node('dummy_add', argmax, mode='before')
    graph.save('case2.onnx')
