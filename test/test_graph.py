from magiconnx import OnnxGraph


def test_create(graph):
    ph = graph.add_placeholder('dummy_input', 'int32', [2, 3, 4])
    init = graph.add_initializer('dummy_init', np.array([[[2], [3], [4]]]))
    add = graph.add_node('dummy_add', 'Add',
                         inputs=['dummy_input', 'dummy_init'],
                         outputs=['add_out'])
    ph_out = graph.add_placeholder(
        'add_out', 'int32', [2, 3, 4], is_input=False)

    graph.save('case1.onnx')
    argmax = graph.add_node('dummy_ArgMax',
                            'ArgMax',
                            {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
    graph.insert_node('dummy_add', argmax, mode='before')
    graph.save('case2.onnx')


def test_retrieve(graph):
    for node in graph.get_nodes('Add'):
        print(node)
    print(graph['Pow_3'])


def test_update(graph):
    argmin = graph.add_node('dummy_ArgMin',
                            'ArgMin',
                            {'axis': 1, 'keepdims': 0})
    graph['dummy_ArgMax'] = argmin
    graph.save('case3.onnx')


def test_del(graph):
    graph.remove('Cast_2')
    graph.save('case4.onnx')
    graph.remove('Div_8', maps={1:0})
    graph.save('case5.onnx')

if __name__ == '__main__':
    import numpy as np
    graph = OnnxGraph.parse('layernorm.onnx')
    print(graph)
    test_create(graph)
    test_retrieve(graph)
    test_update(graph)
    test_del(graph)
