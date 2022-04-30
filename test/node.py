from magiconnx import PlaceHolder, BaseNode, Initializer, Node


def common(node):
    print(f'type(node): {type(node)}\n',
          f'node.node: {node.node}\n',
          f'node.op_type: {node.op_type}\n',
          f'node.name: {node.name}\n',
          f'node.inputs: {node.inputs}\n',
          f'node.outputs: {node.outputs}\n',
          f'node.prev: {node.prev()}\n')


def printph(ph):
    print(f'ph.dtype: {ph.dtype}\n',
          f'ph.shape: {ph.shape}\n',
          f'ph: {ph}')


def printinit(init):
    print(f'init.value: {init.value}\n',
          f'init: {init}')


if __name__ == '__main__':
    import onnx
    import numpy as np
    model = onnx.load('resnet50-v1-12-int8.onnx')
    for vetex in (model.graph.input[0], model.graph.initializer[0], model.graph.node[0]):
        dummy_node = BaseNode.create_node(vetex)
        common(dummy_node)
        # dummy_node.name = f'{dummy_node.name}_test'
        # dummy_node.inputs = ['test1']
        # dummy_node.inputs = ['test2']
        # common(dummy_node)
    ph = PlaceHolder('input', 'int32', (1, -1, 'dynamic'))
    init = Initializer('init', np.array([1, 2, 3]))
    node = Node('node', 'Node')
    for vetex in (ph, init, node):
        common(vetex)
