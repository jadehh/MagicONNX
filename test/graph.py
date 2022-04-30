from magiconnx import OnnxGraph


if __name__ == '__main__':
    import onnx
    import numpy as np
    graph = OnnxGraph.parse('resnet50-v1-12-int8.onnx')
    for node in graph.next2node.values():
        if len(node.inputs) != len(node.prev()):
            import pdb
            pdb.set_trace()
            print(f'length')

        for name, vetex in zip(node.inputs, node.prev()):
            if name not in vetex.outputs:
                import pdb
                pdb.set_trace()
                print(f'name = {name}\nvetex.name = {vetex.name}')
