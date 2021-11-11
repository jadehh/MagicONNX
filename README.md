![logo](./image/logo.png)
- [魔改ONNX](#魔改onnx)
- [安装](#安装)

[学习教程](./docs/tutorials.md)

[API说明](./docs/operations.md)

## [魔改ONNX](#魔改ONNX)
![动画演示](./image/create.gif)
```python
graph = OnnxGraph('layernorm.onnx')

# test for create
ph = graph.add_placeholder('dummy_input', 'int32', [2, 3, 4])
init = graph.add_initializer('dummy_init', np.array([[2, 3, 4]]))
add = graph.add_node('dummy_add', 'Add')      # add_node默认单输入单输出，需要手动修改节点输入输出信息
add.inputs = ['dummy_input', 'dummy_init']
add.outputs = ['add_out']
graph.save('case1.onnx')

argmax = graph.add_node('dummy_ArgMax',
                      'ArgMax',
                      {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
graph.insert_node('dummy_add', argmax, mode='before')
graph.save('case2.onnx')
```

## [安装](#安装)
```shell
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
```
