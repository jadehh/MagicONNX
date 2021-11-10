# MagicONNX
- [MagicONNX](#magiconnx)
  - [概念定义](#概念定义)
    - [PlaceHolder定义](#placeholder定义)
    - [Initializer定义](#initializer定义)
    - [NodeProto定义](#nodeproto定义)
  - [OnnxGraph操作](#onnxgraph操作)
    - [增加node](#增加node)
    - [查找node](#查找node)
    - [删除node](#删除node)
    - [修改node](#修改node)
    - [其他graph操作](#其他graph操作)
  - [OnnxNode操作](#onnxnode操作)
    - [公共API](#公共api)
    - [PlaceHolder专属API](#placeholder专属api)
    - [Initializer专属API](#initializer专属api)
    - [NodeProto专属API](#nodeproto专属api)

## [概念定义](#概念定义)
### [PlaceHolder定义](#PlaceHolder定义)
- 本质上是[onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3)中的 **`ValueInfoProto`**；
- 表示**整网中输入节点类型**；
- 包含属性有 **`name`**, **`shape`**, **`dtype`**。
### [Initializer定义](#Initializer定义)
- 本质上是[onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3)中的 **`TensorProto`**；
- 表示**整网中常量节点类型**；
- 包含属性有 **`name`**, **`value`**。
### [NodeProto定义](#NodeProto定义)
- 本质上是[onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3)中的 **`NodeProto`**；
- 表示**整网中计算节点类型**；
- 包含属性有 **`name`**, **`inputs`**， **`outputs`**, **`attrs`**。

## [OnnxGraph操作](#OnnxGraph操作)
```python
graph = OnnxGraph('layernorm.onnx')
```
### [增加node](#增加node)
```python
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
### [查找node](#查找node)
```python
# test for Retrieve
adds = graph.get_nodes("Add")             # 获得整网中所有Add节点
inits = graph.get_nodes("Initializer")    # 获得整网中所有常量节点
phs = graph.get_nodes("Placeholder")
add_6 = graph['Add_6']                    # 获得Add_6单个节点
```
### [删除node](#删除node)
```python
# test for Delete
graph.del_node('Cast_2')                  # 删除Cast_2节点，仅支持单输入单输出节点
graph.save('case4.onnx')
```
### [修改node](#修改node)
```python
# test for Update
argmax = graph.add_node('dummy_ArgMax',
                  'ArgMax',
                  {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
graph['Cast_2'] = argmax
graph.save('case3.onnx')
```
### [其他graph操作](#其他graph操作)
```python
# test for graph operation
print(graph)
print(graph.inputs)
print(graph.outputs)
print(graph.graph)
# graph.connection('Cast_2', [0], 'Add_6', [1])
# graph.save('case5.onnx')
data = np.randn(20, 5, 10, 10).astype(np.float32)
graph.dump([data])
ret = graph.run([data])
for output in ret:
    print(output.shape)
graph.simplify(True).save('case6.onnx')
```
## [OnnxNode操作](#OnnxNode操作)

### [公共API](#公共API)
对一个 **`OnnxNode`** 对象而言，公共API包含 **`name`**, **`print`**， **`node`**, **`op_type`**, **`doc_string`**。
```python
# 以下示例中node均是OnnxNode实例

print(node)                         # 打印 node 节点详细信息
node.node                           # 获取 node 对应onnx的原始节点
node.name                           # 获取 node 节点名称
node.name = 'Arbitrary string'      # 修改 node 节点名称
node.op_type                        # 获取 node 节点op_type
node.op_type = 'op_type'            # 修改 node 节点op_type(仅NodeProto支持)
node.doc_string
node.doc_string = 'doc_string'
```
### [PlaceHolder专属API](#PlaceHolder专属API)
**`PlaceHolder`** 对象主要用于定义网络输入输出，其专属API包含 **`dtype`**, **`shape`**。
> 说明：
> - **必须使用**[numpy基础数据结构](https://numpy.org/doc/stable/user/basics.types.html)修改`dtype`
> - **可以使用** `list/tuple/np.array` 等迭代器修改`shape`
```python
# 获取/修改 node 节点数据类型
node.dtype
node.dtype = 'int32'
# 获取/修改 node 节点shape
node.shape 
node.shape = [1, 2, 3]          # 必须保证修改前后shape长度相等
```
### [Initializer专属API](#Initializer专属API)
**`Initializer`** 对象主要用于表示常量，为简化API仅提供 **`value`** 操作，对数据值的操作复用numpy接口。
```python
node.value                      # 返回np.array
node.value = np.array([1])      # 修改 node 节点常量值，只能通过numpy修改，会丢失doc_string信息
```
### [NodeProto专属API](#NodeProto专属API)
**`NodeProto`** 对象主要用于定义计算/逻辑类算子，其专属API包含 **`inputs`**， **`outputs`**, **`attrs`**, **`domain`**。
```python
node.inputs                         # 获取输入name列表
node.inputs = ['in_1', 'in_2']      # 修改输入name列表，请谨慎使用
node.set_input(0, 'in_x')           # 修改第0个输入name
# outputs使用方法同inputs
node.attrs                          # 获取所有属性键值对
node['attr_1']                      # 获取attr_1属性
node['attr_x'] = attr_x             # 修改attr_x属性
```
