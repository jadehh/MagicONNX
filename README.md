# MagicONNX
- [OnnxNode操作](#OnnxNode操作)
    - [概念定义](#概念定义)
    - [公共API](#公共API)
    - [PlaceHolder专属API](#PlaceHolder专属API)
    - [Initializer专属API](#Initializer专属API)
    - [NodeProto专属API](#NodeProto专属API)

## [OnnxNode操作](#OnnxNode操作)
### [概念定义](#概念定义)
- PlaceHolder定义
  - 本质上是[onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3)中的 **`ValueInfoProto`**；
  - 表示整网中输入输出节点类型；
  - 包含属性有 **`name`**, **`shape`**, **`dtype`**。
- Initializer定义
  - 本质上是[onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3)中的 **`TensorProto`**；
  - 表示整网中常量节点类型；
  - 包含属性有 **`name`**, **`value`**。
- NodeProto定义
  - 本质上是[onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3)中的 **`NodeProto`**；
  - 表示整网中计算节点类型；
  - 包含属性有 **`name`**, **`inputs`**， **`outputs`**, **`attrs`**。
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
