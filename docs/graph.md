![logo](../image/logo.png)

- [Create](#create)
  - [add_placeholder(name, dtype, shape, is_input=True)](#add_placeholdername-dtype-shape-is_inputtrue)
  - [add_initializer(name, value)](#add_initializername-value)
  - [add_node(name, op_type, attrs={}, inputs=[], outputs=[], domain=None)](#add_nodename-op_type-attrs-inputs-outputs-domainnone)
  - [insert_node(anchor, dst, src_idx=0, mode='after')](#insert_nodeanchor-dst-src_idx0-modeafter)
  - [示例](#示例)
- [Retrieve](#retrieve)
  - [get_nodes(op_type)](#get_nodesop_type)
  - [索引查找](#索引查找)
  - [示例](#示例-1)
- [Update](#update)
- [Delete](#delete)
  - [remove(self, name, maps={0:0})](#removeself-name-maps00)
- [其他graph操作](#其他graph操作)
  - [属性](#属性)
  - [方法](#方法)

# Create

## add_placeholder(name, dtype, shape, is_input=True)

用于在 `graph` 中新增输入 `placeholder` 并返回

- **`name`(string)**: 表示输入输出的名字，在 `graph` 中必须唯一存在；
- **`dtype`(string/np.dtype)**: 表示输入输出的数据类型
- **`shape`(list/tuple)**: 表示输入输出的维度信息
- **`is_input`(bool)**: `True` 表示创建的是输入节点

## add_initializer(name, value)

用于在 `graph` 中新增常量 `initializer` 并返回

- **`name`(string)**: 表示常量节点名字，在 `graph` 中必须唯一存在；
- **`value`(np.ndarray)**: 表示常量节点数值

## add_node(name, op_type, attrs={}, inputs=[], outputs=[], domain=None)

用于在 `graph` 中新增计算节点 `Node` 并返回

- **`name`(string)**: 表示计算节点名字，在 `graph` 中必须唯一存在；
- **`op_type`(string)**: 必须与 [ONNX标准算子IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md)保持一致，比如加法操作只能填 `Add`
- **`attrs`(dict)**: 参见 [ONNX标准算子IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md)中的 `Attributes`
- **`inputs`(list)**: 表示节点输入；
- **`outputs`(list)**: 必须节点输出；
- **`domain`(string/list[string])**: 节点作用域

## insert_node(anchor, dst, src_idx=0, mode='after')

用于在 `anchor` 的第 `src_idx` 个前驱节点(`mode='before'`)或者后继节点(`mode='after'`)上插入 `dst` 结点

- **`anchor`(string)**: 结点名字；
- **`dst`(BaseNode)**: 要插入的结点对象
- **`src_idx`(int)**: 前驱/后继节点索引值
- **`mode`(string)**: 插入模式，只能选择 `after` 和 `before`

## 示例

完整代码见 [test_graph.py](../test/test_graph.py)

```python
def test_create(graph):
    ph = graph.add_placeholder('dummy_input', 'int32', [2, 3, 4])
    init = graph.add_initializer('dummy_init', np.array([[[2], [3], [4]]]))
    add = graph.add_node('dummy_add', 'Add',
                         inputs=['dummy_input', 'dummy_init'],
                         outputs=['add_out'])
    ph_out = graph.add_placeholder('add_out', 'int32', [2, 3, 4], is_input=False)
    graph.save('case1.onnx')

    argmax = graph.add_node('dummy_ArgMax',
                            'ArgMax',
                            {'axis': 0, 'keepdims': 1, 'select_last_index': 0})
    graph.insert_node('dummy_add', argmax, mode='before')
    graph.save('case2.onnx')
```

# Retrieve

## get_nodes(op_type)

获取 `graph` 中所有 `op_type` 结点集合

- **`op_type`(string)**: 结点类型；

## 索引查找

使用 `node = graph[node_name]` 方式获取特定结点

## 示例

完整代码见 [test_graph.py](../test/test_graph.py)

```python
def test_retrieve(graph):
    for node in graph.get_nodes('Add'):
        print(node)
    print(graph['Pow_3'])
```

# Update

使用 `graph[node_name] = new_node` 方式更新替换特定结点
> 不支持更新替换 `PLACEHOLDER` 和 `INITIALIZER`，建议使用node方法修改

```python
def test_update(graph):
    argmin = graph.add_node('dummy_ArgMin',
                            'ArgMin',
                            {'axis': 1, 'keepdims': 0})
    graph['dummy_ArgMax'] = argmin
    graph.save('case3.onnx')
```

# Delete

## remove(self, name, maps={0:0})

删除 `graph` 中特定 `name` 结点，并利用maps将前驱节点和后继节点自动连接起来

- **`name`(string)**: 被删除节点名字
- **`maps`(dict)**: 删除 `name` 后的自动连边规则

```python
def test_del(graph):
    # Cast是单输入单输出，删除Cast_2后，将输入输出自动连接
    graph.remove('Cast_2')
    graph.save('case4.onnx')
    # Div是2输入单输出，删除Div_8后，将其第2个输入与唯一输出自动连接
    graph.remove('Div_8', maps={1:0})
    graph.save('case5.onnx')
```

# 其他graph操作

## 属性

| 属性      |        释义            |
| -------------- |-----------------------|
| `graph.inputs`    |只读属性，返回整网所有输入节点列表|
| `graph.outputs` |只读属性，返回整网所有输出节点列表|
| `graph.inits`    |只读属性，返回整网所有常量节点列表|

## 方法

- **save(path)**

  保存ONNX模型文件
  - **`path`(string)**: 保存路径

- **run(data)**

  以 `data` 作为ONNX模型输入，运行该模型并返回结果
  - **`data`(list[np.ndarray])**: 输入数据

- **dump(data, path='dump', outputs=[])**

  以 `data` 作为ONNX模型输入，运行该模型并把所有结点或 `outputs` 的输出值序列化成 `.npy` 文件
  - **`data`(list[np.ndarray])**: 输入数据
  - **`path`(string)**: `.npy` 保存路径
  - **`outputs`(list[string])**: 指定保存特定结点的输出值，缺省保存所有结点输出值

- **extract(input_tensor_name_list, output_tensor_name_list, new_model_save_path, enable_model_check=True)**

  对模型进行截断处理
  - **`new_model_save_path`(str)**: 截断模型保存路径
  - **`input_tensor_name_list`(list)**: 输入节点列表
  - **`output_tensor_name_list`(list)**: 输出节点列表
  - **`enable_model_check`(bool)**: 是否开启`model_checker`，默认开启，对含有自定义算子的模型可以关闭
