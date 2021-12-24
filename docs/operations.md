![logo](../image/logo.png)
- [OnnxGraph操作](#onnxgraph操作)
  - [增加node](#增加node)
  - [查找node](#查找node)
  - [删除node](#删除node)
  - [修改node](#修改node)
  - [其他graph操作](#其他graph操作)
- [OnnxNode操作](#onnxnode操作)

## [OnnxGraph操作](#OnnxGraph操作)

### [增加node](#增加node)
- **add_placeholder(name, dtype, shape)**
  用于在 `graph` 中新增输入 `placeholder` 结点并返回
  - **`name`(string)**: 表示 `placeholder` 的名字，在 `graph` 中必须是唯一的，否则会报错；
  - **`dtype`(string/np.dtype)**: 表示 `placeholder` 的数据类型
  - **`shape`(list/tuple)**: 表示 `placeholder` 的维度信息
- **add_initializer(name, value)**
  用于在 `graph` 中新增输入 `initializer` 结点并返回
  - **`name`(string)**: 表示 `initializer` 的名字，在 `graph` 中必须是唯一的，否则会报错；
  - **`value`(np.ndarray)**: 表示 `initializer` 的数值，`initializer` 的数据类型和shape自动从 `value` 中获取
- **add_node(name, op_type, attrs={})**
  用于在 `graph` 中新增输入 `NodeProto` 结点并返回
  - **`name`(string)**: 表示 `NodeProto` 的名字，在 `graph` 中必须是唯一的，否则会报错；
  - **`op_type`(string)**: 必须与 [ONNX标准算子IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md)保持一致，比如加法操作只能填 `Add`
  - **`attrs`(dict)**: 参见 [ONNX标准算子IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md)中的 `Attributes`
- **insert_node(anchor, dst, index=0, mode='after')**
  用于在 `graph` 中的 `anchor` 前面或者后面的第 `index` 条边的位置插入 `dst` 结点
  > note:
  > - `graph.insert_node('Add_1', dst, index=1, mode='before')` 表示在 `Add_1` 的第1条**输入边**上插入 `dst` 结点
  > - `graph.insert_node('Add_1', dst, index=0, mode='after')` 表示在 `Add_1` 的第0条**输出边**上插入 `dst` 结点
  - **`anchor`(string)**: `OnnxNode` 结点的名字；
  - **`dst`(OnnxNode)**: 要插入的结点
  - **`index`(int)**: 表示在 `anchor` 结点的第 `index` 边上插入结点
  - **`mode`(string)**: 插入模式，只能选择 `after` 和 `before`

### [查找node](#查找node)
- **get_nodes(op_type)**
  获取 `graph` 中所有 `op_type` 结点集合
  - **`op_type`(string)**: 必须与 [ONNX标准算子IR](https://github.com/onnx/onnx/blob/master/docs/Operators.md)保持一致，比如加法操作只能填 `Add`
- **__getitem__(key)**
  获取 `graph` 中特定 `key` 结点
  > note:
  > - `add_6 = graph['Add_6']` 表示获取 `Add_6` 节点
  - **`key`(string)**: `OnnxNode` 的 `name`

### [删除node](#删除node)
- **del_node(name, maps=None, auto_connection=True)**
  删除 `graph` 中特定 `name` 结点
  - **`name`(string)**: `OnnxNode` 的 `name`
  - **`maps`(dict)**: 删除 `name` 后，使其前继结点的第 `x` 条输出与其后继结点的第 `y` 条输入相连
  - **`auto_connection`(bool)**: 是否自动连接输入输出边

### [修改node](#修改node)
- **__setitem__(key, value)**
  仅支持 `graph` 中名为 `key` 的计算结点(`NodeProto`)替换，且要求替换前后的算子输入输出数量必须一致
  > note:
  > `graph['Cast_2'] = argmax` 将 `Cast_2` 算子替换成 `argmax`
  > <big> warning </big>:
  > - 对应init和ph的修改不支持，可以先获取node再用node方法修改
  - **`key`(string)**: `NodeProto` 的 `name`
  - **`value`(OnnxNode)**: `OnnxNode`对象

### [其他graph操作](#其他graph操作)
- 属性
  - **inputs**
    输出整网所有输入结点 `name` 列表
  - **outputs**
    输出整网所有输出结点 `name` 列表
  - **graph**
    输出整网对象

- **connection(previous, out_idx, behind, in_idx)**
  连接 `previous` 的 `out_idx` 输出与 `behind` 的 `in_idx` 输入
  - **`previous`(string)**: `NodeProto` 的 `name`
  - **`out_idx`(int/list/tuple)**: 输出边位置索引
  - **`behind`(string)**: `NodeProto` 的 `name`
  - **`in_idx`(int/list/tuple)**: 输入边位置索引

- **save(path)**
  报错ONNX模型文件
  - **`path`(string)**: 保存路径

- **run(data)**
  以 `data` 作为ONNX模型输入，运行该模型并返回结果
  - **`data`(list of np.ndarray)**: 输入数据

- **dump(data, path='dump', outputs=[])**
  以 `data` 作为ONNX模型输入，运行该模型并把所有结点或 `outputs` 的输出值序列化成 `.npy` 文件
  - **`data`(list of np.ndarray)**: 输入数据
  - **`path`(string)**: `.npy` 保存路径
  - **`outputs`(list of string)**: 指定保存特定结点的输出值，缺省保存所有结点输出值

- **simplify(inplace, kwargs={})**
  使用 `onnx-simplifer` 工具对原模型进行简化
  - **`inplace`(bool)**: 是否修改原模型对象
  - **`kwargs`(dict)**: 参见 [simplify接口说明](https://github.com/daquexian/onnx-simplifier/blob/master/onnxsim/onnx_simplifier.py#L408)

- **extract(input_tensor_name_list, output_tensor_name_list, new_model_save_path, enable_model_check=True)**
  对模型进行截断处理
  - **`new_model_save_path`(str)**: 截断模型保存路径
  - **`input_tensor_name_list`(list)**: 输入节点列表
  - **`output_tensor_name_list`(list)**: 输出节点列表
  - **`enable_model_check`(bool)**: 是否开启`model_checker`，默认开启，对含有自定义算子的模型可以关闭

## [OnnxNode操作](#OnnxNode操作)

<table border="1">
<caption>OnnxNode属性与方法</caption>
<tr>
  <th rowspan="5">公共属性</th>
  <td>print(node)</td>
  <td>打印 node 节点</td>
</tr>
<tr>
  <td>node.node</td>
  <td>获取ONNX下的proto类型对象</td>
</tr>
<tr>
  <td>node.name</td>
  <td>获取/修改 node 节点名称</td>
</tr>
<tr>
  <td>node.op_type</td>
  <td>获取/修改 node 节点op_type</td>
</tr>
<tr>
  <td>node.doc_string</td>
  <td>获取/修改 node 节点doc_string</td>
</tr>
<tr>
  <th rowspan="2">PlaceHolder</th>
  <td>ph.dtype</td>
  <td>获取/修改 PlaceHolder 节点类型</td>
</tr>
<tr>
  <td>ph.shape</td>
  <td>获取/修改 PlaceHolder 节点维度信息</td>
</tr>
<tr>
  <th rowspan="1">Initializer</th>
  <td>init.value</td>
  <td>获取/修改 Initializer 节点数值<br>会丢失doc_string信息</td>
</tr>
<tr>
  <th rowspan="7">NodeProto</th>
  <td>node.inputs</td>
  <td>获取/修改 NodeProto 节点输入name列表</td>
</tr>
<tr>
  <td>node.set_input(0, 'in_x')</td>
  <td>修改 NodeProto 节点第0个输入name</td>
</tr>
<tr>
  <td>node.outputs</td>
  <td>获取/修改 NodeProto 节点输出name列表</td>
</tr>
<tr>
  <td>node.set_output(0, 'in_x')</td>
  <td>修改 NodeProto 节点第0个输出name</td>
</tr>
<tr>
  <td>node.attrs</td>
  <td>获取 NodeProto 节点所有属性键值对</td>
</tr>
<tr>
  <td>node['attr_1']</td>
  <td>获取 NodeProto 节点attr_1属性</td>
</tr>
<tr>
  <td>node['attr_x'] = attr_x</td>
  <td>修改 NodeProto 节点attr_x属性</td>
</tr>
</table>
