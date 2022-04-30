![logo](../image/logo.png)

- [OnnxNode说明](#onnxnode说明)
  - [关键概念定义](#关键概念定义)
    - [Placeholder定义](#placeholder定义)
    - [Initializer定义](#initializer定义)
    - [Node定义](#node定义)
  - [方法使用说明](#方法使用说明)
    - [通用方法](#通用方法)
    - [私有方法](#私有方法)
    - [关于 `inputs/input_nodes` 和 `outputs/output_nodes` 的解释](#关于-inputsinput_nodes-和-outputsoutput_nodes-的解释)


# OnnxNode说明

## 关键概念定义

### Placeholder定义
- 表示**ONNX整网中的输入输出节点**；
- 特有属性包括 **`shape`**, **`dtype`**；
- 在ONNX框架中对应于 [onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3) 的 **`ValueInfoProto`**。

### Initializer定义
- 表示**ONNX整网中的常量节点**；
- 特有属性包括 **`value`**；
- 在ONNX框架中对应于 [onnx.proto3](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3) 的 **`TensorProto`** 和 [ONNX标准库算子](https://github.com/onnx/onnx/blob/master/docs/Operators.md) 的 **`Constant`** 算子；

### Node定义
- 表示**ONNX整网中的计算节点**，可参见[ONNX标准库算子](https://github.com/onnx/onnx/blob/master/docs/Operators.md)；
- 特有方法包括主要是修改输入输出和属性，下面有描述；

## 方法使用说明

### 通用方法

| 属性/方法                    | 释义                                                                                      |
| ---------------------------- | ----------------------------------------------------------------------------------------- |
| `BaseNode.create_node(node)` | 静态工厂方法</br>输入node为onnxproto对象，输出为 `Placeholder`，`Initializer`，`Node`对象 |
| `node.node`                  | 只读属性，返回ONNX下的proto类型对象                                                       |
| `node.op_type`               | 可读写属性，节点类型(`Placeholder`，`Initializer`，`Node.op_type`)                        |
| `node.name`                  | 可读写属性，节点名称                                                                      |
| `node.inputs`                | 可读写属性，节点输入                                                                      |
| `node.outputs`               | 可读写属性，节点输出                                                                      |

### 私有方法

<table border="1">
<caption>私有属性与方法</caption>
<tr>
  <th rowspan="2">PlaceHolder</th>
  <td>ph.dtype</td>
  <td>可读写属性，PlaceHolder 节点数据类型</td>
</tr>
<tr>
  <td>ph.shape</td>
  <td>可读写属性，PlaceHolder 节点维度信息</td>
</tr>
<tr>
  <th rowspan="1">Initializer</th>
  <td>init.value</td>
  <td>可读写属性，Initializer 节点具体数值（numpy.ndarray形式返回/设置）</td>
</tr>
<tr>
  <th rowspan="7">Node</th>
  <td>node.input_nodes</td>
  <td>只读属性，Node 节点所有前继结点列表。</td>
</tr>
<tr>
  <td>node.set_input(self, idx, name)</td>
  <td>方法，修改 `Node` 节点第 idx 个输入name</td>
</tr>
<tr>
  <td>node.output_nodes</td>
  <td>只读属性，获取 Node 节点所有后继结点列表</td>
</tr>
<tr>
  <td>node.set_output(self, idx, name)</td>
  <td>方法，修改 `Node` 节点第 idx 个输出name</td>
</tr>
<tr>
  <td>node.attrs</td>
  <td>只读属性，获取 Node 节点所有属性键值对</td>
</tr>
<tr>
  <td>node['attr_1'] / node['attr_1'] = attr_x</td>
  <td>可读写属性，控制 Node 节点属性</td>
</tr>
</table>

### 关于 `inputs/input_nodes` 和 `outputs/output_nodes` 的解释