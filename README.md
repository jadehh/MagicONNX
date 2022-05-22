![logo](./image/logo.png)

- [1. MagicONNX简介](#1-magiconnx简介)
- [2. 安装](#2-安装)
- [3. 使用方法](#3-使用方法)

# 1. MagicONNX简介

`MagicONNX` 是一个支持方便修改onnx文件的项目，其主要优势在于：

- 有详细的API文档说明

  相对于ONNX官方接口而言，MagicONNX有详细的接口说明和示例

- 可扩展的支持优化方法

  目前已支持的优化方法：
  - **long2int**: int64转int32计算
  - **constfolding**：常量折叠

# 2. 安装

```shell
git clone https://gitee.com/Ronnie_zheng/MagicONNX.git
cd MagicONNX
pip install .
```

# 3. 使用方法

node使用方法参见[Node说明](docs/node.md) 和 [样例代码](test/test_node.py)

graph使用方法参见[Graph说明](docs/graph.md) 和 [样例代码](test/test_graph.py)

![动画演示](./image/create.gif)
