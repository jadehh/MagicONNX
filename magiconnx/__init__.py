from .interface import (BaseGraph, Operator, INITIALIZER, PLACEHOLDER)
from .core.node import BaseNode, PlaceHolder, Initializer, Node
from .core.graph import OnnxGraph

# TODO: 可以写一个公共的numpy/string与onnx数据类型的转换函数
