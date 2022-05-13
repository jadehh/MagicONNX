import warnings
import numpy as np
from onnx import (NodeProto, TensorProto, ValueInfoProto,
                  helper, numpy_helper)
from onnx.mapping import (TENSOR_TYPE_TO_NP_TYPE, NP_TYPE_TO_TENSOR_TYPE)

from .. import (Operator, PLACEHOLDER, INITIALIZER)
from ..utils.log import typeassert
from .assistant import Assistant


def get_op_type(node):
    if isinstance(node, NodeProto):
        return node.op_type
    elif isinstance(node, TensorProto):
        return INITIALIZER
    elif isinstance(node, ValueInfoProto):
        return PLACEHOLDER


class BaseNode(Operator):
    def __init__(self, name='', op_type='', inputs=None, outputs=None, attrs=None):
        super(BaseNode, self).__init__()
        self._name = name
        self._op_type = op_type
        self._inputs = inputs
        self._outputs = outputs
        self._attrs = attrs

    @staticmethod
    @typeassert(node=(NodeProto, TensorProto, ValueInfoProto))
    def create_node(node):
        op_type = get_op_type(node)
        if op_type == PLACEHOLDER:
            return PlaceHolder.parse(node)
        elif op_type == 'Constant' or op_type == INITIALIZER:
            return Initializer.parse(node)
        else:
            return Node.parse(node)

    @property
    def node(self):
        return self._gen_onnx_node()

    @property
    def op_type(self):
        return self._op_type

    @op_type.setter
    @typeassert(op_type=str)
    def op_type(self, op_type):
        if op_type not in [INITIALIZER, PLACEHOLDER]:
            self._op_type = op_type
        else:
            raise RuntimeError(
                f"{self.name} belongs to {INITIALIZER} or {PLACEHOLDER}, which does not support setting op_type")

    @property
    def name(self):
        return self._name

    @name.setter
    @typeassert(name=str)
    def name(self, name):
        self._name = name

    @property
    def inputs(self):
        return self._inputs

    @inputs.setter
    @typeassert(inputs=list)
    def inputs(self, inputs):
        if self.op_type in (PLACEHOLDER, INITIALIZER):
            raise RuntimeError(
                f"{self.name} belongs to {INITIALIZER} or {PLACEHOLDER}, which does not support update inputs")
        self._inputs = inputs

    @property
    def outputs(self):
        return self._outputs

    @outputs.setter
    @typeassert(outputs=list)
    def outputs(self, outputs):
        self._outputs = outputs

    def prev(self):
        ret = []
        for name in self.inputs:
            if name in Assistant.name2node:
                ret.append(Assistant.name2node[name])
            else:
                ret.extend(Assistant.next2node[name])
        return ret

    def next(self):
        ret = []
        for name in self.outputs:
            if name in Assistant.name2node:
                ret.append(Assistant.name2node[name])
            else:
                ret.extend(Assistant.prev2node[name])
        return ret


class PlaceHolder(BaseNode):
    # TODO:不确定这里是否需要把output=None作为入参
    def __init__(self, name, dtype, shape):
        super(PlaceHolder, self).__init__(name, PLACEHOLDER,
                                          inputs=list(), outputs=list(), attrs=dict())
        try:
            self._dtype = np.dtype(dtype)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f'{dtype} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')

        self._shapes = shape

    @classmethod
    def parse(cls, node):
        tensor_type = node.type.tensor_type
        dtype = TENSOR_TYPE_TO_NP_TYPE[tensor_type.elem_type]
        shape = [dim.dim_value if dim.dim_value > 0 else -1
                 for dim in tensor_type.shape.dim]
        return cls(node.name, dtype, shape)

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, data_type):
        try:
            self._dtype = np.dtype(data_type)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f'{data_type} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')

    @property
    def shape(self):
        return self._shapes

    @shape.setter
    @typeassert(shapes=(tuple, list))
    def shape(self, shapes):
        self._shapes = shapes

    def __str__(self) -> str:
        return f'{self.op_type}({self.name}):=> (shape={self.shape}, dtype={self.dtype})'

    def __repr__(self) -> str:
        return self.__str__()

    def _gen_onnx_node(self):
        return helper.make_tensor_value_info(self.name,
                                             NP_TYPE_TO_TENSOR_TYPE[self.dtype],
                                             self.shape)


class Initializer(BaseNode):
    # TODO:不确定这里是否需要把output=None作为入参
    def __init__(self, name, value):
        super(Initializer, self).__init__(name, INITIALIZER,
                                          inputs=list(), outputs=list(), attrs=dict())
        self._value = value
        self._output_nodes = []

    @classmethod
    def parse(cls, node):
        if hasattr(node, 'op_type') and node.op_type == 'Constant':
            value = numpy_helper.to_array(node.attribute[0].t)
        else:
            value = numpy_helper.to_array(node)
        return cls(node.name, value)

    @property
    def value(self):
        return self._value

    @value.setter
    @typeassert(value=np.ndarray)
    def value(self, value):
        self._value = value

    def _gen_onnx_node(self):
        return helper.make_tensor(self._name,
                                  NP_TYPE_TO_TENSOR_TYPE[self._value.dtype],
                                  self._value.shape,
                                  self._value.flatten())

    def __str__(self) -> str:
        return f'{self.op_type}({self.name}):=> (shape={self._value.shape}, dtype={self._value.dtype})'

    def __repr__(self) -> str:
        return f'{self.__str__()}\n{self.value}'


class Node(BaseNode):
    def __init__(self, name, op_type,
                 inputs=None, outputs=None, attrs=None, domain=None):
        super(Node, self).__init__(name, op_type, inputs, outputs, attrs)
        self._domain = domain

    @classmethod
    def parse(cls, node):
        attrs = {attr.name: helper.get_attribute_value(attr)
                 for attr in node.attribute}
        return cls(node.name, node.op_type, list(node.input), list(node.output), attrs, node.domain)

    @property
    def attrs(self):
        return self._attrs

    def __getitem__(self, key):
        if key not in self._attrs:
            raise KeyError(f'Node({self.name}) do not have {key} attribute')
        return self._attrs[key]

    def __setitem__(self, key, value):
        if key not in self._attrs:
            warnings.warn(
                f'Node({self.name}) do not have {key} attribute, you should be responsible for it.')
        self._attrs[key] = value

    @property
    def domain(self):
        return self._domain

    def clear_domain(self):
        self._domain = None

    def _gen_onnx_node(self):
        return helper.make_node(self.op_type,
                                self.inputs,
                                self.outputs,
                                name=self.name,
                                domain=self.domain,
                                **self.attrs)

    def __str__(self) -> str:
        return f'Node({self.name}):=> \n\tinputs={self.inputs}\n\tattrs = {self.attrs}\n\toutputs={self.outputs}'

    def __repr__(self) -> str:
        return self.__str__()
