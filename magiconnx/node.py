import warnings
import numpy as np
from onnx import (NodeProto, TensorProto, ValueInfoProto,
                  TensorShapeProto, AttributeProto,
                  helper, numpy_helper)
from onnx.mapping import (TENSOR_TYPE_TO_NP_TYPE, NP_TYPE_TO_TENSOR_TYPE)

from interface import (BaseNode, PLACEHOLDER, INITIALIZER)
from utils.log import typeassert


class OnnxNode(BaseNode):
    @typeassert(node=(NodeProto, TensorProto, ValueInfoProto))
    def __init__(self, node):
        self._node = node
        if isinstance(node, NodeProto):
            self._op_type = node.op_type
            self._attr_map = self._parse_attrs()
        elif isinstance(node, TensorProto):
            self._op_type = INITIALIZER
        else:
            self._op_type = PLACEHOLDER

    ###############################################
    #######         common property         #######
    ###############################################
    def __str__(self):
        kvlist = {INITIALIZER: helper.printable_tensor_proto,
                  PLACEHOLDER: helper.printable_value_info}
        return kvlist.get(self._op_type,
                          default=helper.printable_node)(self._node)

    @property
    def node(self):
        return self._node

    @property
    def op_type(self):
        '''https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3#L414'''
        return self._op_type

    @op_type.setter
    def op_type(self, op_type):
        if self._op_type not in [INITIALIZER, PLACEHOLDER]:
            self._node.op_type = op_type
            self._op_type = op_type
        else:
            raise RuntimeError(
                f"Can't assign op_type({self._op_type}) to {self.name}")

    @property
    def name(self):
        # TODO: 使用装饰器
        if self._node.name == '':
            self._node.name = f'{self._node.op_type}_{self._node.output[0]}'
        return self._node.name

    @name.setter
    @typeassert(name=str)
    def name(self, name):
        self._node.name = name

    @property
    def doc_string(self):
        return self._node.doc_string

    @doc_string.setter
    @typeassert(doc_string=str)
    def doc_string(self, doc_string):
        self._node.doc_string = doc_string

    ##############################################
    #######      Placeholder property      #######
    ##############################################
    @property
    def dtype(self):
        tensor = self._node.type.tensor_type
        return TENSOR_TYPE_TO_NP_TYPE[tensor.elem_type]

    @dtype.setter
    def dtype(self, data_type):
        try:
            dtype = np.dtype(data_type)
        except Exception as e:
            print(e)
            raise RuntimeError(
                f'{data_type} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')
        tensor = self._node.type.tensor_type
        tensor.elem_type = NP_TYPE_TO_TENSOR_TYPE[dtype]

    @property
    def shape(self):
        dims = self._node.type.tensor_type.shape
        shapes = [str(dim.dim_value) if dim.dim_value >
                  0 else dim.dim_param for dim in dims.dim]
        return f'[{", ".join(shapes)}]'

    @shape.setter
    @typeassert(shapes=(tuple, list))
    def shape(self, shapes):
        dims = self._node.type.tensor_type.shape
        if len(dims.dim) != len(shapes):
            warnings.warn(
                f'Original len(shapes) is {len(dims.dim)}, but current settings is {shapes}')
        new_shape = TensorShapeProto()
        for shape in shapes:
            new_dim = TensorShapeProto().Dimension()
            if shape > 0:
                new_dim.dim_value = shape
            else:
                new_dim.dim_param = '-1'
            new_shape.dim.append(new_dim)
        dims.CopyFrom(new_shape)

    ##############################################
    #######      Initializer property      #######
    ##############################################
    @property
    def value(self):
        """Convert TensorProto to numpy array."""
        if self._op_type not in [INITIALIZER, 'Constant']:
            raise RuntimeError(
                f'Only support Initializer/Constant, but the current node({self.name}) is {self._op_type}')
        if self._op_type == INITIALIZER:
            ret = numpy_helper.to_array(self._node)
        else:
            ret = numpy_helper.to_array(self._node.attribute[0].t)
        return ret

    @value.setter
    @typeassert(value=np.ndarray)
    def value(self, value):
        if self._op_type not in [INITIALIZER, 'Constant']:
            raise RuntimeError(
                f'Only support Initializer/Constant, but the current node({self.name}) is {self._op_type}')
        if self._op_type == INITIALIZER:
            self._node.CopyFrom(numpy_helper.from_array(value, self.name))
        else:
            self._node.attribute[0].t.CopyFrom(numpy_helper.from_array(value))

    ##############################################
    #######       NodeProto property       #######
    ##############################################
    @property
    def inputs(self):
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            warnings.warn(
                f'Only NodeProto has input_names, but the current node({self.name}) is {self._op_type}')
            return []
        return self._node.input

    @inputs.setter
    @typeassert(value=(list, tuple))
    def inputs(self, value):
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            warnings.warn(
                f'Only NodeProto can set input_names, but the current node({self.name}) is {self._op_type}')
        while self._node.input:
            self._node.input.pop()
        self._node.input.extend(value)

    @typeassert(idx=int, name=str)
    def set_input(self, idx, name):
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            warnings.warn(
                f'Only NodeProto can set input_names, but the current node is {self._op_type}')
        self._node.input[idx] = name

    @property
    def outputs(self):
        # TODO:每个节点都有(输入)输出才对
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            warnings.warn(
                f'Only NodeProto has output_names, but the current node is {self._op_type}')
            return []
        return self._node.output

    @outputs.setter
    @typeassert(value=(list, tuple))
    def outputs(self, value):
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            warnings.warn(
                f'Only NodeProto can set output_names, but the current node is {self._op_type}')
        while self._node.output:
            self._node.output.pop()
        self._node.output.extend(value)

    @typeassert(idx=int, name=str)
    def set_output(self, idx, name):
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            warnings.warn(
                f'Only NodeProto can set input_names, but the current node is {self._op_type}')
        self._node.output[idx] = name

    @property
    def attrs(self):
        return self._attr_map

    def __getitem__(self, key):
        if key not in self._attr_map:
            raise KeyError(f'{self.name} do not have {key} attribute')
        return helper.get_attribute_value(self._attr_map[key])

    def __setitem__(self, key, value):
        if key not in self._attr_map:
            warnings.warn(
                f'{self.name} do not have {key} attribute, you should be responsible for it.')
        attr = helper.make_attribute(key, value)
        if key in self._attr_map:
            self._attr_map[key].CopyFrom(attr)
        else:
            self._node.attribute.append(attr)
            self._attr_map[key] = value

    @property
    def domain(self):
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            raise RuntimeError(f"Only NodeProto can get domain attr")
        return self._node.domain

    def clear_domain(self):
        if self._op_type in [INITIALIZER, PLACEHOLDER]:
            raise RuntimeError(f"Only NodeProto can set domain attr")
        if self._node.HasField('domain'):
            self._node.ClearField('domain')

    def _parse_attrs(self):
        return {attr.name: attr for attr in self._node.attribute}
