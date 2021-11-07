import warnings
import numpy as np
import onnx

from onnx import (NodeProto, TensorProto, ValueInfoProto, TensorShapeProto, AttributeProto)
from onnx import (helper, numpy_helper)
from onnx.mapping import (TENSOR_TYPE_TO_NP_TYPE, NP_TYPE_TO_TENSOR_TYPE)

class OnnxNode():
    def __init__(self, node):
        if not isinstance(node, (NodeProto, TensorProto, ValueInfoProto)):
            raise TypeError(
                f'Only NodeProto, TensorProto and ValueInfoProto are legal, but got {type(node)}')
        self._node = node
        if isinstance(node, NodeProto):
            self._node_type = node.op_type
            self._attr_map = self._parse_attrs()
        elif isinstance(node, TensorProto):
            self._node_type = 'Initializer'
        else:
            self._node_type = 'Placeholder'

    ###############################################
    #######         common property         #######
    ###############################################
    def __str__(self):
        kvlist = {'Initializer': helper.printable_tensor_proto,
                  'Placeholder': helper.printable_value_info}
        return kvlist.setdefault(self._node_type,
                                helper.printable_node)(self._node)

    @property
    def node(self):
        return self._node

    @property
    def op_type(self):
        '''https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3#L414'''
        return self._node_type

    @op_type.setter
    def op_type(self, op_type):
        if self._node_type not in ['Initializer', 'Placeholder']:
            self._node.op_type = op_type
            self._node_type = op_type
        else:
            raise RuntimeError(f"Can't assign op_type to {self._node_type}({self.name})")

    @property
    def name(self):
        return self._node.name

    @name.setter
    def name(self, name):
        self._node.name = name

    @property
    def doc_string(self):
        return self._node.doc_string

    @doc_string.setter
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
            raise RuntimeError(f'{data_type} is illegal, only support basic data type: {NP_TYPE_TO_TENSOR_TYPE.keys()}')
        tensor = self._node.type.tensor_type
        tensor.elem_type = NP_TYPE_TO_TENSOR_TYPE[dtype]

    @property
    def shape(self):
        dims = self._node.type.tensor_type.shape
        shapes = [str(dim.dim_value) if dim.dim_value > 0 else dim.dim_param
                for dim in dims.dim]
        return f'[{", ".join(shapes)}]'

    @shape.setter
    def shape(self, shapes):
        dims = self._node.type.tensor_type.shape
        if len(dims.dim) != len(shapes):
            warnings.warn(f'Original len(shapes) is {len(dims.dim)}, but current settings is {shapes}')
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
        if self._node_type != 'Initializer':
            raise RuntimeError(
                f'Only support Initializer, but the current node({self.name}) is {self._node_type}')
        return numpy_helper.to_array(self._node)

    @value.setter
    def value(self, value):
        if self._node_type != 'Initializer':
            raise RuntimeError(
                f'Only support Initializer, but the current node({self.name}) is {self._node_type}')
        self._node.CopyFrom(numpy_helper.from_array(value, self.name))

    ##############################################
    #######       NodeProto property       #######
    ##############################################
    @property
    def inputs(self):
        if self._node_type in ['Initializer', 'Placeholder']:
            warnings.warn(
                f'Only NodeProto has input_names, but the current node({self.name}) is {self._node_type}')
            return []
        return self._node.input

    @inputs.setter
    def inputs(self, value):
        if self._node_type in ['Initializer', 'Placeholder']:
            warnings.warn(
                f'Only NodeProto can set input_names, but the current node({self.name}) is {self._node_type}')
        while self._node.input:
            self._node.input.pop()
        self._node.input.extend(value)

    def set_input(self, idx, name):
        if self._node_type in ['Initializer', 'Placeholder']:
            warnings.warn(
                f'Only NodeProto can set input_names, but the current node is {self._node_type}')
        self._node.input[idx] = name

    @property
    def outputs(self):
        if self._node_type in ['Initializer', 'Placeholder']:
            warnings.warn(
                f'Only NodeProto has output_names, but the current node is {self._node_type}')
            return []
        return self._node.output

    @outputs.setter
    def outputs(self, value):
        if self._node_type in ['Initializer', 'Placeholder']:
            warnings.warn(
                f'Only NodeProto can set output_names, but the current node is {self._node_type}')
        while self._node.output:
            self._node.output.pop()
        self._node.output.extend(value)

    def set_output(self, idx, name):
        if self._node_type in ['Initializer', 'Placeholder']:
            warnings.warn(
                f'Only NodeProto can set input_names, but the current node is {self._node_type}')
        self._node.output[idx] = name

    @property
    def attrs(self):
        return self._attr_map

    def __getitem__(self, key):
        if key not in self._attr_map:
            raise KeyError(f'{self.name} do not have {key} attribute')
        return self._attr_map[key]
 
    def __setitem__(self, key, value):
        if key not in self._attr_map:
            warnings.warn(f'{self.name} do not have {key} attribute, you should be responsible for it.')
        attr = helper.make_attribute(key, value)
        if key in self._attr_map:
            self._attr_map[key].CopyFrom(attr)
        else:
            self._node.attribute.append(attr)
            self._attr_map[key] = value

    @property
    def domain(self):
        if self._node_type in ['Initializer', 'Placeholder']:
            raise RuntimeError(f"Only NodeProto can get domain attr")
        return self._node.domain

    @domain.setter
    def domain(self, domain):
        if self._node_type in ['Initializer', 'Placeholder']:
            raise RuntimeError(f"Only NodeProto can set domain attr")
        self._node.domain = domain

    def _parse_attrs(self):
        return {attr.name: helper.get_attribute_value(attr)
                for attr in self._node.attribute}
