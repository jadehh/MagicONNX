from abc import ABC, abstractmethod
from onnx import (NodeProto, TensorProto, ValueInfoProto)

PLACEHOLDER = 'Placeholder'
INITIALIZER = 'Initializer'


class Operator(ABC):

    @property
    @abstractmethod
    def node(self):
        pass

    @property
    @abstractmethod
    def op_type(self):
        pass

    @op_type.setter
    @abstractmethod
    def op_type(self, op_type):
        pass

    @property
    @abstractmethod
    def name(self):
        pass

    @name.setter
    @abstractmethod
    def name(self, name):
        pass

    @property
    def inputs(self):
        raise NotImplementedError('This function has not been implemented!')

    @property
    @abstractmethod
    def outputs(self):
        pass

    @property
    def attrs(self):
        raise NotImplementedError('This function has not been implemented!')
