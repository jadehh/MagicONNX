from abc import ABC, abstractmethod


class BaseNode(ABC):
    ###############################################
    #######         common property         #######
    ###############################################
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
    @abstractmethod
    def doc_string(self):
        pass

    @doc_string.setter
    @abstractmethod
    def doc_string(self, doc_string):
        pass

    ##############################################
    #######      Placeholder property      #######
    ##############################################
    @property
    @abstractmethod
    def dtype(self):
        pass

    @dtype.setter
    @abstractmethod
    def dtype(self, data_type):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @shape.setter
    @abstractmethod
    def shape(self, shapes):
        pass

    ##############################################
    #######      Initializer property      #######
    ##############################################
    @property
    @abstractmethod
    def value(self):
        pass

    @value.setter
    @abstractmethod
    def value(self, value):
        pass

    ##############################################
    #######       NodeProto property       #######
    ##############################################
    @property
    @abstractmethod
    def inputs(self):
        pass

    @inputs.setter
    @abstractmethod
    def inputs(self, value):
        pass

    @abstractmethod
    def set_input(self, idx, name):
        pass

    @property
    @abstractmethod
    def outputs(self):
        pass

    @outputs.setter
    @abstractmethod
    def outputs(self, value):
        pass

    @abstractmethod
    def set_output(self, idx, name):
        pass

    @property
    @abstractmethod
    def attrs(self):
        pass

    @property
    @abstractmethod
    def domain(self):
        pass

    @abstractmethod
    def clear_domain(self):
        pass
