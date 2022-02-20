from abc import ABC, abstractmethod

PLACEHOLDER = 'Placeholder'
INITIALIZER = 'Initializer'

class BaseGraph(ABC):
    ###############################################
    #######              Create             #######
    ###############################################
    @abstractmethod
    def add_placeholder(self, name, dtype, shape):
        pass

    @abstractmethod
    def add_initializer(self, name, value):
        pass

    @abstractmethod
    def add_node(self, name, op_type, attrs=dict()):
        pass

    @abstractmethod
    def insert_node(self, anchor, dst, index=0, mode='after'):
        pass

    ###############################################
    #######            Retrieve             #######
    ###############################################
    @abstractmethod
    def get_nodes(self, op_type):
        pass

    @abstractmethod
    def __getitem__(self, key):
        pass

    ###############################################
    #######             Update              #######
    ###############################################
    @abstractmethod
    def __setitem__(self, key, value):
        pass

    ###############################################
    #######             Delete              #######
    ###############################################
    @abstractmethod
    def del_node(self, name, maps={0: 0}, auto_connection=True):
        pass

    ###############################################
    #######         graph operation         #######
    ###############################################
    @property
    @abstractmethod
    def inputs(self):
        pass

    @property
    @abstractmethod
    def outputs(self):
        pass

    @property
    @abstractmethod
    def graph(self):
        pass

    # @property
    # @abstractmethod
    # def domain(self):
    #     pass

    # @domain.setter
    # @abstractmethod
    # def domain(self):
    #     pass

    # @property
    # @abstractmethod
    # def opset_import(self):
    #     pass

    # @opset_import.setter
    # @abstractmethod
    # def opset_import(self):
    #     pass

    @abstractmethod
    def connection(self, previous, out_idx, behind, in_idx):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def run(self, data):
        pass

    @abstractmethod
    def dump(self, data, path='dump', outputs=[]):
        pass

    @abstractmethod
    def simplify(self, inplace, **kwargs):
        pass

    @abstractmethod
    def extract(self, new_model_save_path, input_tensor_name_list, output_tensor_name_list, enable_model_check=True):
        pass


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
        '''https://github.com/onnx/onnx/blob/master/onnx/onnx.proto3#L414'''
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
