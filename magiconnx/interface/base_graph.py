from abc import ABC, abstractmethod


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

    @abstractmethod
    def keep_default_domain(self):
        pass

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
