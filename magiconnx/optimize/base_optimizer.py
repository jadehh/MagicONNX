from abc import ABCMeta, abstractmethod
from ..graph import OnnxGraph
from ..utils.log import typeassert


class BaseOptimizer(metaclass=ABCMeta):
    """
    @des        Sigleton for baseoptimizer
    """
    __instance = {}

    def __new__(cls, name):
        if name not in cls.__instance:
            BaseOptimizer.__instance[name] = super().__new__(cls)
        return cls.__instance[name]

    def __init__(self, name):
        self.__name = name

    def __eq__(self, obj):
        if not isinstance(obj, BaseOptimizer):
            return False
        return self.__name == obj.get_name()

    @abstractmethod
    @typeassert(graph=OnnxGraph)
    def optimize(self, graph):
        """
        具体优化方案
        """
        pass

    def get_name(self):
        return self.__name
