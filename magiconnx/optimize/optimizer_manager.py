from easydict import EasyDict as edict
from collections import OrderedDict as dict
from .base_optimizer import BaseOptimizer
from .optimizers import get_optimizers_info
from ..graph import OnnxGraph
from ..utils.io import load_json_file
from ..utils.log import typeassert


SAFE_OPTIMIZERS = ['Int64ToInt32Optimizer', 'ContinuousSliceOptimizer']


class OptimizerManager():
    """
    @des        Manager for all optimizers, supported:
                  1. safe mode: only load optimizers in SAFE_OPTIMIZERS
                  2. all mode: load all available optimizers
                  3. input config file: load optimizers defined in input config file
    """

    @typeassert(graph=OnnxGraph, cfg_path=str)
    def __init__(self, graph, cfg_path='', mode='safe'):
        self.onnx_graph = graph
        self.cfg_path = cfg_path
        self.mode = mode
        self.__optimizer_map = OptimizerManager.get_available_optimizers()
        if self.cfg_path != '':
            self._get_optimizers_from_cfg(cfg_path)
        elif self.mode == 'safe':
            self.__optimizers = [self.__optimizer_map[optimizer_name] for
                                 optimizer_name in SAFE_OPTIMIZERS]
        elif self.mode == 'all':
            self.__optimizers = [self.__optimizer_map[optimizer_name] for
                                 optimizer_name in self.__optimizer_map.keys()]
        else:
            raise ValueError('Mode {} not in supported modes.'.format(self.mode))

    @typeassert(cfg_path=str)
    def _get_optimizers_from_cfg(self, cfg_path):
        self.__optimizers = []
        cfg_data = load_json_file(cfg_path)
        self.cfg_data = edict(cfg_data)
        for optimizer_name in self.cfg_data.optimizers:
            if self.__optimizer_map.get(optimizer_name) is None:
                raise ValueError('Not support optimizer: {}'.format(optimizer_name))
            self.__optimizers.append(self.__optimizer_map.get(optimizer_name))

    def clear(self):
        self.__optimizers.clear()

    @typeassert(optimizer=BaseOptimizer)
    def add_optimizer(self, optimizer):
        self.__optimizers.append(optimizer)

    @typeassert(optimizer=BaseOptimizer)
    def remove_optimizer(self, optimizer):
        self.__optimizers.remove(optimizer)

    def apply(self):
        onnx_graph = self.onnx_graph
        for optimizer in self.__optimizers:
            onnx_graph, flag_optimized = optimizer.optimize(onnx_graph)
            if flag_optimized:
                print("succeed: {}".format(optimizer.get_name()))
            else:
                print("failed: {}".format(optimizer.get_name()))

    @staticmethod
    def get_available_optimizers():
        available_optimizer_dicts = dict()
        for optimizer_name, optimizer_class in get_optimizers_info().items():
            # build optimizer obj
            optimizer_obj = optimizer_class(name=optimizer_name)
            available_optimizer_dicts[optimizer_obj.get_name()] = optimizer_obj

        return available_optimizer_dicts
