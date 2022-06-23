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
    @typeassert(cfg_path=str)
    @typeassert(optimizers=list)
    def __init__(self, graph, cfg_path='', optimizers=[], mode='safe'):
        self.onnx_graph = graph
        self.mode = mode
        self.cfg_path = cfg_path
        self.__input_optimizers = optimizers
        self.__optimizer_map = get_optimizers_info()
        self.__optimizer_names = self._generate_optimizer_names()
        self.__optimizers = []
        self._collect_optimizers()

    def _generate_optimizer_names(self):
        if not self.__input_optimizers:
            return self.__input_optimizers
        if self.cfg_path != '':
            cfg_data = load_json_file(self.cfg_path)
            cfg_data = edict(cfg_data)
            return cfg_data.optimizers
        if self.mode == 'safe':
            return SAFE_OPTIMIZERS
        elif self.mode == 'all':
            return [optimizer_name for optimizer_name in self.__optimizer_map.keys()]
        else:
            raise ValueError('Mode {} not in supported modes.'.format(self.mode))

    def _collect_optimizers(self):
        for optimizer_name in self.__optimizer_names:
            self.add_optimizer(optimizer_name)

    def get_optimizes(self):
        return self.__optimizers

    def clear(self):
        self.__optimizers.clear()

    @typeassert(optimizer_name=str)
    def add_optimizer(self, optimizer_name):
        optimizer = self.__optimizer_map.get(optimizer_name)
        self.__optimizers.append(optimizer)

    @typeassert(optimizer_name=str)
    @typeassert(remove_all=bool)
    def remove_optimizer(self, optimizer_name, remove_all=True):
        optimizer = self.__optimizer_map[optimizer_name]
        if remove_all:
            for idx in range(len(self.__optimizers)-1, -1, -1):
                if self.__optimizers[idx] == optimizer:
                    self.__optimizers.remove(optimizer)
        else:
            self.__optimizers.remove(optimizer)

    def apply(self):
        onnx_graph = self.onnx_graph
        for optimizer in self.__optimizers:
            onnx_graph, flag_optimized = optimizer.optimize(onnx_graph)
            if flag_optimized:
                print("succeed: {}".format(optimizer.get_name()))
            else:
                print("failed: {}".format(optimizer.get_name()))
        return onnx_graph
