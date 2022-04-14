from .int64toint32_optimizer import Int64ToInt32Optimizer
from .continuousslice_optimizer import ContinuousSliceOptimizer


def get_optimizers_info():
    supported_optimizers = {
        "Int64ToInt32Optimizer": Int64ToInt32Optimizer,
        "ContinuousSliceOptimizer": ContinuousSliceOptimizer,
    }
    return supported_optimizers
