from .int64toint32_optimizer import Int64ToInt32Optimizer
from .continuousslice_optimizer import ContinuousSliceOptimizer
from .bertbigkernel_optimizer import BertBigKernelOptimizer


def get_optimizers_info():
    supported_optimizers = {
        "Int64ToInt32Optimizer": Int64ToInt32Optimizer(name="Int64ToInt32Optimizer"),
        "ContinuousSliceOptimizer": ContinuousSliceOptimizer(name="ContinuousSliceOptimizer"),
        "BertBigKernelOptimizer": BertBigKernelOptimizer(name="BertBigKernelOptimizer"),
    }
    return supported_optimizers
