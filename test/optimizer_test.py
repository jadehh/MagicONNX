from magiconnx.graph import OnnxGraph
from magiconnx.optimize.optimizer_manager import OptimizerManager
import unittest


class OptimizerTestCase(unittest.TestCase):
    def setUp(self):
        self.config_json = "./config.json"

    def tearDown(self):
        pass

    def test_configpathmode(self):
        onnx_model = OnnxGraph("./case6.onnx")
        optimizer_manager = OptimizerManager(onnx_model, cfg_path=self.config_json)
        optimized_graph = optimizer_manager.apply()

    def test_safemode(self):
        onnx_model = OnnxGraph("./case6.onnx")
        optimizer_manager = OptimizerManager(onnx_model)
        optimized_graph = optimizer_manager.apply()

    def test_allmode(self):
        onnx_model = OnnxGraph("./case6.onnx")
        optimizer_manager = OptimizerManager(onnx_model, mode='all')
        optimized_graph = optimizer_manager.apply()

    def test_configdatamode(self):
        onnx_model = OnnxGraph("./case6.onnx")
        optimizer_manager = OptimizerManager(onnx_model, optimizers=['Int64ToInt32Optimizer'])
        optimized_graph = optimizer_manager.apply()

    def test_addoptimizer(self):
        onnx_model = OnnxGraph("./case6.onnx")
        optimizer_manager = OptimizerManager(onnx_model, optimizers=[
            'Int64ToInt32Optimizer', 'ContinuousSliceOptimizer'])
        optimizer_manager.add_optimizer('Int64ToInt32Optimizer')
        optimizers = optimizer_manager.get_optimizers()
        self.assertEqual(len(optimizers), 3, ['add optimizer failed'])

    def test_removeoptimizer(self):
        onnx_model = OnnxGraph("./case6.onnx")
        optimizer_manager = OptimizerManager(onnx_model, optimizers=[
            'Int64ToInt32Optimizer', 'ContinuousSliceOptimizer', 'Int64ToInt32Optimizer'])
        optimizer_manager.remove_optimizer('Int64ToInt32Optimizer')
        optimizers = optimizer_manager.get_optimizers()
        self.assertEqual(len(optimizers), 1, ['remove optimizer failed'])

    def test_clearoptimizer(self):
        onnx_model = OnnxGraph("./case6.onnx")
        optimizer_manager = OptimizerManager(onnx_model, optimizers=[
            'Int64ToInt32Optimizer', 'ContinuousSliceOptimizer'])
        optimizer_manager.clear()
        optimizers = optimizer_manager.get_optimizers()
        self.assertEqual(len(optimizers), 0, ['clear optimizer failed'])


def suite():
    suite = unittest.TestSuite()
    # TODO: need to check model been optimized or not
    suite.addTest(OptimizerTestCase("test_configpathmode"))
    suite.addTest(OptimizerTestCase("test_safemode"))
    suite.addTest(OptimizerTestCase("test_allmode"))
    suite.addTest(OptimizerTestCase("test_configdatamode"))
    suite.addTest(OptimizerTestCase("test_addoptimizer"))
    suite.addTest(OptimizerTestCase("test_removeoptimizer"))
    suite.addTest(OptimizerTestCase("test_clearoptimizer"))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest='suite')
