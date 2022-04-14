from magiconnx.graph import OnnxGraph
from magiconnx.optimize.optimizer_manager import OptimizerManager
import unittest


class OptimizerTestCase(unittest.TestCase):
    def setUp(self):
        self.config_json = "./config.json"

    def tearDown(self):
        pass

    def test_configmode(self):
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


def suite():
    suite = unittest.TestSuite()
    # TODO: need to check model been optimized or not
    suite.addTest(OptimizerTestCase("test_configmode"))
    suite.addTest(OptimizerTestCase("test_safemode"))
    suite.addTest(OptimizerTestCase("test_allmode"))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest='suite')
