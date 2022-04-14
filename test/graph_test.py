from magiconnx.graph import OnnxGraph
import unittest


class OptimizerTestCase(unittest.TestCase):
    def setUp(self):
        self.onnx_model = OnnxGraph("./case6.onnx")

    def tearDown(self):
        pass

    def test_getnextnode(self):
        op_name = 'Add_19'
        next_node_list = self.onnx_model.get_next_nodes(op_name)
        node_name_list = [node.name for node in next_node_list]
        assert len(next_node_list) == 2
        assert 'ReduceMean_20' in node_name_list
        assert 'Sub_21' in node_name_list


def suite():
    suite = unittest.TestSuite()
    suite.addTest(OptimizerTestCase("test_getnextnode"))
    return suite


if __name__ == "__main__":
    unittest.main(defaultTest='suite')
