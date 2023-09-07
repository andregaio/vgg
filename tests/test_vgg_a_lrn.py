import unittest
import torch
from layers.vgg_A_LRN import VGG_A_LRN
from utils import export_onnx, count_parameters


BATCH = 1
HEIGHT = 256
WIDTH = 256
IN_CHANNELS = 3


x = torch.ones(BATCH, IN_CHANNELS, HEIGHT, WIDTH)


class TestVGG_A_LRN(unittest.TestCase):

    def test_default(self):
        model = VGG_A_LRN()
        y = model(x)
        print('In:', x.shape, 'Out:', y.shape)

    def test_export(self):
        model = VGG_A_LRN()
        export_onnx(model, x, 'exports/vgg_A_LRN.onnx')

    def test_model_summmary(self):
        model = VGG_A_LRN()
        count_parameters(model)

if __name__ == '__main__':
    unittest.main()