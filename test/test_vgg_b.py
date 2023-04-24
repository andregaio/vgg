import unittest
import torch
from src.layers.vgg_B import VGG_B
from src.utils import export_onnx, count_parameters


BATCH = 1
HEIGHT = 256
WIDTH = 256
IN_CHANNELS = 3


x = torch.ones(BATCH, IN_CHANNELS, HEIGHT, WIDTH)


class TestVGG_B(unittest.TestCase):

    def test_default(self):
        model = VGG_B()
        y = model(x)
        print('In:', x.shape, 'Out:', y.shape)

    def test_export(self):
        model = VGG_B()
        export_onnx(model, x, 'exports/vgg_B.onnx')

    def test_model_summmary(self):
        model = VGG_B()
        count_parameters(model)

if __name__ == '__main__':
    unittest.main()