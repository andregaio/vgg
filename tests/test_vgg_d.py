import unittest
import torch
from layers.vgg_D import VGG_D
from utils import export_onnx, count_parameters


BATCH = 1
HEIGHT = 256
WIDTH = 256
IN_CHANNELS = 3


x = torch.ones(BATCH, IN_CHANNELS, HEIGHT, WIDTH)


class TestVGG_C(unittest.TestCase):

    def test_default(self):
        model = VGG_D()
        y = model(x)
        print('In:', x.shape, 'Out:', y.shape)

    def test_export(self):
        model = VGG_D()
        export_onnx(model, x, 'exports/vgg_D.onnx')

    def test_model_summmary(self):
        model = VGG_D()
        count_parameters(model)

if __name__ == '__main__':
    unittest.main()