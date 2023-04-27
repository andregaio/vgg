import unittest
import torch
from src.model import VGG
from src.model import Networks
from src.utils import export_onnx, count_parameters


BATCH = 1
HEIGHT = 256
WIDTH = 256
IN_CHANNELS = 3
MODEL = Networks.vgg_A


x = torch.ones(BATCH, IN_CHANNELS, HEIGHT, WIDTH)


class TestVGG(unittest.TestCase):

    def test_default(self):
        model = VGG(MODEL)
        y = model(x)
        print('In:', x.shape, 'Out:', y.shape)

    def test_model_summmary(self):
        model = VGG(MODEL)
        count_parameters(model)

if __name__ == '__main__':
    unittest.main()