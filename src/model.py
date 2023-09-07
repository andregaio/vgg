import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from layers.vgg_A import VGG_A
from layers.vgg_A_LRN import VGG_A_LRN
from layers.vgg_B import VGG_B
from layers.vgg_C import VGG_C
from layers.vgg_D import VGG_D


class Networks(Enum):
    vgg_A = VGG_A
    vgg_A_LRN = VGG_A_LRN
    vgg_B = VGG_B
    vgg_C = VGG_C
    vgg_D = VGG_D


class VGG(nn.Module):

    def __init__(self, network = VGG_A):
        super(VGG, self).__init__()
        self.network = VGG_A()


    def forward(self, x):
        return self.network(x)