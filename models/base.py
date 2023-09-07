import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum
from models.vgg_A import VGG_A
from models.vgg_A_LRN import VGG_A_LRN
from models.vgg_B import VGG_B
from models.vgg_C import VGG_C
from models.vgg_D import VGG_D


networks = {
    'vgg_A' : VGG_A,
    'vgg_A_LRN' : VGG_A_LRN,
    'vgg_B' : VGG_B,
    'vgg_C' : VGG_C,
    'vgg_D' : VGG_D
}


def _kaiming_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(layer.weight)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


class VGG(nn.Module):
    
    def __init__(self, classes = 10, network = 'vgg_A'):
        super(VGG, self).__init__()
        model = networks[network]
        self.network = model(classes)
        _kaiming_init(self.network)


    def forward(self, x):
        return self.network(x)