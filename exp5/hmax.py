import torch
import torch.nn as nn
import torch.nn.functional as F
from gcn.layers import GConv


class HMax(nn.Module):
    def __init__(self):
        super(HMax, self).__init__()
        ...

    def forward(self, x):
        ...
