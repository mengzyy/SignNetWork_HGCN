import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from utils import hyperboloid


class HypAct(Module):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.act = act
        self.hyperboloid = hyperboloid.Hyperboloid()

    def forward(self, x):
        xt = F.relu(self.hyperboloid.logmap0(x, c=self.c_in))
        xt = self.hyperboloid.proj_tan0(xt, c=self.c_out)
        return self.hyperboloid.proj(self.hyperboloid.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )
