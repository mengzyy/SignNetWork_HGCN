import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from  utils import hyperboloid


class HypAct(Module):
    def __init__(self):
        super(HypAct, self).__init__()
        self.hyperboloid = hyperboloid.Hyperboloid()
    def forward(self, x,c_in,c_out):
        xt = F.relu(self.hyperboloid.logmap0(x, c=c_in))
        xt = self.hyperboloid.proj_tan0(xt, c=c_out)
        return self.hyperboloid.proj(self.hyperboloid.expmap0(xt, c=c_out), c=c_out)
