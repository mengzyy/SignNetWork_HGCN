import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils import hyperboloid


class HypLinear(nn.Module):
    """
    Hyperbolic linear layer.
    """
    def __init__(self, in_features, out_features, c=1, dropout=0, use_bias=False):
        super(HypLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()
        # 工具类
        self.hyperboloid = hyperboloid.Hyperboloid()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        init.constant_(self.bias, 0)

    def forward(self, x):
        drop_weight = F.dropout(self.weight, self.dropout)
        mv = self.hyperboloid.mobius_matvec(drop_weight, x, self.c)
        res = self.hyperboloid.proj(mv, self.c)
        if self.use_bias:
            bias = self.hyperboloid.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.hyperboloid.expmap0(bias, self.c)
            hyp_bias = self.hyperboloid.proj(hyp_bias, self.c)
            res = self.hyperboloid.mobius_add(res, hyp_bias, c=self.c)
            res = self.hyperboloid.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )
