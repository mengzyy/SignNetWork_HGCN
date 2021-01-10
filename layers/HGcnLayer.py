import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from layers import HypLinear, HypAgg, Hypact


class HGraphConvolution(Module):

    def __init__(self, in_features, out_features, dropout, act, use_bias, c_in, c_out, use_att, local_agg):
        super(HGraphConvolution, self).__init__()
        self.dropout = dropout
        self.act = act
        self.in_features = in_features
        self.out_features = out_features
        self.linear = HypLinear.HypLinear(in_features, out_features, c_in, dropout, use_bias)
        self.agg = HypAgg.HypAgg(c_in, out_features, dropout, use_att, local_agg)
        self.hyp_act = Hypact.HypAct(c_in, c_out, act)

    # 这里的x必须为双曲空间的投影 ，需要提前处理
    def forward(self, x, adj):
        # x为特征 adj为邻接矩阵
        h = self.linear.forward(x)
        # h:3188*16 h:3188*3188
        h = self.agg.forward(h, adj)
        # h:3188*16
        h = self.hyp_act.forward(h)
        return h
