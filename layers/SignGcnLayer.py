import torch
from torch.nn.modules.module import Module
from torch.nn import Linear
import numpy as np
from layers.aggmethod import computeNegFeaMean
from torch.nn.parameter import Parameter
import torch.nn.functional as F

class SignedConv(Module):
    def __init__(self, in_features, out_features, posAtt, negAtt, first_aggr=True):
        super(SignedConv, self).__init__()
        # pos ,activate
        self.lin_pos = None
        # pos,residual
        self.lin_pos_cc = None
        # neg ,activate
        self.lin_neg = None
        # neg,residual
        self.lin_neg_cc = None
        # att
        self.pos_att_layer = None
        self.neg_att_layer = None
        self.first_aggr = first_aggr
        self.in_features = in_features
        self.posAtt = posAtt
        self.negAtt = negAtt
        self.posAttMat = None
        self.negAttMat = None
        self.posAttToRank = None
        self.negAttToRank = None

        if first_aggr:
            self.lin_pos = Linear(in_features, out_features, False)
            self.lin_pos_cc = Linear(in_features, out_features, True)
            self.lin_neg = Linear(in_features, out_features, False)
            self.lin_neg_cc = Linear(in_features, out_features, True)
            # att
            self.pos_att_layer = Linear(in_features, out_features, True)
            self.neg_att_layer = Linear(in_features, out_features, True)
            # 权重参数容器
            self.posAttMat1 = torch.nn.Parameter(torch.Tensor(self.posAtt, out_features * 2), requires_grad=False)
            # 权重参数容器
            self.negAttMat1 = torch.nn.Parameter(torch.Tensor(self.negAtt, out_features * 2), requires_grad=False)
            self.posAttToRank = Linear(out_features * 2, 1, True)
            self.negAttToRank = Linear(out_features * 2, 1, True)
        else:
            self.lin_pos = Linear(in_features * 2, out_features, False)
            self.lin_pos_cc = Linear(in_features, out_features, True)
            self.lin_neg = Linear(in_features * 2, out_features, False)
            self.lin_neg_cc = Linear(in_features, out_features, True)
            # att
            self.pos_att_layer = Linear(in_features * 2, out_features, True)
            self.neg_att_layer = Linear(in_features * 2, out_features, True)
            # 权重参数
            self.posAttMat1 = torch.nn.Parameter(torch.Tensor(self.posAtt, out_features * 2), requires_grad=False)
            # 权重参数
            self.negAttMat1 = torch.nn.Parameter(torch.Tensor(self.negAtt, out_features * 2), requires_grad=False)
            self.posAttToRank = Linear(out_features * 2, 1, True)
            self.negAttToRank = Linear(out_features * 2, 1, True)
        self.posAttMat2 = torch.nn.Parameter(torch.FloatTensor([1] * self.posAtt), requires_grad=True)
        self.negAttMat2 = torch.nn.Parameter(torch.FloatTensor([1] * self.negAtt), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_pos_cc.reset_parameters()
        self.lin_neg.reset_parameters()
        self.lin_neg_cc.reset_parameters()
        # att
        self.pos_att_layer.reset_parameters()
        self.neg_att_layer.reset_parameters()
        self.posAttToRank.reset_parameters()
        self.negAttToRank.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        x, y1, y2 = x, x, x

        # pos_att = self.pos_att_layer(x)
        # neg_att = self.neg_att_layer(x)
        # att = torch.cat((pos_att, neg_att), 1)
        # for i in range(0, self.posAtt):
        #     r, l = pos_edge_index[0][i], pos_edge_index[1][i]
        #     self.posAttMat1[i] = torch.cat([pos_att[r], pos_att[l]])
        # for i in range(0, self.negAtt):
        #     r, l = neg_edge_index[0][i], neg_edge_index[1][i]
        #     self.negAttMat1[i] = torch.cat([neg_att[r], neg_att[l]])
        # 赋值
        # self.posAttMat2 = Parameter(self.posAttToRank(self.posAttMat1))
        # self.negAttMat2 = Parameter(self.negAttToRank(self.negAttMat1))
        #归一化
        # self.posAttMat2 = F.normalize(self.posAttMat2 , p=2, dim=1)
        # self.posAttMat2 = F.normalize(self.posAttMat2, p=2, dim=1)

        if self.first_aggr:
            out_pos = computeNegFeaMean(pos_edge_index, self.posAttMat2, x=y1)
            out_pos = self.lin_pos(out_pos)
            out_pos += self.lin_pos_cc(x)
            out_neg = computeNegFeaMean(neg_edge_index, self.negAttMat2, x=y2)
            out_neg = self.lin_neg(out_neg)
            # 类似残差连接

            out_neg += self.lin_neg_cc(x)
            return torch.cat([out_pos, out_neg], dim=-1)
        else:
            F_in = self.in_features
            out_pos1 = computeNegFeaMean(pos_edge_index, self.posAttMat2, x=(y1[..., :F_in]))
            out_pos2 = computeNegFeaMean(neg_edge_index, self.negAttMat2, x=(y2[..., F_in:]))
            out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
            out_pos = self.lin_pos(out_pos)
            out_pos += self.lin_pos_cc(x[..., :F_in])

            out_neg1 = computeNegFeaMean(pos_edge_index, self.posAttMat2, x=(x[..., F_in:]))
            out_neg2 = computeNegFeaMean(neg_edge_index, self.negAttMat2, x=(x[..., :F_in]))
            out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
            out_neg = self.lin_neg(out_neg)
            out_neg += self.lin_neg_cc(x[..., F_in:])
            return torch.cat([out_pos, out_neg], dim=-1)
