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
        self.first_aggr = first_aggr
        self.in_features = in_features

        # att
        self.posAtt = posAtt
        self.negAtt = negAtt
        self.pos_att_layer = None
        self.neg_att_layer = None

        # 注意力存放容器 如果采用权重法 则改为训练向量
        self.posAttMat2 = [1.0] * self.posAtt
        self.negAttMat2 = [1.0] * self.negAtt
        # self.posAttMat2 = torch.nn.Parameter(torch.FloatTensor([1] * self.posAtt), requires_grad=True)
        # self.negAttMat2 = torch.nn.Parameter(torch.FloatTensor([1] * self.negAtt), requires_grad=True)

        if first_aggr:
            self.lin_pos = Linear(in_features, out_features, False)
            self.lin_pos_cc = Linear(in_features, out_features, True)
            self.lin_neg = Linear(in_features, out_features, False)
            self.lin_neg_cc = Linear(in_features, out_features, True)
            # 注意力计分层  64*64
            self.pos_att_layer = Linear(in_features, out_features * 2, True)
            self.neg_att_layer = Linear(in_features, out_features * 2, True)

        else:
            self.lin_pos = Linear(in_features * 2, out_features, False)
            self.lin_pos_cc = Linear(in_features, out_features, True)
            self.lin_neg = Linear(in_features * 2, out_features, False)
            self.lin_neg_cc = Linear(in_features, out_features, True)
            # 注意力计分层  64*64
            self.pos_att_layer = Linear(in_features * 2, out_features * 2, True)
            self.neg_att_layer = Linear(in_features * 2, out_features * 2, True)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_pos_cc.reset_parameters()
        self.lin_neg.reset_parameters()
        self.lin_neg_cc.reset_parameters()

        # att
        self.pos_att_layer.reset_parameters()
        self.neg_att_layer.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        x, y1, y2 = x, x, x
        pos_att_mat = self.pos_att_layer(x)
        neg_att_mat = self.neg_att_layer(x)
        for i in range(0, self.posAtt):
            # 对于每条pos边
            r, l = pos_edge_index[0][i], pos_edge_index[1][i]
            temp_val = torch.dot(pos_att_mat[r], (x[l].t()))
            self.posAttMat2[i] = temp_val.item()
        for i in range(0, self.negAtt):
            # 对于每条neg边
            r, l = neg_edge_index[0][i], neg_edge_index[1][i]
            temp_val = torch.dot(neg_att_mat[r], (x[l].t()))
            self.negAttMat2[i] = temp_val.item()

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
