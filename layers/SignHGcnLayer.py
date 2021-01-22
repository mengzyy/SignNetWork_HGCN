import torch
from torch.nn.modules.module import Module
from layers import HypLinear
from layers.aggmethod import computeNegFeaMean

class HSignedConv(Module):

    def __init__(self, in_features, out_features, c=1,first_aggr=True):
        super(HSignedConv, self).__init__()
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
        self.c=c

        if first_aggr:
            self.lin_pos = HypLinear.HypLinear(in_features=in_features, out_features=out_features,c=self.c, use_bias=False)
            self.lin_pos_cc = HypLinear.HypLinear(in_features=in_features, out_features=out_features, c=self.c,use_bias=True)
            self.lin_neg = HypLinear.HypLinear(in_features=in_features, out_features=out_features,c=self.c, use_bias=False)
            self.lin_neg_cc = HypLinear.HypLinear(in_features=in_features, out_features=out_features,c=self.c, use_bias=True)
        else:
            self.lin_pos = HypLinear.HypLinear(in_features=in_features * 2, out_features=out_features, c=self.c,use_bias=False)
            self.lin_pos_cc = HypLinear.HypLinear(in_features=in_features, out_features=out_features,c=self.c, use_bias=True)
            self.lin_neg = HypLinear.HypLinear(in_features=in_features * 2, out_features=out_features, c=self.c,use_bias=False)
            self.lin_neg_cc = HypLinear.HypLinear(in_features=in_features, out_features=out_features, c=self.c,use_bias=True)
        self.reset_parameters()


    def reset_parameters(self):
        self.lin_pos.reset_parameters()
        self.lin_pos_cc.reset_parameters()
        self.lin_neg.reset_parameters()
        self.lin_neg_cc.reset_parameters()

    def forward(self, x,pos_edge_index, neg_edge_index):
        x, y1, y2 = x, x, x
        if self.first_aggr:
            out_pos = computeNegFeaMean(pos_edge_index, x=y1)
            out_pos = self.lin_pos(out_pos)
            out_pos += self.lin_pos_cc(x)
            out_neg = computeNegFeaMean(neg_edge_index, x=y2)
            out_neg = self.lin_neg(out_neg)
            out_neg += self.lin_neg_cc(x)
            return torch.cat([out_pos, out_neg], dim=-1)
        else:
            F_in = self.in_features
            out_pos1 = computeNegFeaMean(pos_edge_index, x=(y1[..., :F_in]))
            out_pos2 = computeNegFeaMean(neg_edge_index, x=(y2[..., F_in:]))
            out_pos = torch.cat([out_pos1, out_pos2], dim=-1)
            out_pos = self.lin_pos(out_pos)
            out_pos += self.lin_pos_cc(x[..., :F_in])

            out_neg1 = computeNegFeaMean(pos_edge_index, x=(x[..., F_in:]))
            out_neg2 = computeNegFeaMean(neg_edge_index, x=(x[..., :F_in]))
            out_neg = torch.cat([out_neg1, out_neg2], dim=-1)
            out_neg = self.lin_neg(out_neg)
            out_neg += self.lin_neg_cc(x[..., F_in:])
            return torch.cat([out_pos, out_neg], dim=-1)

