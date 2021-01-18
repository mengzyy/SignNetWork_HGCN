import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from layers import HypLinear, HypAgg, Hypact
import utils.loadData as LD


class HGraphConvolution(Module):

    def __init__(self, in_features, out_features, args):
        super(HGraphConvolution, self).__init__()
        self.dropout = args["dropout"]
        self.act = args["act"]
        self.use_bias = args["use_bias"]
        self.args = args
        self.c_in = 1
        self.c_out = 1

        self.linearOneB = HypLinear.HypLinear(in_features * 2, out_features, self.c_in, self.dropout, self.use_bias)
        self.linearOneH = HypLinear.HypLinear(in_features * 2, out_features, self.c_in, self.dropout, self.use_bias)
        self.linearTwoB = HypLinear.HypLinear(in_features * 3, out_features * 2, self.c_in, self.dropout, self.use_bias)
        self.linearTwoH = HypLinear.HypLinear(in_features * 3, out_features * 2, self.c_in, self.dropout, self.use_bias)
        self.linearThreeB = HypLinear.HypLinear(in_features * 2, out_features, self.c_in, self.dropout, self.use_bias)
        self.linearThreeH = HypLinear.HypLinear(in_features * 2, out_features, self.c_in, self.dropout, self.use_bias)

        self.hyp_act1 = Hypact.HypAct(self.c_in, self.c_out, self.act)
        self.hyp_act2 = Hypact.HypAct(self.c_in, self.c_out, self.act)
        self.hyp_act3 = Hypact.HypAct(self.c_in, self.c_out, self.act)
        self.hyp_act4 = Hypact.HypAct(self.c_in, self.c_out, self.act)
        self.hyp_act5 = Hypact.HypAct(self.c_in, self.c_out, self.act)
        self.hyp_act6 = Hypact.HypAct(self.c_in, self.c_out, self.act)

    def forward(self, x):
        _, r = x.shape
        pos_emb_1 = []
        neg_emb_1 = []
        for node in range(0, _):
            node += 1
            pos_emb_1_temp, neg_emb_1_temp = LD.getNodePosAndNegEmbegingByDiffLayer(self.args["data"],
                                                                                    1, node,
                                                                                    x.detach().numpy(),
                                                                                    self.args["data"]["adj_lists_pos"],
                                                                                    self.args["data"]["adj_lists_neg"])
            pos_emb_1.append(pos_emb_1_temp)
            neg_emb_1.append(neg_emb_1_temp)

        # input: N*2d output: N*d
        h_b_1 = self.hyp_act1(self.linearOneB(torch.cat([torch.tensor(pos_emb_1, dtype=torch.float32), x], dim=1)))
        # N*2d
        h_n_1 = self.hyp_act2(self.linearOneH(torch.cat([torch.tensor(neg_emb_1, dtype=torch.float32), x], dim=1)))
        # 平衡卷积第二层
        pos_emb_2 = []
        neg_emb_2 = []
        for node in range(0, _):
            node += 1
            pos_emb_2_temp, neg_emb_2_temp = LD.getNodePosAndNegEmbegingByDiffLayer(self.args["data"],
                                                                                    2, node,
                                                                                    [h_b_1.detach().numpy(),
                                                                                     h_n_1.detach().numpy()],
                                                                                    self.args["data"]["adj_lists_pos"],
                                                                                    self.args["data"]["adj_lists_neg"])
            pos_emb_2.append(pos_emb_2_temp)
            neg_emb_2.append(neg_emb_2_temp)

        # input: N*3d output:N*2d
        h_b_2 = self.hyp_act3(self.linearTwoB(torch.cat([torch.tensor(pos_emb_2, dtype=torch.float32), h_b_1], dim=1)))
        h_n_2 = self.hyp_act4(self.linearTwoH(torch.cat([torch.tensor(neg_emb_2, dtype=torch.float32), h_n_1], dim=1)))
        # input: N*4d
        h_b_3 = self.hyp_act5(self.linearThreeB(h_b_2))
        h_n_3 = self.hyp_act6(self.linearThreeH(h_n_2))
        h = torch.cat([h_b_3, h_n_3, x], dim=1)

        return h
