import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import utils.loadData as LD


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, args):
        super(GraphConvolution, self).__init__()
        self.args = args
        self.dropout = self.args["dropout"]
        self.linear = nn.Linear(in_features, out_features, bias=args["use_bias"])
        self.linearOneB = nn.Linear(in_features * 2, int(out_features / 2), bias=args["use_bias"])
        self.linearOneH = nn.Linear(in_features * 2, int(out_features / 2), bias=args["use_bias"])
        self.linearTwoB = nn.Linear(int(in_features * 1.5), out_features, bias=args["use_bias"])
        self.linearTwoH = nn.Linear(int(in_features * 1.5), out_features, bias=args["use_bias"])
        self.linearThreeB = nn.Linear(in_features, out_features, bias=args["use_bias"])
        self.linearThreeH = nn.Linear(in_features, out_features, bias=args["use_bias"])
        # self.linearFour = nn.Linear(in
        # _features * 4, out_features * 3,bias=args["use_bias"])
        self.act = self.args["act"]
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # x为特征 adj为邻接矩阵
        # x = torch.tanh(self.linear.forward(x))
        # x = F.dropout(x, self.dropout)

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
        h_b_1 = torch.tanh(self.linearOneB(torch.cat([torch.tensor(pos_emb_1, dtype=torch.float32), x], dim=1)))
        # N*2d
        h_n_1 = torch.tanh(self.linearOneH(torch.cat([torch.tensor(neg_emb_1, dtype=torch.float32), x], dim=1)))

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
        h_b_2 = torch.tanh(self.linearTwoB(torch.cat([torch.tensor(pos_emb_2, dtype=torch.float32), h_b_1], dim=1)))
        h_n_2 = torch.tanh(self.linearTwoH(torch.cat([torch.tensor(neg_emb_2, dtype=torch.float32), h_n_1], dim=1)))
        # input: N*4d
        # h_b_3 = torch.tanh(self.linearThreeB(h_b_2))
        # h_n_3 = torch.tanh(self.linearThreeH(h_n_2))
        # output = torch.tanh(self.linearFour(torch.cat([h_b_2, h_n_2], dim=1)))
        output = torch.cat([h_b_2, h_n_2], dim=1)
        return output
