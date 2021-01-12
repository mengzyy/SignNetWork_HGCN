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
        self.linear = nn.Linear(in_features, out_features, self.args["use_bias"])
        self.linearOneB = nn.Linear(in_features * 2, out_features, self.args["use_bias"])
        self.linearOneH = nn.Linear(in_features * 2, out_features, self.args["use_bias"])
        self.linearTwoB = nn.Linear(in_features * 3, out_features * 2, self.args["use_bias"])
        self.linearTwoH = nn.Linear(in_features * 3, out_features * 2, self.args["use_bias"])
        self.linearFour = nn.Linear(in_features * 4, out_features, self.args["use_bias"])
        self.act = self.args["act"]
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # x为特征 adj为邻接矩阵
        # x = self.linear.forward(x)
        # x = F.dropout(x, self.dropout)
        # 平衡卷积第一层
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
        # 平衡卷积第二层
        pos_emb_2 = []
        neg_emb_2 = []
        for node in range(0, _):
            node += 1
            pos_emb_2_temp, neg_emb_2_temp = LD.getNodePosAndNegEmbegingByDiffLayer(self.args["data"],
                                                                                    2, node,
                                                                                    [h_b_1.detach().numpy(), h_n_1.detach().numpy()],
                                                                                    self.args["data"]["adj_lists_pos"],
                                                                                    self.args["data"]["adj_lists_neg"])
            pos_emb_2.append(pos_emb_2_temp)
            neg_emb_2.append(neg_emb_2_temp)

        # input: N*3d output:N*2d
        h_b_2 = torch.tanh(self.linearTwoB(torch.cat([torch.tensor(pos_emb_2, dtype=torch.float32), h_b_1], dim=1)))
        h_n_2 = torch.tanh(self.linearTwoH(torch.cat([torch.tensor(neg_emb_2, dtype=torch.float32), h_n_1], dim=1)))
        # input: N*4d
        output = torch.tanh(self.linearFour(torch.cat([h_b_2, h_n_2], dim=1)))
        return output
