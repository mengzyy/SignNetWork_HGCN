import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, args):
        super(GraphConvolution, self).__init__()
        self.args = args
        self.dropout = self.args["dropout"]
        self.linear = nn.Linear(in_features, out_features, self.args["use_bias"])
        self.linearOneB = nn.Linear(in_features * 2, out_features, self.args["use_bias"])
        self.linearOneH = nn.Linear(in_features * 2, out_features, self.args["use_bias"])
        self.linearTwoB = nn.Linear(in_features * 3, out_features, self.args["use_bias"])
        self.linearTwoH = nn.Linear(in_features * 3, out_features, self.args["use_bias"])
        self.linearThreeB = nn.Linear(in_features * 3, out_features, self.args["use_bias"])
        self.linearThreeH = nn.Linear(in_features * 3, out_features, self.args["use_bias"])
        self.linearFour = nn.Linear(in_features * 2, out_features, self.args["use_bias"])
        self.act = self.args["act"]
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        # x为特征 adj为邻接矩阵
        x = self.linear.forward(x)
        # x = F.dropout(x, self.dropout)
        # 平衡卷积第一层
        layer1_pos = torch.tensor(self.args["data"]["layers_pos_neg_emb"][1][0], dtype=torch.float32)
        layer1_neg = torch.tensor(self.args["data"]["layers_pos_neg_emb"][1][1], dtype=torch.float32)
        h_b_1 = torch.tanh(self.linearOneB(torch.cat([layer1_pos, x], dim=1)))
        h_n_1 = torch.tanh(self.linearOneH(torch.cat([layer1_neg, x], dim=1)))
        # 平衡卷积第二层
        layer2_pos = torch.tensor(self.args["data"]["layers_pos_neg_emb"][2][0], dtype=torch.float32)
        layer2_neg = torch.tensor(self.args["data"]["layers_pos_neg_emb"][2][1], dtype=torch.float32)
        h_b_2 = torch.tanh(self.linearTwoB(torch.cat([layer2_pos, h_b_1], dim=1)))
        h_n_2 = torch.tanh(self.linearTwoH(torch.cat([layer2_neg, h_n_1], dim=1)))
        # 平衡卷积第三层
        layer3_pos = torch.tensor(self.args["data"]["layers_pos_neg_emb"][3][0], dtype=torch.float32)
        layer3_neg = torch.tensor(self.args["data"]["layers_pos_neg_emb"][3][1], dtype=torch.float32)
        h_b_3 = torch.tanh(self.linearThreeB(torch.cat([layer3_pos, h_b_2], dim=1)))
        h_n_3 = torch.tanh(self.linearThreeH(torch.cat([layer3_neg, h_n_2], dim=1)))
        # 全连接层
        output = torch.tanh(self.linearFour(torch.cat([h_b_3, h_n_3], dim=1)))
        return output
