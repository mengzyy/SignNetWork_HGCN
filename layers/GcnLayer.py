import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):

    def __init__(self, in_features, out_features, dropout, act, use_bias):
        super(GraphConvolution, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.act = act
        self.in_features = in_features
        self.out_features = out_features


    def forward(self, x, adj):
        # x为特征 adj为邻接矩阵
        hidden = self.linear.forward(x)
        hidden = F.dropout(hidden, self.dropout)
        # A * X * W  adj：A  hidden ：X
        support = torch.mm(adj, hidden)
        output = F.relu(support)
        return output
