import model.BaseModel
from layers import GcnLayer, HGcnLayer
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from utils import hyperboloid


class SignHGCN(model.BaseModel.BaseModel):

    def __init__(self, args):
        super(SignHGCN, self).__init__(args)
        # 曲率
        self.c = torch.tensor(args["c"])
        self.hyperboloid = hyperboloid.Hyperboloid()
        # gcn核心层
        # gcn层，需要注意正负分开编码，使用特征大小应该是一半+1
        self.HGraphConvolution = HGcnLayer.HGraphConvolution(int((self.features / 2) + 1), int((self.features / 2) + 1),
                                                             args["dropout"],
                                                             args["act"],
                                                             args["use_bias"], self.c, self.c, args["use_att"],
                                                             args["local_agg"], )

    def encode(self, feature_data_pos, feature_data_neg, adj_pos_matrix, adj_neg_matrix):
        # 特征需要先获得在HGCN空间的映射
        o = torch.zeros_like(feature_data_pos)
        feature_data_pos = torch.cat([o[:, 0:1], feature_data_pos], dim=1)
        x = torch.zeros_like(feature_data_neg)
        feature_data_neg = torch.cat([x[:, 0:1], feature_data_neg], dim=1)
        feature_data_pos = self.hyperboloid.proj_tan0(feature_data_pos, self.c)
        feature_data_pos = self.hyperboloid.expmap0(feature_data_pos, c=self.c)
        feature_data_pos = self.hyperboloid.proj(feature_data_pos, c=self.c)
        feature_data_neg = self.hyperboloid.proj_tan0(feature_data_neg, self.c)
        feature_data_neg = self.hyperboloid.expmap0(feature_data_neg, c=self.c)
        feature_data_neg = self.hyperboloid.proj(feature_data_neg, c=self.c)

        # 正编码 output 正特征 ，hgcn卷积三次
        feature_data_pos = self.HGraphConvolution.forward(feature_data_pos, adj_pos_matrix)
        feature_data_pos = self.HGraphConvolution.forward(feature_data_pos, adj_pos_matrix)
        feature_data_pos = self.HGraphConvolution.forward(feature_data_pos, adj_pos_matrix)
        # 负编码 output 负特征 ，hgcn卷积三次
        feature_data_neg = self.HGraphConvolution.forward(feature_data_neg, adj_neg_matrix)
        feature_data_neg = self.HGraphConvolution.forward(feature_data_neg, adj_neg_matrix)
        feature_data_neg = self.HGraphConvolution.forward(feature_data_neg, adj_neg_matrix)
        # concat res.shape:self.nodes*self.feature
        res = torch.cat([feature_data_pos, feature_data_neg], dim=1)
        return res

    # loss计算
    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg, final_embedding):
        return super(SignHGCN, self).loss(center_nodes, adj_lists_pos, adj_lists_neg, final_embedding)

    # 指标测试
    def test_func(self, adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg, final_embedding):
        return super(SignHGCN, self).test_func(adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg,
                                               final_embedding)
