import model.BaseModel
from layers import GcnLayer, HGcnLayer
import torch
from layers import GcnLayer
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.modules.module import Module
from utils import hyperboloid
from layers import HGcnLayer


class SignHGCN(model.BaseModel.BaseModel):

    def __init__(self, args):
        super(SignHGCN, self).__init__(args)
        # 曲率
        self.c = torch.tensor(args["c"])
        self.hyperboloid = hyperboloid.Hyperboloid()
        self.hgcnLayer=HGcnLayer.HGraphConvolution(self.features,  self.features, self.args)

    def encode(self):
        x = torch.tensor(self.args["data"]["feat_data"], dtype=torch.float32)
        # 特征需要先获得在HGCN空间的映射
        x = self.hyperboloid.proj_tan0(x, self.c)
        x = self.hyperboloid.expmap0(x, c=self.c)
        x = self.hyperboloid.proj(x, c=self.c)

        res = self.hgcnLayer(x)
        return res

    # loss计算
    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg, final_embedding):
        return super(SignHGCN, self).loss(center_nodes, adj_lists_pos, adj_lists_neg, final_embedding)

    # 指标测试
    def test_func(self, adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg, final_embedding):
        return super(SignHGCN, self).test_func(adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg,
                                               final_embedding)
