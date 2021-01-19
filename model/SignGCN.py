import model.BaseModel
from layers import GcnLayer
import torch
import torch.nn as nn
from torch.nn import init

class SignGCN(model.BaseModel.BaseModel):

    def __init__(self, args):
        super(SignGCN, self).__init__(args)
        # gcn核心层
        # gcn层,使用平衡理论
        self.gcnlayer = GcnLayer.GraphConvolution(self.features, self.features, self.args)

    def encode(self):
        x = torch.tensor(self.args["data"]["feat_data"],dtype=torch.float32)
        res = self.gcnlayer(x)
        return res

    # loss计算
    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg, final_embedding):
        return super(SignGCN, self).loss(center_nodes, adj_lists_pos, adj_lists_neg, final_embedding)

    # 指标测试
    def test_func(self, adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg, final_embedding):
        return super(SignGCN, self).test_func(adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg,
                                              final_embedding)
