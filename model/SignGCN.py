import model.BaseModel
from layers import GcnLayer
import torch


class SignGCN(model.BaseModel.BaseModel):

    def __init__(self, args):
        super(SignGCN, self).__init__(args)
        # gcn核心层
        # gcn层，需要注意正负分开编码，使用特征大小应该是一半
        self.gcnlayer = GcnLayer.GraphConvolution(int(self.features / 2), int(self.features / 2), args["dropout"],
                                                  args["act"],
                                                  args["use_bias"])

    def encode(self, feature_data_pos, feature_data_neg, adj_pos_matrix, adj_neg_matrix):
        # 正编码 output 正特征 ，gcn卷积三次
        feature_data_pos = self.gcnlayer.forward(feature_data_pos, adj_pos_matrix)
        feature_data_pos = self.gcnlayer.forward(feature_data_pos, adj_pos_matrix)
        feature_data_pos = self.gcnlayer.forward(feature_data_pos, adj_pos_matrix)

        # 负编码 output 负特征 ，gcn卷积三次
        feature_data_neg = self.gcnlayer.forward(feature_data_neg, adj_neg_matrix)
        feature_data_neg = self.gcnlayer.forward(feature_data_neg, adj_neg_matrix)
        feature_data_neg = self.gcnlayer.forward(feature_data_neg, adj_neg_matrix)

        # concat res.shape:self.nodes*self.feature
        res = torch.cat([feature_data_pos, feature_data_neg], dim=1)
        return res

    # loss计算
    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg, final_embedding):
        return super(SignGCN, self).loss(center_nodes, adj_lists_pos, adj_lists_neg, final_embedding)

    # 指标测试
    def test_func(self, adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg, final_embedding):
        return super(SignGCN, self).test_func(adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg,
                                              final_embedding)
