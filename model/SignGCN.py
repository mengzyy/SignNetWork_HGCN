import model.BaseModel
import torch.nn.functional as F
import torch
from SignGcnLayer import SignedConv


class SignGCN(model.BaseModel.BaseModel):

    def __init__(self, in_features, out_features, lambda_structure, num_layers):
        super(SignGCN, self).__init__(in_features, out_features, lambda_structure)
        self.in_features = in_features
        self.out_features = out_features
        self.lambda_structure = lambda_structure
        self.num_layers = num_layers
        # agg 1
        self.conv1 = SignedConv(in_features, out_features // 2,
                                first_aggr=True)
        # agg 2other
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                SignedConv(out_features // 2, out_features // 2,
                           first_aggr=False))

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):
        z = F.relu(self.conv1(x, pos_edge_index, neg_edge_index))
        for conv in self.convs:
            z = F.relu(conv(z, pos_edge_index, neg_edge_index))
        return z
