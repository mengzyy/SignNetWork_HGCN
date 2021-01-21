import model.BaseModel
from layers import SignGcnLayer, SignHGcnLayer
import torch
from layers.Hypact import HypAct
from layers.SignHGcnLayer import HSignedConv
from utils import hyperboloid


class SignHGCN(model.BaseModel.BaseModel):

    def __init__(self, in_features, out_features, lambda_structure, num_layers):
        super(SignHGCN, self).__init__(in_features, out_features, lambda_structure)
        self.in_features = in_features
        self.out_features = out_features
        self.lambda_structure = lambda_structure
        self.num_layers = num_layers
        self.hyperboloid = hyperboloid.Hyperboloid()

        # agg 1
        self.conv1 = HSignedConv(in_features, out_features // 2,
                                 first_aggr=True)
        self.hrelu1 = HypAct(0.5, 0.5)

        # agg 2other
        self.convs = torch.nn.ModuleList()
        self.hrelus = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                HSignedConv(out_features // 2, out_features // 2,
                            first_aggr=False))
            self.hrelus.append(HypAct(0.5, 0.5))

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, pos_edge_index, neg_edge_index):

        x = self.hyperboloid.proj_tan0(x, 0.5)
        x = self.hyperboloid.expmap0(x, 0.5)
        x = self.hyperboloid.proj(x, 0.5)

        z = self.hrelu1(self.conv1(x, pos_edge_index, neg_edge_index))
        for i in range(self.num_layers - 1):
            z = self.hrelus[i](self.convs[i](z, pos_edge_index, neg_edge_index))
        return z
