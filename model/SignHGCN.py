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

        # agg 2other
        self.convs = torch.nn.ModuleList()
        self.hrelus = torch.nn.ModuleList()

        # 可训练曲率list
        self.c = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.clist = []
        for i in range(num_layers):
            self.clist.append([torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True),
                               torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)])

        # self.hrelu1 = HypAct(self.clist[0][0], self.clist[0][1])
        # for i in range(num_layers - 1):
        #     self.convs.append(
        #         HSignedConv(out_features // 2, out_features // 2,
        #                     first_aggr=False))
        #     self.hrelus.append(HypAct(self.clist[i + 1][0], self.clist[i + 1][1]))
        self.updatec()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
        self.c.data.fill_(0.5)
        for ct in self.clist:
            ct[0].data.fill_(0.5)
            ct[1].data.fill_(0.5)

    def forward(self, x, pos_edge_index, neg_edge_index):
        self.updatec()
        x = self.hyperboloid.proj_tan0(x, self.c)
        x = self.hyperboloid.expmap0(x, self.c)
        x = self.hyperboloid.proj(x, self.c)
        z = self.hrelu1(self.conv1(x, pos_edge_index, neg_edge_index))
        for i in range(self.num_layers - 1):
            z = self.hrelus[i](self.convs[i](z, pos_edge_index, neg_edge_index))
        return z

    def updatec(self):
        self.hrelu1 = HypAct(self.clist[0][0], self.clist[0][1])
        for i in range(self.num_layers - 1):
            self.convs.append(
                HSignedConv(self.out_features // 2, self.out_features // 2,
                            first_aggr=False))
            self.hrelus.append(HypAct(self.clist[i + 1][0], self.clist[i + 1][1]))
