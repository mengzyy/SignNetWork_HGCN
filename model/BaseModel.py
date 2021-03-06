import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from torch_sparse import coalesce
import scipy.sparse
import torch.nn.functional as F
from torch_geometric.utils import (negative_sampling,
                                   structured_negative_sampling)

from sklearn.metrics import roc_auc_score, f1_score
from utils import hyperboloid
from model.MutualNet import MutualInfoNet


# base
class BaseModel(nn.Module):
    def __init__(self, feature, hidden_channels, lambda_structure, posAtt, negAtt):
        super(BaseModel, self).__init__()
        self.feature = feature
        self.lambda_structure = lambda_structure
        self.lin = torch.nn.Linear(2 * hidden_channels, 3)
        self.posAtt = posAtt
        self.negAtt = negAtt
        self.hyperboloid = hyperboloid.Hyperboloid()
        self.c = 3
        self.gamma=0.1

        # 互信息层
        self.info_net = MutualInfoNet(2 * (hidden_channels+1))

    def create_spectral_features(self, pos_edge_index, neg_edge_index,
                                 num_nodes=None):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        N = edge_index.max().item() + 1 if num_nodes is None else num_nodes
        edge_index = edge_index.to(torch.device('cpu'))

        pos_val = torch.full((pos_edge_index.size(1),), 2, dtype=torch.float)
        neg_val = torch.full((neg_edge_index.size(1),), 0, dtype=torch.float)
        val = torch.cat([pos_val, neg_val], dim=0)

        row, col = edge_index
        edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        val = torch.cat([val, val], dim=0)

        edge_index, val = coalesce(edge_index, val, N, N)
        val = val - 1

        edge_index = edge_index.detach().numpy()
        val = val.detach().numpy()
        A = scipy.sparse.coo_matrix((val, edge_index), shape=(N, N))
        svd = TruncatedSVD(n_components=self.feature, n_iter=128)
        svd.fit(A)
        x = svd.components_.T
        return torch.from_numpy(x).to(torch.float).to(pos_edge_index.device)

    def loss(self, z, pos_edge_index, neg_edge_index):
        # 【1.映射至双曲空间 2.算距离】
        # 映射双曲时需要加一个特征点
        vals = torch.cat([z[:, 0:1], z], 1)
        # 映射至双曲
        x = self.hyperboloid.proj_tan0(vals, self.c)
        x = self.hyperboloid.expmap0(x, self.c)
        x = self.hyperboloid.proj(x, self.c)
        nll_loss = self.nll_loss(z, pos_edge_index, neg_edge_index)
        loss_1 = self.pos_embedding_loss(x, pos_edge_index)
        loss_2 = self.neg_embedding_loss(x, neg_edge_index)
        loss_3 =self.computeMutual(x, pos_edge_index, neg_edge_index)

        return nll_loss + self.lambda_structure * (loss_1 + loss_2)+self.gamma*loss_3

    def nll_loss(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        nll_loss = 0
        nll_loss += F.nll_loss(
            self.discriminate(z, pos_edge_index),
            pos_edge_index.new_full((pos_edge_index.size(1),), 0))
        nll_loss += F.nll_loss(
            self.discriminate(z, neg_edge_index),
            neg_edge_index.new_full((neg_edge_index.size(1),), 1))
        nll_loss += F.nll_loss(
            self.discriminate(z, none_edge_index),
            none_edge_index.new_full((none_edge_index.size(1),), 2))
        return nll_loss / 3.0

    def pos_embedding_loss(self, z, pos_edge_index):
        i, j, k = structured_negative_sampling(pos_edge_index, z.size(0))
        hdis1 = self.hyperboloid.sqdist(z[i], z[j], self.c)
        hdis2 = self.hyperboloid.sqdist(z[i], z[k], self.c)
        out = hdis1 - hdis2
        # out = (z[i] - z[j]).pow(2).sum(dim=1) - (z[i] - z[k]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def neg_embedding_loss(self, z, neg_edge_index):
        i, j, k = structured_negative_sampling(neg_edge_index, z.size(0))
        hdis1 = self.hyperboloid.sqdist(z[i], z[k], self.c)
        hdis2 = self.hyperboloid.sqdist(z[i], z[j], self.c)
        out = hdis1 - hdis2
        # out = (z[i] - z[k]).pow(2).sum(dim=1) - (z[i] - z[j]).pow(2).sum(dim=1)
        return torch.clamp(out, min=0).mean()

    def discriminate(self, z, edge_index):
        value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
        value = self.lin(value)
        return torch.log_softmax(value, dim=1)

    def computeMutual(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        none_edge_index = negative_sampling(edge_index, z.size(0))

        pos_y = pos_edge_index.new_full((pos_edge_index.size(1),), 0).float()
        neg_y = neg_edge_index.new_full((neg_edge_index.size(1),), 1).float()
        neu_y = none_edge_index.new_full((none_edge_index.size(1),), 2).float()
        all_y = torch.cat((pos_y, neg_y, neu_y))
        idx = torch.randperm(all_y.size()[0])
        shuffle_y = all_y[idx]
        index = torch.cat((pos_edge_index, neg_edge_index, none_edge_index), 1)
        info_pred = self.discriminate2(z, index, id=all_y)
        info_shuffle = self.discriminate2(z, index, id=shuffle_y)
        mutual_loss = torch.mean(info_pred) - torch.log(torch.mean(torch.exp(info_shuffle)))
        return -mutual_loss

    def discriminate2(self, z, edge_index, id=None, last=False):
        """
        Args:
            x (Tensor): The input node features.
            edge_index (LongTensor): The edge indices.
        """
        if id is not None:
            value = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
            out = self.info_net(value, id)

        else:
            out = torch.clamp_min(1. / (torch.exp(
                (self.hyperboloid.sqdist(z[edge_index[0]], z[edge_index[1]], 1) - self.r) / self.t) + 1.0), 0)
        return out

    def test(self, z, pos_edge_index, neg_edge_index):
        # 【1.映射至双曲空间 2.算距离】
        # 映射双曲时需要加一个特征点
        # vals = torch.cat([z[:, 0:1], z], 1)
        # 映射至双曲
        # x = self.hyperboloid.proj_tan0(vals, self.c)
        # x = self.hyperboloid.expmap0(x, self.c)
        # x = self.hyperboloid.proj(x, self.c)
        with torch.no_grad():
            pos_p = self.discriminate(z, pos_edge_index)[:, :2].max(dim=1)[1]
            neg_p = self.discriminate(z, neg_edge_index)[:, :2].max(dim=1)[1]
        pred = (1 - torch.cat([pos_p, neg_p])).cpu()
        y = torch.cat(
            [pred.new_ones((pos_p.size(0))),
             pred.new_zeros(neg_p.size(0))])
        pred, y = pred.numpy(), y.numpy()

        auc = roc_auc_score(y, pred)
        f1 = f1_score(y, pred, average='binary') if pred.sum() > 0 else 0

        return auc, f1
