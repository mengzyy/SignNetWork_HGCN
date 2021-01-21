import os.path as osp
from torch_geometric.nn import SignedGCN
import torch
from torch_geometric.datasets import BitcoinOTC
from model.SignGCN import SignGCN
from model.SignHGCN import SignHGCN
# data pre
name = 'BitcoinOTC-2'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
dataset = BitcoinOTC(path, edge_window_size=1)
pos_edge_indices, neg_edge_indices = [], []
for data in dataset:
    pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
    neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

# Build and train model.

model = SignGCN(64, 64, num_layers=2, lambda_structure=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

train_pos_edge_index, test_pos_edge_index = model.split_edges(pos_edge_index)
train_neg_edge_index, test_neg_edge_index = model.split_edges(neg_edge_index)
# 奇异值分解获得 n*64 的 特征表示
x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)

def train():
    model.train()
    optimizer.zero_grad()
    z = model(x, train_pos_edge_index, train_neg_edge_index)
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    loss.backward()
    optimizer.step()
    return loss.item()

def est3():
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)

for epoch in range(1010):
    loss = train()
    auc, f1 = est3()
    print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, F1: {:.4f}'.format(
        epoch, loss, auc, f1))
