import torch
from model.SignGCN import SignGCN
from model.SignHGCN import SignHGCN
from utils.loadData import loadData
from utils.loadData import split_edges

# name = 'BitcoinOTC-2'
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', name)
# dataset = BitcoinOTC(path, edge_window_size=1)
# pos_edge_indices, neg_edge_indices = [], []
# for data in dataset:
#     pos_edge_indices.append(data.edge_index[:, data.edge_attr > 0])
#     neg_edge_indices.append(data.edge_index[:, data.edge_attr < 0])
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# pos_edge_index = torch.cat(pos_edge_indices, dim=1).to(device)
# neg_edge_index = torch.cat(neg_edge_indices, dim=1).to(device)

# 网络文件索引 最大为9 最小为1
trainFiles = "0"
# 网络训练网络文件名
trainName = "bitcoinOTC"  # bitcoinAlpha bitcoinOTC epinions_truncated slashdot_truncated
# 网络经过tsvd分解的特征大小 默认为64
int_features = 64
# 模型
modelName = "GCN"  # GCN or HGCN
# 网络卷积后的特征大小 默认为64
out_features = 64
# 卷积层大小,即1次邻居卷积 加num_layers-1层间接邻居卷积
num_layers = 2
# loss 参数
lambda_structure = 5
# 学习率
lr = 0.01

pos_edge_index, neg_edge_index = loadData(trainFiles, trainName)
train_pos_edge_index, test_pos_edge_index = split_edges(pos_edge_index)
train_neg_edge_index, test_neg_edge_index = split_edges(neg_edge_index)
# 自训练向量大小
posAtt = train_pos_edge_index.shape[1]
negAtt = train_neg_edge_index.shape[1]



model = None
if modelName == "HGCN":
    model = SignHGCN(int_features, out_features, num_layers=num_layers, lambda_structure=lambda_structure)
if modelName == "GCN":
    model = SignGCN(int_features, out_features, num_layers=num_layers, lambda_structure=lambda_structure, posAtt=posAtt,
                    negAtt=negAtt)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

# 奇异值分解获得 n*64 的 特征表示
x = model.create_spectral_features(train_pos_edge_index, train_neg_edge_index)


def train():
    model.train()
    optimizer.zero_grad()

    z = model(x, train_pos_edge_index, train_neg_edge_index)
    # 算loss
    loss = model.loss(z, train_pos_edge_index, train_neg_edge_index)
    # 反向传播 更新的w
    loss.backward()
    # 写回w
    optimizer.step()

    return loss.item()


def est3():
    model.eval()
    with torch.no_grad():
        z = model(x, train_pos_edge_index, train_neg_edge_index)
    return model.test(z, test_pos_edge_index, test_neg_edge_index)


for epoch in range(2000):
    loss = train()
    auc, f1 = est3()
    print('Epoch: {:03d}, Loss: {:.4f}, AUC: {:.4f}, F1: {:.4f}'.format(
        epoch, loss, auc, f1))
