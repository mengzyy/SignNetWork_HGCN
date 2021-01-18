import argparse
from random import *
from model import SignGCN, SignHGCN
import numpy as np
import torch
import random
from utils import loadData

"""
启动参数设置
"""
# ================================================================================= #
parser = argparse.ArgumentParser(description="""Params to build Sign_HGCN """)
parser.add_argument('--structure_network_file_name', type=str, required=False,
                    default="data\\train\\bitcoinAlpha\\bitcoinAlpha_train0.edgelist")
parser.add_argument('--feature_network_file_name', type=str, required=False,
                    default="data\\features\\bitcoinAlpha\\bitcoinAlpha_train0_features64_tsvd.pkl")
parser.add_argument('--test_structure_network_file_name', type=str, required=False,
                    default="data\\test\\bitcoinAlpha\\bitcoinAlpha_test0.edgelist")
parser.add_argument('--features', type=int, required=False, default=64)
parser.add_argument('--lambda_structure', type=float, default=4.5)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--model_regularize', type=float, default=0.01)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--dropout', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1000)
parser.add_argument('--train_count', type=int, default=1000)
parser.add_argument('--train_size', type=int, default=500)
parser.add_argument('--test_interval', type=int, default=1)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=randint(0, 2147483648))
parser.add_argument('--method', type=str, default="HGCN")
parser.add_argument('--local_agg', type=int, default=0)
parser.add_argument('--act', type=str, default="relu")
parser.add_argument('--use_bias', type=bool, default=True)
parser.add_argument('--use_att', type=bool, default=False)
parser.add_argument('--c', type=int, default=1)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--class_weight_no', type=float, default=0.35)
parameters = parser.parse_args()
args = {}
for arg in vars(parameters):
    args[arg] = getattr(parameters, arg)
args['class_weights'] = loadData.calculate_class_weights(3775, 10176, 1119, w_no=args['class_weight_no'])

# ================================================================================= #
# 载入
data = loadData.loadSignData(args["structure_network_file_name"], args["feature_network_file_name"],
                             args["test_structure_network_file_name"], args["features"])
# 数据预处理----->根据符号网络特点进行特征处理,可后期扩展
data = loadData.preProcessData(data, args["layers"])
args["data"] = data
args["nodes"] = data["num_nodes"]
# 初始化模型
if args["method"] == "GCN":
    model = SignGCN.SignGCN(args)
else:
    model = SignHGCN.SignHGCN(args)

optimizer = torch.optim.Adagrad(model.parameters(),
                                lr=args['learning_rate'], weight_decay=args['model_regularize'])
train = list(np.random.permutation(list(range(0, args["nodes"]))))
for batch in range(args["train_count"]):
    random.shuffle(train)
    batch_center_nodes = train[0:args["train_size"]]
    model.train()
    # 梯度清0
    optimizer.zero_grad()
    embeddings = model.encode(None, None, None, None)
    loss = model.loss(batch_center_nodes, data["adj_lists_pos"], data["adj_lists_neg"], embeddings)
    loss.backward()
    optimizer.step()
    print('batch {} loss: {}'.format(batch, loss))
    if (batch + 1) % args["test_interval"] == 0 or batch == args["train_count"] - 1:
        model.eval()
        auc, f1 = model.test_func(data["adj_lists_pos"], data["adj_lists_neg"], data["test_adj_lists_pos"],
                                  data["test_adj_lists_neg"], embeddings)
        print(batch, ' test_func sign prediction (auc,f1) :', auc, '\t', f1)

optimizer.zero_grad()
embeddings = model.encode(None, None, None, None)
auc, f1 = model.test_func(data["adj_lists_pos"], data["adj_lists_neg"], data["test_adj_lists_pos"],
                          data["test_adj_lists_neg"], embeddings)
print("final info-->", "prediction (auc,f1) :", auc, '\t', f1)
