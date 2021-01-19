import argparse
from random import randint
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pickle
import datetime
import torch
import numpy as np


def read_in_undirected_network(file_name):
    links = {}
    with open(file_name) as fp:
        n, m = [int(val) for val in fp.readline().split()[-2:]]
        for l in fp:
            rater, rated, sign = [int(val) for val in l.split()]
            assert (sign != 0)
            sign = 1 if sign > 0 else -1

            edge1, edge2 = (rater, rated), (rated, rater)
            if edge1 not in links:
                links[edge1], links[edge2] = sign, sign
            elif links[edge1] == sign:  # we had it before and it was the same
                pass
            else:  # we had it before and now it's a different value
                links[edge1], links[edge2] = -1, -1  # set to negative

    adj_lists_pos, adj_lists_neg = defaultdict(set), defaultdict(set)
    num_edges_pos, num_edges_neg = 0, 0
    for (i, j), s in links.items():
        if s > 0:
            adj_lists_pos[i].add(j)
            num_edges_pos += 1
        else:
            adj_lists_neg[i].add(j)
            num_edges_neg += 1
    num_edges_pos /= 2
    num_edges_neg /= 2

    return n, [num_edges_pos, num_edges_neg], adj_lists_pos, adj_lists_neg


def read_in_feature_data(feature_file_name, num_input_features):
    feat_data = pickle.load(open(feature_file_name, "rb"))
    if num_input_features is not None:
        # we perform a shrinking as to which features we are using
        feat_data = feat_data[:, :num_input_features]
    num_nodes, num_feats = feat_data.shape
    # standardizing the input features
    feat_data = StandardScaler().fit_transform(feat_data)  # .T).T
    return num_feats, feat_data


def loadSignData(network_file_name, feature_file_name, test_network_file_name, num_input_features):
    num_nodes, num_edges, adj_lists_pos, adj_lists_neg = read_in_undirected_network(network_file_name)
    num_feats, feat_data = read_in_feature_data(feature_file_name, num_input_features)

    test_num_nodes, test_num_edges, test_adj_lists_pos, test_adj_lists_neg = \
        read_in_undirected_network(test_network_file_name)

    # 加入正负邻接矩阵
    adj_pos_matrix = [[0] * num_nodes for i in range(num_nodes)]
    adj_neg_matrix = [[0] * num_nodes for i in range(num_nodes)]

    # 正邻接矩阵
    for key in adj_lists_pos.keys():
        for d in adj_lists_pos[key]:
            adj_pos_matrix[key - 1][d - 1] = 1

    # 负邻接矩阵
    for key in adj_lists_neg.keys():
        for d in adj_lists_neg[key]:
            adj_neg_matrix[key - 1][d - 1] = 1

    res = {}
    res["num_nodes"] = num_nodes
    res["num_edges"] = num_edges
    res["feat_data"] = feat_data

    res["adj_lists_pos"] = adj_lists_pos
    res["adj_lists_neg"] = adj_lists_neg
    res["test_adj_lists_pos"] = test_adj_lists_pos
    res["test_adj_lists_neg"] = test_adj_lists_neg

    res["adj_pos_matrix"] = adj_pos_matrix
    res["adj_neg_matrix"] = adj_neg_matrix

    return res


# 这里的data是loadSignData之后的数据，需要进一步的处理，比如正负特征处理
def preProcessData(data, layers):
    _, features = data["feat_data"].shape
    data["adj_pos_matrix"] = torch.tensor(data["adj_pos_matrix"], dtype=torch.float32)
    data["adj_neg_matrix"] = torch.tensor(data["adj_neg_matrix"], dtype=torch.float32)
    data["node_pos_neg_set"] = {}
    for node in range(0, _):
        node += 1
        # update： layer不影响邻接矩阵
        data["node_pos_neg_set"][node] = getNodePosAndNegByLayers(node, data["adj_lists_pos"],
                                                                  data["adj_lists_neg"])
    return data


# 获取某个节点正邻居，负邻居
def node2PosAndNeg(nodeIndex, adj_pos_set, adj_neg_set):
    node_pos_set = adj_pos_set[nodeIndex]
    node_neg_set = adj_neg_set[nodeIndex]
    return node_pos_set, node_neg_set


# 通过某个节点，获取对应的朋友，敌人向量表示
def getNodePosAndNegEmbeging(nodeIndex, featureData, adj_pos_set, adj_neg_set):
    node_pos_set, node_neg_set = node2PosAndNeg(nodeIndex, adj_pos_set, adj_neg_set)
    # 获取节点特征
    l, r = featureData.shape
    res = [[0] * r for i in range(2)]

    if (len(node_pos_set) != 0):
        temp = [0] * r
        for key in node_pos_set:
            temp += featureData[key - 1]
        res[0] = [i / len(node_pos_set) for i in temp]

    if (len(node_neg_set) != 0):
        temp = [0] * r
        for key in node_neg_set:
            temp += featureData[key - 1]
        res[1] = [i / len(node_neg_set) for i in temp]

    return res[0], res[1], len(node_pos_set), len(node_neg_set)


# 根据前一层的特征矩阵------》获得某个节点的feature表示,【需要注意的是，2层及其之后，特征矩阵为 3*d,因为包含了：朋友下层表示，敌人下层表示，当前自己表示】
# L为当前层
def getNodePosAndNegEmbegingByDiffLayer(data, L, nodeIndex, featureData, adj_pos_set, adj_neg_set):
    res = []
    if L == 1:
        l, r = featureData.shape
        # 首先定义第一层，因为向量表达不同
        pos_emb, neg_emb, c1, c2 = getNodePosAndNegEmbeging(nodeIndex, featureData, adj_pos_set, adj_neg_set)
        res = [pos_emb, neg_emb]
    elif L > 1:
        pos_pos_emb, pos_neg_emb, c1, c2 = getNodePosAndNegEmbeging(nodeIndex, featureData[0], adj_pos_set, adj_neg_set)
        neg_pos_emb, neg_neg_emb, c1_, c2_ = getNodePosAndNegEmbeging(nodeIndex, featureData[1], adj_pos_set,
                                                                      adj_neg_set)
        pos_pos_emb.extend(neg_neg_emb)
        pos_neg_emb.extend(neg_pos_emb)
        res = [pos_pos_emb, pos_neg_emb]
    return res
# 返回节点---字典形式list
def getNodePosAndNegByLayers(node, adj_pos_set, adj_neg_set):
    B, H = node2PosAndNeg(node, adj_pos_set, adj_neg_set)
    res = [B, H]
    return res

def calculate_class_weights(num_V, num_pos, num_neg, w_no=None):
    num_E = num_pos + num_neg
    num_V = num_V * 2  # sampling 2 non-connected nodes for each node in network.

    num_total = num_E + num_V

    if w_no is None:
        w_no = round(num_V * 1.0 / num_total, 2)
    else:
        assert isinstance(w_no, float) and 0 < w_no < 1
    w_pos_neg = 1 - w_no

    w_pos = round(w_pos_neg * num_neg / num_E, 2)
    w_neg = round(w_pos_neg - w_pos, 2)

    return w_pos, w_neg, w_no
