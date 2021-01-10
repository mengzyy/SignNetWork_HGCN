import argparse
from random import randint
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pickle
import datetime
import torch


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
def preProcessData(data):
    _, features = data["feat_data"].shape
    split = int(features / 2)
    data["adj_pos_matrix"] = torch.tensor(data["adj_pos_matrix"], dtype=torch.float32)
    data["adj_neg_matrix"] = torch.tensor(data["adj_neg_matrix"], dtype=torch.float32)
    data["feature_data_pos"] = torch.tensor(data["feat_data"][:, 0:split], dtype=torch.float32)
    data["feature_data_neg"] = torch.tensor(data["feat_data"][:, split-1:-1], dtype=torch.float32)
    return data
