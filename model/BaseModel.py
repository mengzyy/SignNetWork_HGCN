import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
from random import *

"""
父类模型构造必须实现的方法即可
"""


class BaseModel(nn.Module):
    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.nodes = args["nodes"]
        self.features = args["features"]
        # 定义交叉熵
        self.CrossEntLoss = nn.CrossEntropyLoss()
        # loss出口
        self.addParam = 0 if args["method"] == "GCN" else 2
        self.param_src = nn.Parameter(torch.FloatTensor((self.features + self.addParam) * 2, 3))
        self.structural_distance = nn.PairwiseDistance(p=2)
        self.lambda_structure = args["lambda_structure"]

    def encode(self, feature_data_pos, feature_data_neg, adj_pos_matrix, adj_neg_matrix):
        raise NotImplementedError

    # 不是计算全部节点loss，仅仅计算center_nodes部分的loss
    def loss(self, center_nodes, adj_lists_pos, adj_lists_neg, final_embedding):
        max_node_index = self.nodes - 1
        i_loss2 = []
        pos_no_loss2 = []
        no_neg_loss2 = []
        i_indices = []
        j_indices = []
        ys = []
        all_nodes_set = set()
        skipped_nodes = []
        for i in center_nodes:
            # 没有节点直接忽视
            if (len(adj_lists_pos[i]) + len(adj_lists_neg[i])) == 0:
                skipped_nodes.append(i)
                continue
            # add
            all_nodes_set.add(i)
            for j_pos in adj_lists_pos[i]:
                i_loss2.append(i)
                pos_no_loss2.append(j_pos)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                no_neg_loss2.append(temp)
                all_nodes_set.add(temp)

                i_indices.append(i)
                j_indices.append(j_pos)
                ys.append(0)
                all_nodes_set.add(j_pos)
            for j_neg in adj_lists_neg[i]:
                i_loss2.append(i)
                no_neg_loss2.append(j_neg)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in adj_lists_pos[i]) and (temp not in adj_lists_neg[i]):
                        break
                pos_no_loss2.append(temp)
                all_nodes_set.add(temp)
                i_indices.append(i)
                j_indices.append(j_neg)
                ys.append(1)
                all_nodes_set.add(j_neg)
            need_samples = 2  # number of sampling of the no links pairs
            cur_samples = 0
            while cur_samples < need_samples:
                temp_samp = randint(0, max_node_index)
                if (temp_samp not in adj_lists_pos[i]) and (temp_samp not in adj_lists_neg[i]):
                    # got one we can use
                    i_indices.append(i)
                    j_indices.append(temp_samp)
                    ys.append(2)
                    all_nodes_set.add(temp_samp)
                cur_samples += 1

        all_nodes_list = list(all_nodes_set)
        all_nodes_map = {node: i for i, node in enumerate(all_nodes_list)}
        # final_embedding = self.forward(all_nodes_list)
        i_indices_mapped = [all_nodes_map[i] for i in i_indices]
        j_indices_mapped = [all_nodes_map[j] for j in j_indices]
        ys = torch.LongTensor(ys)
        # now that we have the mapped indices and final embeddings we can get the loss
        loss_entropy = self.CrossEntLoss(
            torch.mm(torch.cat((final_embedding[i_indices_mapped],
                                final_embedding[j_indices_mapped]), 1),
                     self.param_src),
            ys)
        i_loss2 = [all_nodes_map[i] for i in i_loss2]
        pos_no_loss2 = [all_nodes_map[i] for i in pos_no_loss2]
        no_neg_loss2 = [all_nodes_map[i] for i in no_neg_loss2]

        tensor_zeros = torch.zeros(len(i_loss2))

        loss_structure = torch.mean(
            torch.max(
                tensor_zeros,
                self.structural_distance(final_embedding[i_loss2], final_embedding[pos_no_loss2]) ** 2
                - self.structural_distance(final_embedding[i_loss2], final_embedding[no_neg_loss2]) ** 2
            )
        )
        return loss_entropy + self.lambda_structure * loss_structure

    # 指标测试
    def test_func(self, adj_lists_pos, adj_lists_neg, test_adj_lists_pos, test_adj_lists_neg, final_embedding):
        # no map necessary for ids as we are using all nodes
        final_embedding = final_embedding.detach().numpy()
        # training dataset
        X_train = []
        y_train = []
        X_val = []
        y_test_true = []
        for i in range(self.nodes):
            for j in adj_lists_pos[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(1)

            for j in adj_lists_neg[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(-1)

            for j in test_adj_lists_pos[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(1)

            for j in test_adj_lists_neg[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(-1)

        y_train = np.asarray(y_train)
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        y_test_true = np.asarray(y_test_true)
        model = RandomForestClassifier(n_estimators=10, random_state=11, class_weight='balanced')
        X_train_isf = np.isfinite(X_train)
        if X_train_isf.all() == False:
            return 0, 0
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_val)
        auc = roc_auc_score(y_test_true, y_test_pred)
        f1 = f1_score(y_test_true, y_test_pred)
        return auc, f1
