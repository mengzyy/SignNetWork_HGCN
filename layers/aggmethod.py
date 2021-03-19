import numpy as np
import torch


def computeNegFeaMean(adj, Att, x):
    adj = adj.detach().numpy()
    att = Att.detach().numpy()
    x = x.detach().numpy()
    nodes, feas = x.shape
    info = dict()
    temp = np.zeros((nodes, feas))
    for i in range(0, adj.shape[1]):
        r, l = adj[0][i], adj[1][i]
        temp[r] += x[l]*att[i]
        info[r] = 1 if r not in info else info[r] + 1
    for key in info.keys():
        temp[key] = temp[key] / info[key]
    for i in range(0, nodes):
        if i not in info:
            temp[i] += x[i]
    return torch.FloatTensor(temp)
