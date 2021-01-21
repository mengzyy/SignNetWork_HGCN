import numpy as np
import torch


def computeNegFeaMean(adj, x):
    adj = adj.cuda().data.cpu().numpy()
    x = x.cuda().data.cpu().numpy()
    nodes, feas = x.shape
    info = dict()
    temp = np.zeros((nodes, feas))
    for i in range(0, adj.shape[1]):
        r, l = adj[0][i], adj[1][i]
        temp[r] += x[l]
        temp[l] += x[r]
        info[r] = 1 if r not in info else info[r] + 1
        info[l] = 1 if l not in info else info[l] + 1
    for key in info.keys():
        temp[key] = temp[key] / info[key]
    for i in range(0, nodes):
        if i not in info:
            temp[i] += x[i]
    return torch.FloatTensor(temp).cuda()
