import torch
import numpy as np

# print(torch.cuda.is_available())


adj = torch.tensor([[1, 2, 3, 1], [6, 7, 8, 2]])
x = torch.tensor([[0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1],
                  [2, 2, 2, 2, 2],
                  [3, 3, 3, 3, 3],
                  [4, 4, 4, 4, 4],
                  [5, 5, 5, 5, 5],
                  [6, 6, 6, 6, 6],
                  [7, 7, 7, 7, 7],
                  [8, 8, 8, 8, 8],
                  [9, 9, 9, 9, 9]])

adj = adj.numpy()
x = x.numpy()
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



print("sd")
