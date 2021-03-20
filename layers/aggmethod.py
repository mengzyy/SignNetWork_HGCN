import numpy as np
import torch




def computeNegFeaMean(adj, Att, x):
    # # 权重写法
    # adj = adj.detach().numpy()
    # # 权重需要取值
    # att = Att.detach().numpy()
    # x = x.detach().numpy()
    # nodes, feas = x.shape
    # info = dict()
    # temp = np.zeros((nodes, feas))
    # for i in range(0, adj.shape[1]):
    #     r, l = adj[0][i], adj[1][i]
    #     temp[r] += x[l] * att[i]
    #     info[r] = 1 if r not in info else info[r] + 1
    # for key in info.keys():
    #     temp[key] = temp[key] / info[key]
    # for i in range(0, nodes):
    #     if i not in info:
    #         temp[i] += x[i]
    # return torch.FloatTensor(temp)


    # 注意力  adj为连边  Att为 边*分数 x为嵌入向量
    adj = adj.detach().numpy()
    x = x.detach().numpy()
    nodes, feas = x.shape
    scoreByDot = dict()
    vecByDot = dict()
    temp = np.zeros((nodes, feas))
    for i in range(0, adj.shape[1]):
        r, l = adj[0][i], adj[1][i]
        if r not in scoreByDot:
            scoreByDot[r] = []
        # 加入边分数
        scoreByDot[r].append(Att[i])
        if r not in vecByDot:
            vecByDot[r] = []
        # 加入边向量
        vecByDot[r].append(x[l])
    #这边为计算整合
    for r, lis in scoreByDot.items():
        softmaxLis=softmax(np.array(lis)).tolist()[0]
        le=len(softmaxLis)
        for i in range(0,le):
            temp[r] +=vecByDot[r][i]*softmaxLis[i]
    for i in range(0, nodes):
        if i not in scoreByDot:
            temp[i] += x[i]
    return torch.FloatTensor(temp)


def softmax(x):
    # 计算每行的最大值
    row_max = x.max(axis=0)
    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s



