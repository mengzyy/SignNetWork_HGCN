import argparse
from random import randint
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pickle
import datetime
import torch
import numpy as np
import os



def loadData(trainFileCount,trainName):
    pos_edge_index=[]
    neg_edge_index=[]

    path="data/"+trainName+"/"+trainName+"_train"+trainFileCount+".edgelist"
    f = open(path)
    iter_f = iter(f)
    firstLine=True
    for line in iter_f:  # 遍历文件，一行行遍历，读取文本
        if firstLine:
            firstLine=False
            continue
        split=line.split(" ")
        l=int(split[0])
        r=int(split[1])
        attr=int(split[2])
        if attr>0:
            pos_edge_index.append([l,r])
            #无向处理
            pos_edge_index.append([r, l])
        if attr<0:
            neg_edge_index.append([l, r])
            #无向处理
            neg_edge_index.append([r, l])

    # add 返回

    return torch.LongTensor(pos_edge_index).t(),torch.LongTensor(neg_edge_index).t()


def split_edges(edge_index, test_ratio=0.2):
    mask = torch.ones(edge_index.size(1), dtype=torch.bool)
    mask[torch.randperm(mask.size(0))[:int(test_ratio * mask.size(0))]] = 0
    train_edge_index = edge_index[:, mask]
    test_edge_index = edge_index[:, ~mask]
    return train_edge_index, test_edge_index












# x,y=loadData("9", "bitcoinAlpha")

