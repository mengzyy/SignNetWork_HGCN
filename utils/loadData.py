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
        if attr<0:
            neg_edge_index.append([l, r])

    return torch.LongTensor(pos_edge_index).t(),torch.LongTensor(neg_edge_index).t()












# x,y=loadData("9", "bitcoinAlpha")

