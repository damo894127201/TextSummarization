# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 16:40
# @Author  : Weiyang
# @File    : pad.py

#---------------------------------------------------
# torch.utils.data.DataLoader的collate_fn回调函数
# 用于对一个batch进行处理：
# 1. 填充到等长
# 2. 返回batch各条数据实际长度，并逆序(target随source排序)
# 3. 对应的是MyDataSet数据集
#---------------------------------------------------

from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

def pad(batch):
    padding_value = 0 # 填充值索引
    source,target = [],[]
    source_lens = []
    target_lens = []
    for x,y in batch:
        source.append(x)
        target.append(y)
        source_lens.append(len(x))
        target_lens.append(len(y))
    # 对source长度倒排，target随其排序
    source_lens = np.array(source_lens)
    descendingOrders = np.argsort(-source_lens)
    source = np.array(source)[descendingOrders].tolist()
    target = np.array(target)[descendingOrders].tolist()
    source_lens = torch.tensor(np.sort(source_lens)[::-1].tolist()) # 相应的长度从大到小排序
    target_lens = np.array(target_lens)[descendingOrders].tolist() # target端的长度随source而排序
    # 转为torch.tensor
    source = [torch.tensor(seq) for seq in source]
    target = [torch.tensor(seq) for seq in target]
    # 填充,batch*max_time_step
    source = pad_sequence(source,batch_first=True,padding_value=padding_value)
    target = pad_sequence(target,batch_first=True,padding_value=padding_value)
    return source,target,source_lens,target_lens