# -*- coding: utf-8 -*-
# @Time    : 2020/3/31 0:25
# @Author  : Weiyang
# @File    : recoveryOrder.py

# ------------------------------------------
# 对torch.tensor排序后，再恢复其原有顺序
# -------------------------------------------

import torch

def recoveryOrder(sorted_tensor,descendingOrders):
    '''
    :param sorted_tensor: 排序后的tensor
    :param descendingOrders: 排序索引列表
    :return: 返回原有顺序的tensor
    '''
    batch_size,max_time_step = sorted_tensor.size(0),sorted_tensor.size(1)
    recoveryTensor = torch.zeros((batch_size,max_time_step)) # 恢复顺序后的tensor
    for i,index in enumerate(descendingOrders):
        recoveryTensor[index] = sorted_tensor[i]
    return recoveryTensor # [batch_size,max_time_step]