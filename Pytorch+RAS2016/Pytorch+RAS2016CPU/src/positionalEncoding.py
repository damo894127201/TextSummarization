# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 11:43
# @Author  : Weiyang
# @File    : positionalEncoding.py

# ----------------------------------------------
# Positional Encoding(位置编码)
# 这里的实现采用Transformer的位置编码
# ----------------------------------------------

import torch
import numpy as np

def postionalEncoding(inputs,input_lens,embedding_size):
    '''
    返回输入序列inputs的位置编码向量,padding位置各维度的值全部置为0
    :param inputs: tensor,[batch_size,max_time_step]
    :param input_lens: [batch_size],是input中各条数据未填充前的长度，倒排
    :param embedding_size: embedding_size
    :return: positionalEmbedding,[batch_size,max_time_step,embedding_size]
    '''
    batch_size,max_time_step = inputs.size(0),inputs.size(1)
    position_enc = np.array([
        [pos/np.power(10000,(i-i%2)/embedding_size) for i in range(embedding_size)]
        for pos in range(max_time_step)]) # positionEmbedding各个维度的取值，即正弦和余弦函数括号内的取值,[max_time_step,embedding_size]

    # 序列偶数位取值
    position_enc[:,0::2] = np.sin(position_enc[:,0::2]) # 2i
    # 序列奇数位取值
    position_enc[:,1::2] = np.cos(position_enc[:,1::2]) # 2i+1

    # 存储batch数据的位置编码信息
    positionEmbeddings = np.zeros((batch_size,max_time_step,embedding_size))

    # 获取当前输入inputs每条数据每个位置的位置编码
    # 遍历每条数据
    for i in range(batch_size):
        # 遍历每条数据的实际长度
        for j in range(input_lens[i]):
            positionEmbeddings[i][j] += position_enc[j]

    # 转为torch.tensor
    positionEmbeddings = torch.tensor(positionEmbeddings,dtype=torch.float32,requires_grad=True) # [batch_size,max_time_step,embedding_size]

    return positionEmbeddings # [batch_size,max_time_step,embedding_size]

if __name__ == '__main__':
    inputs = torch.tensor([[1,5,3,0,0],[2,3,6,4,5]]) # [2,5]
    input_lens = torch.tensor([3,5])
    embedding_size = 50
    pe = postionalEncoding(inputs,input_lens,embedding_size)
    print(pe.size())
    print(pe[0])