# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 10:50
# @Author  : Weiyang
# @File    : test.py

import torch.nn as nn
import torch

gru = nn.GRU(input_size=50,hidden_size=100,batch_first=False,bidirectional=True)
embed = nn.Embedding(3,50)
input_x = torch.LongTensor([[0,1,2],[2,1,0],[1,2,0],[1,1,1]]) # [4,3]
x_embed = embed(input_x)
print(x_embed.size()) # [batch_size,max_time_step,embedding_size]
print('---------------------------------')
x_embed = x_embed.permute(1,0,2)
print(x_embed.size()) # [max_time_step,batch_size,embedding_size]
print('---------------------------------')
out,hidden = gru(x_embed)
print(out.size())
print(hidden.size())
print(out.size(1))

'''
torch.Size([4, 3, 50])
---------------------------------
torch.Size([3, 4, 50])
---------------------------------
torch.Size([3, 4, 200])
torch.Size([2, 4, 100])
'''
