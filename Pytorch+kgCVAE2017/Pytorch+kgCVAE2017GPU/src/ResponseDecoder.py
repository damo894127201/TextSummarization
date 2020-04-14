# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : ResponseDecoder.py

#-----------------------------------------------------------------------------------------------------------------------
# ResponseDecoder: 使用单向GRU
# 输入空间和输出空间一致，共用同一个词包
#-----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResponseDecoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,output_size,Embedding_table,keep_prob):
        super(ResponseDecoder,self).__init__()
        self.hidden_size = hidden_size # Encoder的RNN单元的维度,Decoder单元的维度要适应Encoder的维度
        self.output_size = output_size # 解码器输出词包大小
        self.Embedding_table = Embedding_table # 词向量表用于查询单词词向量

        self.GRU = nn.GRU(input_size=embedding_size,hidden_size=self.hidden_size,batch_first=False) # Decoder解码单元
        self.out = nn.Linear(self.hidden_size+embedding_size,self.output_size) # 线性层输出层
        self.drop_out = nn.Dropout(keep_prob) # 在输入层后设置Dropout层

    def forward(self,input,pre_hidden,flag=True):
        '''
        :param input: [batch_size,1] 批量的单个字符的ID，即上一解码时刻的输出
        :param pre_hidden: [batch_size,1,hidden_size],上一时刻的隐状态，初始时为Encoder最后一时刻的隐状态
        :return: 返回当前时刻的预测的结果，即每个可能值的概率；当前隐藏层状态；当前时刻的注意力权重
        '''

        # Decoder端输入input: [batch_size,1,embedding_size]
        embeddings = self.Embedding_table(input)
        if flag == True:
            embeddings = self.drop_out(embeddings) # dropout层
        new_input = embeddings.permute(1,0,2) # [1,batch_size,embedding_size]
        pre_hidden = pre_hidden.permute(1,0,2) # [1,batch_size,hidden_size]

        # ------------------------------------------ 计算当前时刻的GRU输出的隐状态--------------------------------------
        # 计算当前解码时刻的隐状态current_hidden
        # new_input: [1,batch_size,embedding_size]
        # pre_hidden:[1,batch_size,hidden_size]
        # output: [1,batch_size,hidden_size]
        # current_hidden: [1,batch_size,hidden_size]
        output,current_hidden = self.GRU(new_input,pre_hidden)
        current_hidden = current_hidden.permute(1,0,2) # [batch_size,1,hidden_size]

        # ------------------------------------------生成模块的概率的计算------------------------------------------------
        # 当前时刻的输入embeddings: [batch_size,1,embedding_size]
        # 当前时刻的隐状态current_hidden: [batch_size,1,hidden_size]
        # 将上述两者拼接一起，然后通过一个线性变换层+softmax层生成各个单词的概率
        output = torch.cat((embeddings, current_hidden),2)  # [batch_size,1,embedding_size+hidden_size]
        output = output.squeeze(1)  # [batch_size,embedding_size+hidden_size]
        # 解码当前时刻的输出，生成预测
        output = self.out(output) # [batch_size,output_size]
        # 转为概率
        output = F.softmax(output,dim=1)

        return output,current_hidden

if __name__ == '__main__':
    model = ResponseDecoder(150,300,1000,None,0.5)
    print(model)