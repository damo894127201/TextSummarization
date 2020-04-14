# -*- coding: utf-8 -*-
# @Time    : 2020/3/21 10:45
# @Author  : Weiyang
# @File    : BowEncoder.py

# ----------------------------------------------------------------------
# Bag-of-Words Encoder：词袋模型编码器
# Contextual encoder term that returns a vector of size H representing the input and current context

# 不足: while ignoring properties of the original order or relationships between neighboring words
# ----------------------------------------------------------------------


import torch
import torch.nn as nn

class BowEncoder(nn.Module):

    def __init__(self,vocab_size,hidden_size):
        super(BowEncoder,self).__init__()
        self.vocab_size = vocab_size # 词包大小
        self.hidden_size = hidden_size # 隐层维度,同时也是Encoder编码input为一个固定向量的维度
        self.F_embedding_tabel = nn.Embedding(self.vocab_size,self.hidden_size) # F语义向量表,[vocabulary_size,hidden_size],注意与E_embedding_tabel不同
                                                   # 后者是传统的词向量表，用于获取词向量的表示；前者是论文中提出的
                                                   # input_side embedding,用于计算输入序列input的语义表示。本质上，两者是一样的
                                                   # 但E词向量表的维度是embedding_size，而F语义向量表维度是hidden_size

    def forward(self,input,input_lens):
        '''
        返回输入input的且大小为hidden_size的语义表示,[batch_size,hidden_size]
        :param input: [batch_size,max_time_step]
        :param input_lens: [batch_size],是input中各条数据未填充前的长度，倒排
        :return:
        '''
        batch_size,max_time_step = input.size(0),input.size(1)

        # 获取输入序列中每个单词的语义表示
        context_vectors = self.F_embedding_tabel(input) # [batch_size,max_time_step,hidden_size]

        # 获取输入序列input中每个单词的均匀分布，由于有些是填充符pad，因此需要将相应的pad位置为0
        probs = torch.zeros((batch_size,max_time_step)) # [batch_size,max_time_step]
        # 遍历每条数据的实际长度，将相应真实位置的概率置为1/len,pad位置为0
        for i in range(batch_size):
            length = input_lens[i].item() # 当前数据的实际长度
            prob = [1/length] * length + [0] * (max_time_step - length)
            prob = torch.tensor(prob,dtype=torch.float32)
            probs[i] = prob

        # 为probs增加维度，便于三维矩阵相乘 ,三维相乘时，事实上不考虑batch_size，但必须一致
        probs = probs.unsqueeze(1) # [batch_size,1,max_time_step]

        # 计算当期输入的语义表示context
        context = torch.matmul(probs,context_vectors) # [batch_size,1,hidden_size],注意torch.mm只能用于二维矩阵相乘
        context = context.squeeze(1) # [batch_size,hidden_size]

        return context

if __name__ == '__main__':
    encoder = BowEncoder(vocab_size=10,hidden_size=6)

    inputs = torch.tensor([
        [1,2,3,0],
        [5,2,0,0],
        [3,2,1,3]
    ]) # [3,4],batch_size=3,max_time_step=4
    input_lens = torch.tensor([3,2,4]) # [3]

    context = encoder(inputs,input_lens)

    print(context)
    print(context.size())