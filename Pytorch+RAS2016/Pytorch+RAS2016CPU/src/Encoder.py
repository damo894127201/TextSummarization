# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:42
# @Author  : Weiyang
# @File    : Encoder.py

# -------------------------------------------------------------------------------------------------------------------
# Encoder: 使用卷积神经网络CNN
# 1. 增加了位置编码，需要单词的位置编码向量+Embedding，作为新的Embedding输入模型,参见论文 3.2 Attentive Encoder
#    这里的位置编码，我们采用Transformer的实现，即用正弦函数编码偶数位，余弦函数编码奇数位
# 2. 输入序列中各个位置对应的隐状态，通过卷积而得，即在某个单词卷积一次后，其结果作为当前位置的隐状态
# 3. 由于Decoder是LSTM，它需要一个初始细胞状态和隐状态，论文中并没有细说这两个值该如何构造，因此我们采用如下方式获得：
#    将最后那个单词的卷积结果，通过线性变换作为Decoder初始时刻的隐状态；将所有单词的卷积结果，通过线性变换作为Decoder
#    初始时刻的细胞状态

# 从论文3.2 Attentive Encoder部分的公式(2)可看出，卷积网络只有一个卷积层，即局部感受野(卷积核，过滤器)，没有池化层和加激活函数的操作；
# 并且，卷积核的长为单词embedding的大小，宽为5个单词的宽度，stride为1，前后填充哑变量;
# 只对输入层做卷积操作即可,由于只是一个方向的，因此是一维卷积。一维卷积常用于自然语言处理领域中。
# 注意论文中的卷积操作与一般的卷积操作有些不同，无法直接调用torch.nn.Conv1d，需要手动实现

# Conv1d: 自然语言处理领域
# Conv2d: 计算机视觉、图像领域
# Conv3d: 医学领域(CT影响)、视频处理领域(检查动作及人物行为)
# -------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable
from positionalEncoding import postionalEncoding
from Conv1d import Conv1d

class Encoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,Embedding_table,B_matrix,q_size):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size # Decoder的RNN计算单元维度
        self.embedding_size = embedding_size # 词向量维度
        self.Embedding_table = Embedding_table # 词向量表用于查询单词词向量
        self.B_matrix = B_matrix  # a learnable weight matrix which is used to convolve over the full embeddings of consecutive words
        self.q_size = q_size # 卷积核的高度height,其宽度为embedding_size
        self.transform = nn.Linear(self.embedding_size,self.hidden_size) # 用于计算Decoder初始时刻的细胞状态和隐状态


    def forward(self,input,input_lens):
        '''
        :param input: [batch_size,max_time_step]
        :param input_lens: [batch_size],是input中各条数据未填充前的长度，倒排
        :return: 返回各个时刻RNN的输出和最后一时刻的隐状态
        '''
        # 查表获取词向量序列,维度转为[max_time_step,batch_size,embedding_size]
        embeddings = self.Embedding_table(input).permute(1,0,2)
        # 获取位置编码
        positionalEmbeddings = postionalEncoding(input,input_lens,self.embedding_size) # [batch_size,max_time_step,embedding_size]
        positionalEmbeddings = positionalEmbeddings.permute(1,0,2) # [max_time_step,batch_size,embedding_size]

        embeddings = embeddings + positionalEmbeddings # 加上位置编码信息

        conv_outputs = Conv1d(embeddings,input_lens,self.B_matrix,self.q_size) # [batch_size,max_time_step,embedding_size]

        # 由于decoder采用LSTM，因此它需要一个初始cell_state和hidden_state,
        # 由于论文中对此语焉不详，因此这里我们让cell_state和hidden_state初始时一致
        # 我们先用一个线性变换层，再通过relu函数来计算得到cell_state和hidden_state
        # 先累加各个时刻的卷积结果
        combine_input = torch.sum(conv_outputs,dim=1) # [batch_size,embedding_size]
        combine_input = combine_input.view(-1,1,self.embedding_size) # [batch_size,1,embedding_size]
        cell_state = self.transform(combine_input) # [batch_size,1,hidden_size]
        cell_state = torch.relu(cell_state) # [batch_size,1,hidden_size]

        # hidden: [batch_size,1,hidden_size]
        # outputs: ([batch_size,max_time_step,embedding_size],input_lens)
        hidden = cell_state.clone()  # [batch_size,1,hidden_size]
        cell_state = cell_state.permute(1,0,2) # cell_state: [1,batch_size,hidden_size]
        outputs = (conv_outputs,input_lens)

        return outputs,hidden,cell_state  # outputs是个二元组，第一个是output，第二个是序列的长度


if __name__ == '__main__':
    model = Encoder(100,150,None)
    print(model)