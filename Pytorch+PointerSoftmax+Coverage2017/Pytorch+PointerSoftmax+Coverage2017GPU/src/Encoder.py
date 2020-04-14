# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:42
# @Author  : Weiyang
# @File    : Encoder.py

#---------------------------------------------------------------
# Encoder: 使用双向单层的LSTM
#---------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,Embedding_table):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size # Encoder的RNN计算单元维度
        self.Embedding_table = Embedding_table # 词向量表用于查询单词词向量
        # input = [max_time_step,batch_size,embedding_size],pack_padded_sequence
        self.BiLSTM = nn.LSTM(input_size=embedding_size,hidden_size=hidden_size,batch_first=False,bidirectional=True)

    def forward(self,input,input_lens):
        '''
        :param input: [batch_size,max_time_step]
        :param input_lens: [batch_size],是input中各条数据未填充前的长度，倒排
        :return: 返回各个时刻RNN的输出和最后一时刻的隐状态
        '''
        # 查表获取词向量序列,维度转为[max_time_step,batch_size,embedding_size]
        embeddings = self.Embedding_table(input).permute(1,0,2)

        # pack掉padding位
        pack_embeddings = pack_padded_sequence(embeddings,input_lens,batch_first=False,enforce_sorted=True)

        # outputs 是各个时刻LSTM最后一层的输出，用于计算注意力权重: [max_time_step,batch_size,hidden_size*num_directions]
        # hidden是最后一个时刻的隐状态,init_hidden是初始时刻隐状态: [num_layers*num_directions,batch_size,hidden_size]
        batch_size = input.size(0)  # 当前批次的实际大小
        init_hidden = self.initHidden(batch_size)
        # cell_state是初始时刻LSTM的细胞状态
        cell_state = self.initHidden(batch_size)
        # hidden: [num_layers*num_directions,batch_size,hidden_size]
        # cell_state: [num_layers*num_directions,batch_size,hidden_size]
        # hidden: [2,batch_size,hidden_size]
        # cell_state: [2,batch_size,hidden_size]
        # outputs: [max_time_step,batch_size,hidden_size*num_directions]
        outputs,(hidden,cell_state) = self.BiLSTM(pack_embeddings,(init_hidden,cell_state))

        # padding:([batch_size,max_time_step,hidden_size*num_directions],input_lens)
        outputs = pad_packed_sequence(outputs,batch_first=True,padding_value=0.0)
        # 将前后两个方向最后时刻的隐状态和细胞状态，拼接在一起，作为解码器的隐状态和细胞状态的初始值,由于是单层LSTM，因此我们让第一个维度为1
        # 表示只是将各层的前后向最后时刻的隐状态分别拼接，层与层之间不拼接；第一个维度为2仅是为了输入解码器LSTM时方便；
        # 细胞状态同理，因此解码器的隐藏层维度要是Encoder的两倍。
        hidden = hidden.view(1,-1,self.hidden_size*2).contiguous() # [1,batch_size,hidden_size*2]
        cell_state = cell_state.view(1,-1,self.hidden_size*2).contiguous() # [1,batch_size,hidden_size*2]

        return outputs,hidden,cell_state  # outputs是个二元组，第一个是output，第二个是序列的长度

    def initHidden(self,batch_size):
        '''初始化初始时刻的隐状态,batch_size必须是实际batch的尺度，有的批次并不满足一个batch'''
        # [num_layers*num_directions,batch_size,hidden_size]
        init_hidden = Variable(torch.zeros(2,batch_size,self.hidden_size)).cuda()
        return init_hidden

if __name__ == '__main__':
    model = Encoder(100,150,None)
    print(model)