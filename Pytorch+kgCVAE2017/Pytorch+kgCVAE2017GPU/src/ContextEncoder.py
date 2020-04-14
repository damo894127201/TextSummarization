# -*- coding: utf-8 -*-
# @Time    : 2020/3/31 9:29
# @Author  : Weiyang
# @File    : ContextEncoder.py

# -------------------------------------------------------------------------------------------------------------------
# ContextEncoder
# The context encoder is a 1-layer GRU network that encodes the preceding k-1 utterances by taking u1:k−1
# and the corresponding conversation floor as inputs
# The last hidden state hc of the context encoder is concatenated with meta features and c = [hc; m]

# 简单点，就是一个句子级别的编码器，将每个句子的固定向量表示作为输入，最后输出整篇文章的固定向量表示：最后时刻隐状态的拼接
# 使用单向单层的GRU
# --------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.autograd import Variable

class ContextEncoder(nn.Module):
    def __init__(self,hidden_size):
        super(ContextEncoder,self).__init__()
        self.hidden_size = hidden_size # GRU隐状态维度
        # input = [max_time_step,batch_size,hidden_size*2],pack_padded_sequence
        self.BiGRU = nn.GRU(input_size=hidden_size*2,hidden_size=hidden_size,batch_first=False,bidirectional=False)

    def forward(self,input_embeddings,input_lens,enforce_sorted):
        '''
        :param input_embeddings: [batch_size,max_time_step,hidden_size*2]
        :param input_lens: [batch_size],是input中各条数据未填充前的长度，倒排
        :return: 返回各个时刻RNN的输出和最后一时刻的隐状态
        '''
        # 维度转为[max_time_step,batch_size,hidden_size*2]
        embeddings = input_embeddings.permute(1,0,2)

        # pack掉padding位
        pack_embeddings = pack_padded_sequence(embeddings,input_lens,batch_first=False,enforce_sorted=enforce_sorted)

        # outputs 是各个时刻GRU最后一层的输出，用于计算注意力权重: [max_time_step,batch_size,hidden_size*num_directions]
        # hidden是最后一个时刻的隐状态,init_hidden是初始时刻隐状态: [num_layers*num_directions,batch_size,hidden_size]
        batch_size = input_embeddings.size(0)  # 当前批次的实际大小
        init_hidden = self.initHidden(batch_size)
        outputs,hidden = self.BiGRU(pack_embeddings,init_hidden)

        # padding:([batch_size,max_time_step,hidden_size*num_directions],input_lens)
        outputs = pad_packed_sequence(outputs,batch_first=True,padding_value=0.0)
        # 论文中需将最后时刻的隐状态与meta拼接在一起，作为文章的固定向量表示；此处并没有
        hidden = hidden.view(-1,self.hidden_size).contiguous() # [batch_size,hidden_size]

        return outputs,hidden  # outputs是个二元组，第一个是output，第二个是序列的长度

    def initHidden(self,batch_size):
        '''初始化初始时刻的隐状态,batch_size必须是实际batch的尺度，有的批次并不满足一个batch'''
        # [num_layers*num_directions,batch_size,hidden_size]
        init_hidden = Variable(torch.zeros(1,batch_size,self.hidden_size)).cuda()
        return init_hidden

if __name__ == '__main__':
    model = ContextEncoder(100)
    print(model)

