# -*- coding: utf-8 -*-
# @Time    : 2020/3/21 17:34
# @Author  : Weiyang
# @File    : NNLMDecoder.py

# ----------------------------------------------------------------------
# Decoder解码器: 局部注意力机制
# a standard feed-forward neural network language model (NNLM)
# ----------------------------------------------------------------------

import torch
import torch.nn as nn

class NNLMDecoder(nn.Module):

    def __init__(self,output_size,vocab_size,embedding_size,hidden_size,C_size):
        super(NNLMDecoder,self).__init__()
        self.vocab_size = vocab_size # 输入空间词包大小
        self.output_size = output_size # 输出空间的大小，这里是词包大小
        self.embedding_size = embedding_size # 词向量维度
        self.hidden_size = hidden_size # Encoder编码输入input为语义向量的维度
        self.C_size = C_size # 在解码时，利用已知解码序列的单词个数，比如在解码当前时刻时，要基于先前的C_size个单词解码，即马尔可夫假设的范围
                             # 这是局部注意力机制的体现
        self.E_embedding_table = nn.Embedding(self.vocab_size,self.embedding_size) # E词向量表,[vocab_size,embedding_size]
        self.linear1 = nn.Linear(in_features=self.C_size*self.embedding_size,out_features=self.hidden_size) # 计算当前隐状态
        self.linear2 = nn.Linear(in_features=self.hidden_size,out_features=self.output_size) # 输出层
        self.linear3 = nn.Linear(in_features=self.hidden_size,out_features=self.output_size)



    def forward(self,encoder_output,predict_sequence):
        '''
        返回当前时刻的预测结果,[batch_size,output_size],其中最后一维度是各个词对应的概率
        :param encoder_output: [batch_size,hidden_size]
        :param predict_sequence: 先前时刻已预测的单词序列,[batch_size,None]
        :return:
        '''
        batch_size = encoder_output.size(0)
        # 获取当前解码时刻前面的C_size个已解码的单词序列
        C_size_predict_sequence = predict_sequence[:, -self.C_size:]  # [batch_size,C_size],当然，若已知解码的长度小于C_size
        # 则最后一维为实际已解码的长度
        # 获取当前解码时刻前面的C_size个已解码的单词序列的词向量
        C_size_contexts = self.E_embedding_table(C_size_predict_sequence)  # [batch_size,C_size,embedding_size]

        # 由于当前解码时刻前面不一定有C_size个已解码的单词，因此需要为其扩充到C_size个,扩充的部分全部置为0向量
        actual_len = C_size_contexts.size(1)  # 实际长度
        if actual_len < self.C_size:
            padding_contexts = torch.zeros((batch_size, self.C_size - actual_len, self.embedding_size))
            C_size_contexts = torch.cat((C_size_contexts, padding_contexts), 1)

        # 变换C_size_contexts的维度
        C_size_contexts = C_size_contexts.view(batch_size,-1)  # [batch_size,C_size*embedding_size]

        h = self.linear1(C_size_contexts) # [batch_size,hidden_size]
        h = torch.tanh(h) # [batch_size,hidden_size]
        Vh = self.linear2(h) # [batch_size,output_size]

        # encoder_output: [batch_size,hidden_size]
        Wenc = self.linear3(encoder_output) # [batch_size,output_size]

        output = Vh + Wenc # [batch_size,output_size]

        output = torch.softmax(output,dim=1) # [batch_size,output_size]

        return output

if __name__ == '__main__':
    from BowEncoder import BowEncoder

    encoder = BowEncoder(vocab_size=10, hidden_size=6)

    inputs = torch.tensor([
        [1, 2, 3, 0],
        [5, 2, 0, 0],
        [3, 2, 1, 3]
    ])  # [3,4],batch_size=3,max_time_step=4
    input_lens = torch.tensor([3, 2, 4])  # [3]

    context = encoder(inputs, input_lens)

    y = torch.tensor([
        [7, 5, 2],
        [6, 2, 1],
        [4, 3, 2]
    ])  # batch_size=3,

    decoder = NNLMDecoder(output_size=10, vocab_size=10, embedding_size=12, hidden_size=6, C_size=2)

    output = decoder(context, y)

    print(output.size())
    print(output)
    print(output)