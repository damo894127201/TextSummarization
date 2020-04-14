# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : Seq2Seq.py

#---------------------------------------------------------
# Seq2Seq+Attention
#---------------------------------------------------------

from BowEncoder import BowEncoder
from NNLMDecoder import NNLMDecoder
import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):

    def __init__(self,config):
        super(Seq2Seq,self).__init__()
        self.output_size = config.output_size # 输出空间大小
        # 词向量表用于查询单词词向量,Encoder和Decoder共用同一张表
        self.Embedding_table = nn.Embedding(config.vocab_size, config.embedding_size)
        self.MAX_LENGTH = config.max_decoding_steps # 最大解码长度
        # 初始化encoder和decoder
        self.Encoder = BowEncoder(vocab_size=config.vocab_size,hidden_size=config.hidden_size)
        self.Decoder = NNLMDecoder(output_size=config.output_size,vocab_size=config.vocab_size,embedding_size=config.embedding_size,
                                   hidden_size=config.hidden_size,C_size=config.C_size,alpha1=config.alpha1,
                                   alpha2=config.alpha2,alpha3=config.alpha3,alpha4=config.alpha4,alpha5=config.alpha5)

    def forward(self,source,source_lens,target,teacher_forcing_ratio=0.5):
        '''
        :param source: [batch_size,max_time_step]
        :param source_lens: [batch_size]
        :param target: [batch_size,max_time_step]
        :param teacher_forcing_ratio: 使用TeacherForcing训练的数据比例
        :return:outputs: [target_len,batch_size,vocab_size] # 各个时刻的解码结果，保持概率分布的形式
        '''
        target_len = target.size(1) # target的序列长度
        batch_size = target.size(0) # 当前批次大小
        # Encoder
        context_vectors = self.Encoder(source,source_lens) # [batch_size,hidden_size]

        # Decoder
        # 以输出的各单元概率的形式，存储预测序列，便于交叉熵计算
        outputs = torch.zeros(target_len,batch_size,self.output_size).cuda() # [target_len,batch_size,output_size]
        # 输入解码起始符_GO开始解码,需要给batch中每条数据都输入
        GO_token = torch.tensor([[1]]*batch_size).cuda() # [batch_size,1]
        decoder_outputs = GO_token # 存储解码的单词ID结果
        # 解码长度为 batch中的序列长度
        for step in range(target_len):
            output = self.Decoder(context_vectors,decoder_outputs,source,source_lens) # [batch_size,output_size]
            outputs[step] = output # 记录当前时刻的预测结果
            # 如果为True，则TeacherForcing
            TeacherForcing = random.random() < teacher_forcing_ratio # 随机生成[0,1)之间的数
            topv,topi = output.topk(1) # topk对应的具体值; topi对应相应的索引，即ID,形状是[batch_size,1]
            # target: [batch_size,max_time_step] ，取出当前时刻的一个batch的target
            current_decoder_output = target[:,step] if TeacherForcing else topi # [batch_size]
            current_decoder_output = current_decoder_output.view(-1,1).contiguous() # [batch_size,1]
            decoder_outputs = torch.cat((decoder_outputs,current_decoder_output),dim=1) # 将当前解码结果存储到解码结果的序列中
        return outputs # [max_time,batch_size,vocab_size]

    def BatchSample(self,source,source_lens):
        '''
        批量预测
        :param source: batch输入，[batch,max_time_step]
        :param source_lens: batch输入中各个样例的真实长度,[batch_size]
        :return: 返回预测结果 和 注意力权重分布
        '''
        batch_size = source.size(0)  # 当前批次大小
        # Encoder
        # context_vectors: [batch_size,hidden_size]
        context_vectors = self.Encoder(source, source_lens)

        # Decoder
        # 记录batch中，每条数据的各个时刻的预测结果,解码的最大长度为self.MAX_LENGTH
        results = []
        # 输入解码起始符_GO开始解码
        GO_token = torch.tensor([1]).cuda() # [1]
        # 解码终止符
        EOS_token = torch.tensor(2).cuda()

        # 逐一对batch中的各条数据解码
        for i in range(batch_size):
            current_context_vectors = context_vectors[i].view(1,-1) # [1,hidden_size]
            # 存储当前序列的解码结果
            result = []
            # 解码起始符
            decoder_input = GO_token
            # 由于是单条数据，因此各个维度必须要符合Decoder的输入
            decoder_outputs = decoder_input.unsqueeze(0) # [1,1]
            # 当前输入序列
            current_source = source[i].view(1, -1).contiguous()  # [1,max_time_step]
            # 当前输入序列的长度
            current_source_lens = source_lens[i].view(1).contiguous()  # [1]
            for j in range(self.MAX_LENGTH):
                # output: [1,output_size]
                output = self.Decoder(current_context_vectors,decoder_outputs,current_source,current_source_lens) # [1,output_size]
                topv,topi = output.topk(1) # [1]
                result.append(topi.item())
                decoder_output = topi.view(1,1).contiguous() # [1,1]
                decoder_outputs = torch.cat((decoder_outputs,decoder_output),dim=1) # [1,current_time]
                if topi == EOS_token:
                    break
            results.append(result)
        return results # [batch_size,some_time_step]

if __name__ == '__main__':
    from Config import Config
    config = Config()
    model = Seq2Seq(config)
    print(model)