# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : PointerGeneratorCoverage.py

#---------------------------------------------------------
# PointerGenerator(PointerSoftmax) + Coverage.
#---------------------------------------------------------

from Encoder import Encoder
from AttentionPointerCoverageDecoder import AttentionPointerCoverageDecoder
import torch
import torch.nn as nn
from torch.autograd import Variable
import random

class PointerGeneratorCoverage(nn.Module):

    def __init__(self,config):
        super(PointerGeneratorCoverage,self).__init__()
        self.output_size = config.output_size # 输出空间大小
        # 词向量表用于查询单词词向量,Encoder和Decoder共用同一张表
        self.Embedding_table = nn.Embedding(config.vocab_size, config.embedding_size)
        self.MAX_LENGTH = config.max_decoding_steps # 最大解码长度
        # 初始化encoder和decoder
        self.Encoder = Encoder(embedding_size=config.embedding_size,hidden_size=config.hidden_size,Embedding_table=self.Embedding_table)
        self.Decoder = AttentionPointerCoverageDecoder(embedding_size=config.embedding_size,hidden_size=config.hidden_size,
                                        output_size=config.output_size,Embedding_table=self.Embedding_table,keep_prob=config.keep_prob)

    def forward(self,source,source_lens,target,teacher_forcing_ratio=0.5):
        '''
        :param source: [batch_size,max_time_step],max_time_step指的是Encoder端
        :param source_lens: [batch_size]
        :param target: [batch_size,max_time_step],max_time_step指的是Decoder端
        :param teacher_forcing_ratio: 使用TeacherForcing训练的数据比例
        :return:outputs: [target_len,batch_size,vocab_size] # 各个时刻的解码结果，保持概率分布的形式
        '''
        target_len = target.size(1) # target的序列长度
        batch_size = target.size(0) # 当前批次大小
        max_time_step = source.size(1) # Encoder端的最大时间步
        # Encoder
        Encoder_outputs,hidden,current_cell_state = self.Encoder(source,source_lens)

        # Coverage vector 覆盖向量，初始为0,[batch_size,1,max_time_step]
        CoverageVector = torch.zeros((batch_size,1,max_time_step),dtype=torch.float32)
        # Coverage损失
        CoverageLoss = torch.tensor(0,dtype=torch.float32)

        # Decoder
        # 以输出的各单元概率的形式，存储预测序列，便于交叉熵计算
        outputs = torch.zeros(target_len,batch_size,self.output_size)
        # 输入解码起始符_GO开始解码,需要给batch中每条数据都输入
        GO_token = torch.tensor([[1]]*batch_size)
        decoder_input = GO_token
        # 解码长度为 batch中的序列长度
        for step in range(target_len):
            output, hidden, current_cell_state,attention_weights,copy_weights,CoverageVector,Coverageloss = self.Decoder(decoder_input,hidden,current_cell_state,Encoder_outputs,source,CoverageVector,flag=True)
            outputs[step] = output # 记录当前时刻的预测结果
            CoverageLoss += Coverageloss # 记录Coverage损失
            # 如果为True，则TeacherForcing
            TeacherForcing = random.random() < teacher_forcing_ratio # 随机生成[0,1)之间的数
            topv,topi = output.topk(1) # topk对应的具体值，topi对应相应的索引，即ID
            # target: [batch_size,max_time_step] ，取出当前时刻的一个batch的target
            decoder_input = target[:,step] if TeacherForcing else topi
            decoder_input = decoder_input.view(-1,1) # 将维度设置为[batch_size,1]
        CoverageLoss = Variable(CoverageLoss,requires_grad=True) # Coverage 损失
        return outputs ,CoverageLoss # [max_time,batch_size,vocab_size],tensor

    def BatchSample(self,source,source_lens):
        '''
        批量预测
        :param source: batch输入，[batch,max_time_step]
        :param source_lens: batch输入中各个样例的真实长度
        :return: 返回预测结果 和 注意力权重分布
        '''
        batch_size = source.size(0)  # 当前批次大小
        max_time_step = source.size(1)  # Encoder端的最大时间步max_time_step = source.size(1) # Encoder端的最大时间步
        # Encoder
        # Encoder_outputs: ([batch_size,max_time_step,hidden_size*2],source_lens) 这是个元组
        # hiddens:[1,batch_size,hidden_size*2]
        # current_cell_state: [1,batch_size,hidden_size*2]
        Encoder_outputs, hiddens,current_cell_states = self.Encoder(source, source_lens)

        # Decoder
        # 记录batch中，每条数据的各个时刻的预测结果,解码的最大长度为self.MAX_LENGTH
        results = []
        # 记录每个batch中每条数据的各个时刻的注意力权重
        attention_weights = []
        # 输入解码起始符_GO开始解码
        GO_token = torch.tensor([1])
        # 解码终止符
        EOS_token = torch.tensor(2)

        # 逐一对batch中的各条数据解码
        for i in range(batch_size):
            # 存储当前序列的解码结果
            result = []
            # 存储当前序列各个时刻的注意力结果
            atten = []
            # 解码起始符
            decoder_input = GO_token
            hidden = hiddens[:,i,:] # 当前数据对应的Encoder最后时刻前后向的隐藏状态,[1,hidden_size*2]
            current_cell_state = current_cell_states[:,i,:] # 当前数据对应的Encoder最后时刻前后向的细胞状态,[1,hidden_size*2]
            Encoder_output = Encoder_outputs[0][i] # 当前数据对应的Encoder各个时刻的输出
            # 由于是单条数据，因此各个维度必须要符合Decoder的输入
            decoder_input = decoder_input.unsqueeze(0)
            hidden = hidden.unsqueeze(1) # [1,1,hidden_size*2]
            current_cell_state = current_cell_state.unsqueeze(1) # [1,1,hidden_size*2]
            Encoder_output = (Encoder_output.unsqueeze(0),0) # 将Encoder_output转为二元元组，以符合Decoder的输入

            # Coverage vector 覆盖向量，初始为0, [1,1,max_time_step]
            CoverageVector = torch.zeros((1, 1, max_time_step),dtype=torch.float32)

            for j in range(self.MAX_LENGTH):
                # output: [1,output_size]
                output, hidden,current_cell_state, attention_weight,copy_weight,CoverageVector,Coverageloss = self.Decoder(decoder_input,hidden,current_cell_state,Encoder_output,source,CoverageVector,flag=False)
                topv,topi = output.topk(1)
                result.append(topi.item())
                atten.append(attention_weight.tolist())
                decoder_input = topi.view(-1,1)
                if topi == EOS_token:
                    break
            results.append(result)
            attention_weights.append(atten)
        return results,attention_weights # [batch_size,some_time_step]

if __name__ == '__main__':
    from Config import Config
    config = Config()
    model = PointerGeneratorCoverage(config)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')