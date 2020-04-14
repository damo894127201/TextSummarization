# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : AttentionCopyCoverageDecoder.py

#---------------------------------------------------------------------------------------------------------------------
# 基于论文: YONG ZHANG AND WEIDONG XIAO等人的 Keyphrase Generation Based on Deep Seq2seq Model
# Decoder: 使用单向双层的GRU,注意论文中的Decoder图示画错了！

# 1. 基于传统Attention+Encoder+Decoder的生成模块
# 2. 基于Copy机制的拷贝模块
# 3. 预测单词的概率 = 生成模块的概率 + copy模块的概率(只增幅哪些在source中且在vocabulary中的单词)
# 4. 与CopyRNN的区别：
#      1. CopyRNN的Encoder和Decoder都使用单层的GRU，而CovRNN使用双层的GRU
#      2. CopyRNN和Seq2Seq+Attention注意力计算使用的是前一刻GRU输出的隐状态与Enocder各个时刻的隐状态，而CovRNN使用的是当前时刻
#         GRU的输出隐状态与Encoder各个时刻的隐状态
#      3. 生成模块的概率Pvocab计算方式：将当前时刻的隐状态和注意力向量拼接在一起通过两个全连接网络+softmax层计算,
#                                       而CopyRNN和Seq2Seq+Attention，则是将当前输入、当前隐状态和注意力向量拼接在一起
#                                       经过一个全连接网络+softmax层计算
#      4. copy模块的概率：增加那些在原文中出现的单词的生成概率
#      5. 选择概率Pgen：决定生成模块概率和copy模块概率各占的权重
#----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttentionCopyCoverageDecoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,output_size,Embedding_table,keep_prob):
        super(AttentionCopyCoverageDecoder,self).__init__()
        self.hidden_size = hidden_size # Encoder的RNN单元的维度,Decoder单元的维度要适应Encoder的维度
        self.output_size = output_size # 解码器输出词包大小
        self.Embedding_table = Embedding_table # 词向量表用于查询单词词向量
        self.drop_out = nn.Dropout(keep_prob)  # 在输入层后设置Dropout层

        # 前馈网络，一个线性变换，将上一时刻隐藏层状态和Enocder中某个RNN的输出拼接一起，计算注意力权重
        self.alignment = nn.Linear(self.hidden_size*6, 1)
        self.GRU = nn.GRU(input_size=embedding_size,hidden_size=self.hidden_size*2,num_layers=2,batch_first=False) # Decoder解码单元
        self.coverage = nn.Linear(1,self.hidden_size*2) # 对coverageVector做线性变换，便于计算注意力权重
        self.FullConnection1 = nn.Linear(self.hidden_size*4,self.hidden_size*4) # 计算Pvocab的第一个全连接层,不改变输入维度
        self.FullConnection2 = nn.Linear(self.hidden_size*4,self.output_size) # 计算Pvocab的第二个全连接层,改变输入维度为输出空间的大小
        self.gen = nn.Linear(self.hidden_size*4+embedding_size,1) # 用于选择门的概率生成的线性变换

    def forward(self,input,pre_hidden,Encoder_outputs,sourceInput,CoverageVector,flag=True):
        '''
        :param input: [batch_size,1] 批量的单个字符的ID，即上一解码时刻的输出
        :param pre_hidden: [2,batch_size,hidden_size*2],上一时刻的隐状态，初始时为Enocder最后一时刻的隐状态，由于是双层GRU，因此第一个维度为2
                                                         表示各层的初始隐状态，或各层上一时刻的隐状态
        :param Encoder_outputs: ([batch_size,max_time_step,hidden_size*2],source_lens),Encoder各个时刻RNN的输出，用于注意力权重计算
        :param sourceInput: [batch_size,max_time_step],编码器端的输入序列,用于拷贝机制
        :param CoverageVector: coverage vector, 覆盖向量,用于记录Encoder各个时刻的注意力权重累计和,作用是抑制Decoder关注那些已被关注过的位置
                              The coverage vector is the cumulative sum of attention distributions over all previous decoder steps
                              [batch_size,1,max_time_step]
        :param: flag为True,表示训练，采用dropout;为False,表示预测,不采用dropout
        :return: 返回当前时刻的预测的结果，即每个可能值的概率；当前隐藏层状态；当前时刻的注意力权重; 当前时刻Encoder各个位置的拷贝权重
                 CoverageVector ; CoverageLoss
        '''
        max_time_steps = Encoder_outputs[0].size(1)  # Encoder端每个batch中填充后的长度

        # Decoder端输入input: [batch_size,1,embedding_size]
        embeddings = self.Embedding_table(input)
        if flag == True:
            embeddings = self.drop_out(embeddings) # dropout层

        # -----------------------------------基于注意力机制的生成模块---------------------------------------------------
        # ----------------------------------论文 III. METHODOLOGY C. ATTENTION MECHANISM -------------------------------

        # 根据当前的输入计算当前的GRU隐状态

        # 计算当前解码时刻的隐状态current_hidden
        # embeddings: [batch_size,1,embedding_size]
        # pre_hidden:[2,batch_size,hidden_size*2]
        # output: [1,batch_size,hidden_size*2]
        # current_hidden: [2,batch_size,hidden_size*2]
        input_embeddings = embeddings.permute(1,0,2) # [1,batch_size,embedding_size]
        output, current_hidden = self.GRU(input_embeddings, pre_hidden)

        # 计算注意力向量context_vector ,基于公式18

        # 由于current_hidden包含了当前时刻上下两层的隐状态值，在计算注意力时，我们有两种选择：
        # 上下两层的值都采用，或只采用最后一层的隐状态值。这里我们依照论文中所述，只采用最后一层的隐状态输出。
        # 获取当前时刻最后一层GRU的隐状态,并改造形状便于计算
        last_layers_current_hidden = current_hidden[1,:,:].view(-1,1,self.hidden_size*2) # [batch_size,1,hidden_size*2]
        # 将last_layers_current_hidden沿着第1维复制max_time_step份，便于与Encoder_outputs的max_time_step个值逐一拼接，计算注意力权重
        multiple_current_hidden = last_layers_current_hidden.repeat(1,max_time_steps,1) # [batch_size,max_time_step,hidden_size*2]

        # 计算注意力权重

        # CoverageVector: [batch_size,1,max_time_step]
        # 将CoverageVector经过线性变换，变为[batch_size,max_time_step,hidden_size*2]
        temp_CoverageVector = CoverageVector.permute(0, 2, 1)  # [batch_size,max_time_step,1]
        temp_CoverageVector = self.coverage(temp_CoverageVector)  # [batch_size,max_time_step,hidden_size*2]

        # CoverageVector: [batch_size,max_time_step,hidden_size*2]
        # Encoder_outputs[0]: [batch_size,max_time_step,hidden_size*2]
        # multiple_current_hidden: [batch_size,max_time_step,hidden_size*2]
        # combine_input : [batch_size,max_time_step,hidden_size*2+hidden_size*2+hidden_size*2]
        combine_input = torch.cat((Encoder_outputs[0],multiple_current_hidden,temp_CoverageVector),2)

        # attention_weights: [batch_size,max_time_step,1]
        attention_weights = self.alignment(combine_input)
        attention_weights = torch.tanh(attention_weights)  # 加一层激活函数，作为各个维度的门,但不改变形状，只改变各维度的值[batch_size,max_time_step,1]
        # [batch_size,max_time_step,1]
        attention_weights = F.softmax(attention_weights,dim=1)
        # [batch_size,1,max_time_step]
        attention_weights = attention_weights.view(-1,1,max_time_steps)

        # --------------------------------------------- Coverage Mechanism ---------------------------------------------

        # 计算Coverage损失,用于惩罚那些注意力权重经常比较大的位置
        # coverage mechanism
        # needs an extra loss function to penalize repeatedly attending
        # to the same locations, otherwise it would be ineffective with
        # no discernible reduction in repetition

        # attention_weights: [batch_size,1,max_time_step]
        # CoverageVector: [batch_size,1,max_time_step]
        # 比较上述两个tensor的最后一维度的相应值，取较小值作为当前序列中当前时刻的损失,公式(19)
        aw = attention_weights.permute(0,2,1) # [batch_size,max_time_step,1]
        cv = CoverageVector.permute(0,2,1) # [batch_size,max_time_step,1]
        ac = torch.cat((aw,cv),2) # [batch_size,max_time_step,2]
        CoverageLoss = torch.min(ac,dim=2) # (min,min_index)
        CoverageLoss = torch.sum(CoverageLoss[0].data,dtype=torch.float32)

        CoverageVector += attention_weights  # 累计各个时刻的注意力权重值, [batch_size,1,max_time_step]

        # ----------------------------------------  计算Pvocab  --------------------------------------------------------

        # 计算上下文语境向量context_vector: 公式(10)
        # Encoder_outputs: [batch_size,max_time_step,hidden_size*2]
        # 计算当前时刻的context vector:[batch_size,1,hidden_size*2]
        context_vector = torch.bmm(attention_weights,Encoder_outputs[0])

        # 计算概率Pvocab: 公式(11)
        # last_layers_current_hidden与context_vector拼接起来输入第一个全连接层进行线性变换
        # last_layers_current_hidden: [batch_size,1,hidden_size*2]
        # context_vector: [batch_size,1,hidden_size*2]
        new_input = torch.cat((last_layers_current_hidden,context_vector),2) # [batch_size,1,hidden_size*2+hidden_size*2]
        new_input = self.FullConnection1(new_input) # 第一个全连接层不改变输入维度[batch_size,1,hidden_size*2+hidden_size*2]
        new_input = torch.tanh(new_input) # 加一个激活函数作为各个维度的门,[batch_size,1,hidden_size*2+hidden_size*2]
        new_input = self.FullConnection2(new_input) # [batch_size,1,output_size]
        new_input = new_input.squeeze(1) # 去除第2个维度,[batch_size,output_size]
        # 计算Pvocab,解码当前时刻的输出，生成预测
        Pvocab = F.softmax(new_input,dim=1) # [batch_size,output_size]

        # ---------------------------------------------     Copy机制     -----------------------------------------------
        # --------------------------    论文 III. METHODOLOGY  D. COPY MECHANISM  --------------------------------------

        # 基于论文中所阐述的：copying a word directly from the corresponding source text based on attention distribution
        # 在解码时，我们将输入序列中各个时刻的注意力权重，作为相应时刻对应单词的copy概率
        batch_size = Encoder_outputs[0].size(0) # 当前batch的大小
        source_position_copy_prob = torch.tensor([[0]*self.output_size]*batch_size,dtype=torch.float32) # 存储当前解码时刻，source端各个位置的累积的copy概率(注意力权重),
                                                                 # 其形状要与Pvocab一致，便于两者相加
                                                                 # [batch_size,output_size]
        # 遍历每条数据
        for i in range(batch_size):
            # 当前输入序列的ID序列
            A = sourceInput[i]  # [max_time_step,]
            # 当前输入序列的copy概率(注意力权重)序列
            B = attention_weights[i]
            # 构建一个[max_time_step,output_size]的零tensor,以存储每个时刻output_size维度是否出现有A中的ID
            C = torch.zeros((max_time_steps, self.output_size))
            index = torch.arange(max_time_steps)
            C[index, A] = torch.tensor(1, dtype=torch.float32)
            # 将A中ID在B中的概率累加起来，并存储到维度为output_size的tensor中
            D = torch.matmul(B, C)  # [output_size,]
            source_position_copy_prob[i] = D

        # --------------------------------------------- 选择门Pgen的计算 -----------------------------------------------
        # --------------------------------------- 论文 公式(16) --------------------------------------------------------
        # context vector: [batch_size,1,hidden_size*2]
        # last_layers_current_hidden: [batch_size,1,hidden_size*2]
        # embeddings: [batch_size,1,embedding_size]
        gen_input = torch.cat((context_vector,last_layers_current_hidden,embeddings),2) # [batch_size,1,hidden_size*4+embedding_size]
        gen_prob = self.gen(gen_input) # [batch_size,1,1]
        gen_prob = gen_prob.squeeze(1) # [batch_size,1]
        gen_prob = torch.sigmoid(gen_prob) # [batch_size,1] ,选择门的概率

        #  ---------------------------------------单词最终的概率--------------------------------------------------------
        #  -------------------------------------- 论文 公式(5) ---------------------------------------------------------
        # Pvocab: [batch_size,output_size]
        # source_position_copy_prob: [batch_size,output_size]
        # gen_prob: [batch_size,1]
        # ones : [batch_size,1]
        ones = torch.ones(batch_size,1) # 概率1
        output = Pvocab * gen_prob + source_position_copy_prob * (ones - gen_prob)# [batch_size,output_size]
        output = Variable(output,requires_grad=True) # 需用Variable包裹，并且requires_grad=True,表示对此变量求梯度，参与反向传播

        return output,current_hidden,attention_weights,source_position_copy_prob,CoverageVector,CoverageLoss

if __name__ == '__main__':
    model = AttentionCopyCoverageDecoder(150,300,1000,None,0.5)
    print(model)