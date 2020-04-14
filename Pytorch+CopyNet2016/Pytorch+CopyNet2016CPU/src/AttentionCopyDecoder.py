# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : AttentionCopyDecoder.py

#---------------------------------------------------------------------------------------------------------------------
# 基于论文:Jiatao Gu, Zhengdong Lu等人的 Incorporating Copying Mechanism in Sequence-to-Sequence Learning
# 模型称之为CopyNet，它在CopyRNN的基础上增加了一个Selective Read M 模块，便于生成连续的原文子片段
# 该论文要求Encoder的输入词包要与Decoder的输出词包不一致，并且细分了好几种: 词表+UNK+仅存于输入序列而不在词表中的词集
# 这里，我们不采用论文中划分词包的方式，因为这会使模型代码更加复杂，而且对于同一类语言的生成任务，似乎也没有必要。
# 我们将输入词包和输出词包设置为一致，其它设置及其网络结构均依照论文实现。

# Decoder: 使用单向GRU
# 1. 基于传统Attention+Encoder+Decoder的生成模块
# 2. 基于Copy机制的拷贝模块：
#    From a cognitive perspective, the copying mechanism is related to rote memorization, requiring
#    less understanding but ensuring high literal fidelity
# 3. 预测单词的概率 = 生成模块的概率 + copy模块的概率(只增幅哪些在source中且在vocabulary中的单词)

# 1. Attentive Read
# 2. Selective Read
# 3. Determining when to enter/quit the copy-mode
#----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttentionCopyDecoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,output_size,Embedding_table,keep_prob):
        super(AttentionCopyDecoder,self).__init__()
        self.hidden_size = hidden_size # Encoder的RNN单元的维度,Decoder单元的维度要适应Encoder的维度
        self.output_size = output_size # 解码器输出词包大小
        self.Embedding_table = Embedding_table # 词向量表用于查询单词词向量

        # 前馈网络，一个线性变换，将上一时刻隐藏层状态和Enocder中某个RNN的输出拼接一起，计算注意力权重
        self.alignment = nn.Linear(self.hidden_size*4, 1)
        self.DNN = nn.Linear(embedding_size+self.hidden_size*2,embedding_size) # 用于线性变换[embedding;SelectiveReadM]
        self.GRU = nn.GRU(input_size=embedding_size,hidden_size=self.hidden_size*2,batch_first=False) # Decoder解码单元
        self.atten_combine = nn.Linear(self.hidden_size*2+embedding_size,embedding_size) # 用于将当前解码器输入与注意力语境向量
                                                                                         # 转化为GRU的输入的线性变换，全连接层
        self.out = nn.Linear(self.hidden_size*4+embedding_size,self.output_size) # 线性层输出层
        self.drop_out = nn.Dropout(keep_prob) # 在输入层后设置Dropout层
        self.copy_atten = nn.Linear(self.hidden_size*2,self.hidden_size*2) # 论文中公式(6)sigmoid函数中的部分
                                                                           # 目的在于将Encoder_outputs做线性变换，以便计算copy得分

    def forward(self,input,pre_hidden,Encoder_outputs,sourceInput,flag=True,epsilon=1e-8):
        '''
        :param input: [batch_size,1] 批量的单个字符的ID，即上一解码时刻的输出
        :param pre_hidden: [batch_size,1,hidden_size*2],上一时刻的隐状态，初始时为Enocder最后一时刻的隐状态
        :param Encoder_outputs: ([batch_size,max_time_step,hidden_size*2],source_lens),Encoder各个时刻RNN的输出，用于注意力权重计算
        :param sourceInput: [batch_size,max_time_step],编码器端的输入序列,用于拷贝机制
        :param: flag为True,表示训练，采用dropout;为False,表示预测,不采用dropout
        :param: epsilon: 一个极小值，防止分母为0
        :return: 返回当前时刻的预测的结果，即每个可能值的概率；当前隐藏层状态；当前时刻的注意力权重
        '''
        max_time_steps = Encoder_outputs[0].size(1)  # Encoder端每个batch中填充后的长度
        batch_size = Encoder_outputs[0].size(0)  # 当前batch的大小

        # -------------------                   Attentive Read                         ---------------------------------
        # -----------------------------------  基于注意力机制计算Context_vector向量    ----------------------------------
        # pre_hidden: [batch_size,1,hidden_size*2]
        # 将pre_hidden沿着第1维复制max_time_step份，便于与Encoder_outputs的max_time_step个值逐一拼接，计算注意力权重
        multiple_pre_hidden = pre_hidden.repeat(1, max_time_steps, 1) # [batch_size,max_time_step,hidden_size*2]

        # 计算注意力权重
        # [batch_size,max_time_step,hidden_size*2+hidden_size*2]
        attention_weights = self.alignment(torch.cat((Encoder_outputs[0], multiple_pre_hidden), 2))
        attention_weights = torch.tanh(attention_weights) # 加一层激活函数，但不改变形状，只改变维度的取值，[batch_size,max_time_step,1]
        # [batch_size,max_time_step,1]
        attention_weights = F.softmax(attention_weights, dim=1)
        # [batch_size,1,max_time_step]
        attention_weights = attention_weights.view(-1, 1, max_time_steps)
        # Encoder_outputs: [batch_size,max_time_step,hidden_size*2]
        # 计算当前时刻的context vector:[batch_size,1,hidden_size*2]
        context_vector = torch.bmm(attention_weights, Encoder_outputs[0])

        # ------------------------------------   论文 3.3 State Update  ----------------------------------------

        # Embedding
        # Decoder端输入input: [batch_size,1,embedding_size]
        embeddings = self.Embedding_table(input)

        # Selective Read M
        # is specifically designed for the copy mode
        #  the weighted sum of hidden states in M corresponding to yt
        # 如果当前输入(预测时是上一时刻的输出)存在于Encoder输入序列中，则计算输入序列中相应位置的Encoder隐状态的
        # 加权和，作为当前Decoder输入的一部分，即与输入的embedding拼接一起，作为Decoder的新输入；如果不存在，则将这部分置为0；
        # Selective Read M 实际上换了一种 由 当前输入 计算 当前隐状态的方式，它将当然输入进行了扩充，加入了一部分与当前输入
        # 相关的Enocder阶段各个位置的隐状态的信息作为补充；这可以视为一种注意力计算，不过只考虑输入序列中为当前输入的位置的信息
        # 当前输入可能存在于输入序列中的多个位置，因此各个位置的权重应该基于这多个位置而得到。
        # this particular design potentially helps copy-mode in covering a consecutive sub-sequence of words
        # Selective Read M 有助于copy模块生成原序列中连续的子序列

        # 由于论文公式(9)中并未指明，如何计算与当前输入相关的Enocder各个时刻的权重，因此，我们采用这些时刻各自的注意力权重
        # 然后进行加和计算，之后在这些时刻内计算各自新的权重，最后和各自时刻的隐状态拼接计算当前输入的 Selective Read M
        # input: [batch_size, 1]
        # attention_weights: [batch_size,1,max_time_step]
        # Encoder_outputs[0]: [batch_size,max_time_step,hidden_size*2] ,Encoder各个时刻的隐状态
        # sourceInput: [batch_size,max_time_step],编码器端的输入序列,用于拷贝机制
        # context vector:[batch_size,1,hidden_size*2]
        # embeddings: [batch_size,1,embedding_size]

        # 存储各自时刻的 Selective Read M : [batch_size,1,hidden_size*2],最后一维与Encoder各个时刻的隐状态维度一致(前后向拼接)
        SelectiveReadM = torch.zeros((batch_size,1,self.hidden_size*2),dtype=torch.float32,requires_grad=True)
        # 遍历当前batch中每条数据的当前输入(解码器阶段，一次输入一个字符解码)
        for i in range(batch_size):
            # inputID: tensor(ID)
            # mask: [max_time_steps,]
            # sourceInput[i]: [max_time_steps,]
            # attention_weights[i][0]: [max_time_steps,]
            # weights: [max_time_steps,]
            # Encoder_outputs[0][i]: [max_time_steps,hidden_size*2]
            inputID = input[i][0]  # 当前Decoder输入的单词ID
            # 构造mask矩阵，将SourceInput中所有位置的单词ID与当前输入ID一致的位置置为1
            mask = torch.zeros((max_time_steps)).cuda()  # [max_time_steps,]
            mask[sourceInput[i] == inputID] = torch.tensor(1,dtype=torch.float32)
            # 保留相应位置为当前输入ID的注意力权重
            weights = attention_weights[i][0] * mask  # [max_time_steps,]
            # 计算 SelectiveReadM向量
            readM = torch.matmul(weights / (torch.sum(weights) + epsilon), Encoder_outputs[0][i])  # [1,hidden_size*2]
            SelectiveReadM[i][0] = readM

        # 将input的Embeddings和SelectiveReadM拼接起来，构成新的输入
        # embeddings: [batch_size, 1, embedding_size]
        # Selective Read M : [batch_size,1,hidden_size*2]
        new_embeddings = torch.cat((embeddings,SelectiveReadM),2) # [batch_size,1,embedding_size+hidden_size*2]
        # 经过一个DNN线性变换为[batch_size, 1, embedding_size]
        embeddings = self.DNN(new_embeddings) # [batch_size, 1, embedding_size]
        if flag == True:
            embeddings = self.drop_out(embeddings) # dropout层

        #------------------------------------------ 计算当前时刻的GRU输出的隐状态---------------------------------------

        # context_vector与current_input拼接起来进行线性变换，结果与pre_hidden共同输入GRU中计算current_hidden
        new_input = torch.cat((embeddings,context_vector),2) # [batch_size,1,hidden_size*2+embedding_size]
        new_input = self.atten_combine(new_input) # [batch_size,1,embedding_size]
        new_input = new_input.permute(1,0,2) # 转为GRU输入格式，[1,batch_size,embedding_size]
        new_input = F.relu(new_input) # 加一个激活函数作为各个维度的门

        pre_hidden = pre_hidden.permute(1,0,2) # [1,batch_size,hidden_size*2]

        # 计算当前解码时刻的隐状态current_hidden
        # new_input: [1,batch_size,hidden_size*2]
        # pre_hidden:[1,batch_size,hidden_size*2]
        # output: [1,batch_size,hidden_size*2]
        # current_hidden: [1,batch_size,hidden_size*2]
        output,current_hidden = self.GRU(new_input,pre_hidden)
        current_hidden = current_hidden.permute(1,0,2) # [batch_size,1,hidden_size*2]

        # ---------------------------------------  Generate-Mode & Copy-Mode   -----------------------------------------
        #   ---------------------------            Generate-Mode       --------------------------------------------

        # 当前时刻的输入embeddings: [batch_size,1,embedding_size]
        # 当前时刻的上下文语境向量context_vector: [batch_size,1,hidden_size*2]
        # 当前时刻的隐状态current_hidden: [batch_size,1,hidden_size*2]
        # 将上述三者拼接一起，然后通过一个线性变换层+softmax层生成各个单词的概率
        output = torch.cat((embeddings,context_vector,current_hidden),2) # [batch_size,1,embedding_size+hidden_size*4]
        output = output.squeeze(1) # [batch_size,embedding_size+hidden_size*4]
        # 解码当前时刻的输出，生成预测
        output = self.out(output) # [batch_size,output_size]
        # 转为概率
        Gen_output = F.softmax(output,dim=1)

        # --------------------------------------------------Copy机制----------------------------------------------------
        # ----------------------------------------     Copy-Mode            --------------------------------------------

        batch_size = Encoder_outputs[0].size(0) # 当前batch的大小
        copy_atten = self.copy_atten(Encoder_outputs[0]) # [batch_size,max_time_step,hidden_size*2]
        copy_atten = torch.sigmoid(copy_atten) # [batch_size,max_time_step,hidden_size*2]
        temp_current_hidden = current_hidden.permute(0,2,1) # [batch_size,hidden_size*2,1]
        copy_attention_weithts = torch.matmul(copy_atten,temp_current_hidden) # [batch_size,max_time_step,1] ,得到每一时刻的权重得分
        copy_attention_weithts = copy_attention_weithts.squeeze(2) # 去除最后一个维度，且其维度为1，维度变为[batch_size,max_time_step]
        # 计算source序列中，各个位置的得分概率分布
        copy_attention_weithts = F.softmax(copy_attention_weithts,dim=1) # [batch_size,max_time_step]
        #batch_size = copy_attention_weithts.size(0) # 当前batch的大小
        source_position_prob = torch.tensor([[0] * self.output_size] * batch_size, dtype=torch.float32)
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
            source_position_prob[i] = D

        #  ---------------------------------------单词最终的概率--------------------------------------------------------
        #  -------------------------------------- 论文 公式(4) ---------------------------------------------------------

        output = Gen_output + source_position_prob # [batch_size,max_time_step]
        output = Variable(output,requires_grad=True) # 需用Variable包裹，并且requires_grad=True,表示对此变量求梯度，参与反向传播

        return output,current_hidden,attention_weights,copy_attention_weithts

if __name__ == '__main__':
    model = AttentionCopyDecoder(150,300,1000,None,0.5)
    print(model)