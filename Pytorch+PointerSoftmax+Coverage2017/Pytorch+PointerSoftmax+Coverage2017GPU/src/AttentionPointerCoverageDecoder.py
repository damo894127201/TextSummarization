# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : AttentionPointerCoverageDecoder.py

#---------------------------------------------------------------------------------------------------------------------
# 基于论文: Abigail See等人的 Get To The Point: Summarization with Pointer-Generator Networks
# 模型称之为PointerGenerator+Coverage,这里PointerGenerator就是PointerSoftmax
# Decoder: 使用单向单层的LSTM

# 1. 基于传统Attention+Encoder+Decoder的生成模块
# 2. 基于PointerGenerator机制的拷贝模块
# 3. 预测单词的概率 = 生成模块的概率*选择概率Pgen + PointerGenerator模块的概率*(1-选择概率Pgen)(只增幅哪些在source中且在vocabulary中的单词)
# 4. 与CovRNN的区别：
#    CovRNN的Encoder使用双向双层的GRU，Decoder使用单向双层的GRU；PointerGenerator+Coverage的Enocder使用双向单层的LSTM
#    Decoder使用单向单层的LSTM；其它结构计算方式一样
#----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AttentionPointerCoverageDecoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,output_size,Embedding_table,keep_prob):
        super(AttentionPointerCoverageDecoder,self).__init__()
        self.hidden_size = hidden_size # Encoder的RNN单元的维度,Decoder单元的维度要适应Encoder的维度
        self.output_size = output_size # 解码器输出词包大小
        self.Embedding_table = Embedding_table # 词向量表用于查询单词词向量
        self.drop_out = nn.Dropout(keep_prob)  # 在输入层后设置Dropout层

        # 前馈网络，一个线性变换，将上一时刻隐藏层状态和Enocder中某个RNN的输出拼接一起，计算注意力权重
        self.alignment = nn.Linear(self.hidden_size*6, 1)
        self.BiLSTM = nn.LSTM(input_size=embedding_size,hidden_size=self.hidden_size*2,batch_first=False) # Decoder解码单元
        self.coverage = nn.Linear(1,self.hidden_size*2) # 对coverageVector做线性变换，便于计算注意力权重
        self.FullConnection1 = nn.Linear(self.hidden_size*4,self.hidden_size*4) # 计算Pvocab的第一个全连接层,不改变输入维度
        self.FullConnection2 = nn.Linear(self.hidden_size*4,self.output_size) # 计算Pvocab的第二个全连接层,改变输入维度为输出空间的大小
        self.gen = nn.Linear(self.hidden_size*4+embedding_size,1) # 用于选择门的概率生成的线性变换

    def forward(self,input,pre_hidden,pre_cell_state,Encoder_outputs,sourceInput,CoverageVector,flag=True):
        '''
        :param input: [batch_size,1] 批量的单个字符的ID，即上一解码时刻的输出
        :param pre_hidden: [1,batch_size,hidden_size*2],上一时刻的隐状态，初始时为Enocder最后一时刻的前后向隐状态的拼接，
                                                        由于Decoder是单层单向的LSTM，因此第一个维度为1
                                                        表示初始隐状态，或上一时刻的隐状态
        :param pre_cell_state:  [1,batch_size,hidden_size*2],上一时刻的细胞状态，初始时为Enocder最后一时刻的前后向细胞状态的拼接，
                                                             由于Decoder是单层单向的LSTM，因此第一个维度为1
                                                             表示初始细胞状态，或上一时刻的细胞状态
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
        # ----------------------------------论文 2.1 Sequence-to-sequence attentional model -------------------------------

        # 根据当前的输入计算当前的GRU隐状态

        # 计算当前解码时刻的隐状态current_hidden和current_cell_state
        # embeddings: [batch_size,1,embedding_size]
        # pre_hidden:[1,batch_size,hidden_size*2]
        # pre_cell_state: [1,batch_size,hidden_size*2]
        # output: [1,batch_size,hidden_size*2]
        # current_hidden: [1,batch_size,hidden_size*2]
        # current_cell_state: [1,batch_size,hidden_size*2]
        input_embeddings = embeddings.permute(1,0,2) # [1,batch_size,embedding_size]
        output, (current_hidden,current_cell_state) = self.BiLSTM(input_embeddings,(pre_hidden,pre_cell_state))

        # 计算注意力向量context_vector ,基于公式18

        # 获取当前时刻LSTM的隐状态current_hidden,并改造形状便于计算
        temp_current_hidden = current_hidden.view(-1,1,self.hidden_size*2).contiguous() # [batch_size,1,hidden_size*2]
        # 将temp_current_hidden沿着第1维复制max_time_step份，便于与Encoder_outputs的max_time_step个值逐一拼接，计算注意力权重
        multiple_current_hidden = temp_current_hidden.repeat(1,max_time_steps,1) # [batch_size,max_time_step,hidden_size*2]

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
        attention_weights = torch.tanh(attention_weights)  # 加一层激活函数，但不改变形状，只改变各维度的值[batch_size,max_time_step,1]
        # [batch_size,max_time_step,1]
        attention_weights = F.softmax(attention_weights,dim=1)
        # [batch_size,1,max_time_step]
        attention_weights = attention_weights.view(-1,1,max_time_steps).contiguous()

        # --------------------------------------------- 2.3 Coverage mechanism -----------------------------------------

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
        # temp_current_hidden与context_vector拼接起来输入第一个全连接层进行线性变换
        # temp_current_hidden: [batch_size,1,hidden_size*2]
        # context_vector: [batch_size,1,hidden_size*2]
        new_input = torch.cat((temp_current_hidden,context_vector),2) # [batch_size,1,hidden_size*2+hidden_size*2]
        new_input = self.FullConnection1(new_input) # 第一个全连接层不改变输入维度[batch_size,1,hidden_size*2+hidden_size*2]
        new_input = torch.tanh(new_input) # 加一个激活函数作为各个维度的门,[batch_size,1,hidden_size*2+hidden_size*2]
        new_input = self.FullConnection2(new_input) # [batch_size,1,output_size]
        new_input = new_input.squeeze(1) # 去除第2个维度,[batch_size,output_size]
        # 计算Pvocab,解码当前时刻的输出，生成预测
        Pvocab = F.softmax(new_input,dim=1) # [batch_size,output_size]

        # ---------------------------------------------     Copy机制     -----------------------------------------------

        # 基于论文中所阐述的：copying a word directly from the corresponding source text based on attention distribution
        # 在解码时，我们将输入序列中各个时刻的注意力权重，作为相应时刻对应单词的copy概率
        batch_size = Encoder_outputs[0].size(0) # 当前batch的大小
        source_position_copy_prob = torch.tensor([[0]*self.output_size]*batch_size,dtype=torch.float32).cuda() # 存储当前解码时刻，source端各个位置的累积的copy概率(注意力权重),
                                                                 # 其形状要与Pvocab一致，便于两者相加
                                                                 # [batch_size,output_size]
        # 遍历每条数据
        for i in range(batch_size):
            # 当前输入序列的ID序列
            A = sourceInput[i]  # [max_time_step,]
            # 当前输入序列的copy概率(注意力权重)序列
            B = attention_weights[i]
            # 构建一个[max_time_step,output_size]的零tensor,以存储每个时刻output_size维度是否出现有A中的ID
            C = torch.zeros((max_time_steps, self.output_size)).cuda()
            index = torch.arange(max_time_steps).cuda()
            C[index, A] = torch.tensor(1,dtype=torch.float32).cuda()
            # 将A中ID在B中的概率累加起来，并存储到维度为output_size的tensor中
            D = torch.matmul(B, C)  # [output_size,]
            source_position_copy_prob[i] = D

        # --------------------------------------------- 选择门Pgen的计算 -----------------------------------------------
        # --------------------------------------- 论文 2.2 Pointer-generator network --------------------------------------------------------
        # context vector: [batch_size,1,hidden_size*2]
        # temp_current_hidden: [batch_size,1,hidden_size*2]
        # embeddings: [batch_size,1,embedding_size]
        gen_input = torch.cat((context_vector,temp_current_hidden,embeddings),2) # [batch_size,1,hidden_size*4+embedding_size]
        gen_prob = self.gen(gen_input) # [batch_size,1,1]
        gen_prob = gen_prob.squeeze(1) # [batch_size,1]
        gen_prob = torch.sigmoid(gen_prob) # [batch_size,1] ,选择门的概率

        #  ---------------------------------------单词最终的概率--------------------------------------------------------
        #  -------------------------------------- 论文 公式(5) ---------------------------------------------------------
        # Pvocab: [batch_size,output_size]
        # source_position_copy_prob: [batch_size,output_size]
        # gen_prob: [batch_size,1]
        # ones : [batch_size,1]
        ones = torch.ones(batch_size,1).cuda() # 概率1
        output = Pvocab * gen_prob + source_position_copy_prob * (ones - gen_prob)# [batch_size,output_size]
        output = Variable(output,requires_grad=True).cuda() # 需用Variable包裹，并且requires_grad=True,表示对此变量求梯度，参与反向传播

        return output,current_hidden,current_cell_state,attention_weights,source_position_copy_prob,CoverageVector,CoverageLoss

if __name__ == '__main__':
    model = AttentionPointerCoverageDecoder(150,300,1000,None,0.5)
    print(model)