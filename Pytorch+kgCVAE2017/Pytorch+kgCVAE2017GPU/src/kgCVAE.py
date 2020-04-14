# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : kgCVAE.py

# ----------------------------------------------------------------------------------------------------------------------
# knowledge-guided CVAE
# kgCVAE: 基于Tiancheng Zhao等人的论文《Learning Discourse-level Diversity for Neural Dialog Models using Conditional
# Variational Autoencoders》而实现，用于文本摘要的生成，其中：
# 1. 我们用文档 表示 the dialog context c,文档中的每个句子表示一个utterance
# 2. 我们用文档的摘要 表示 the response utterance x
# 3. 论文中meta features m 表示额外的特征信息，这里我们不添加此类特征；若果要增加关键词特征信息，可以在此处添加
# 4. 论文中会话层conversational floor 在原论文中表示当前的utterance是否与the response utterance x是同一个人，若是为1，否则为0，
#    这里我们同样不添加此类特征信息到utterance中
# 5. 论文中the linguistic features y 表示the set of linguistic features ，即dialog act ,我们不添加此类特征到 x 中
# 6. 论文中的MLP to predict y0 = MLPy(z; c) based on z and c，该层也无需实现

# 条件是文档，输入是摘要，目标重构摘要，这里条件与目标不独立，体现在KL损失中；
# 训练时，输入文档和摘要，目标重构摘要；预测时，输入文档，目标是生成摘要；
# ----------------------------------------------------------------------------------------------------------------------

from Encoder import Encoder
from ResponseDecoder import ResponseDecoder
import torch
import torch.nn as nn
import random

class kgCVAE(nn.Module):

    def __init__(self,config):
        super(kgCVAE,self).__init__()
        self.output_size = config.output_size # 输出空间大小
        # 词向量表用于查询单词词向量,Encoder和Decoder共用同一张表
        self.Embedding_table = nn.Embedding(config.vocab_size, config.embedding_size)
        self.MAX_LENGTH = config.max_decoding_steps # 最大解码长度
        # 初始化encoder和decoder
        self.Encoder = Encoder(embedding_size=config.embedding_size,hidden_size=config.hidden_size,Embedding_table=self.Embedding_table)
        self.Decoder = ResponseDecoder(embedding_size=config.embedding_size,hidden_size=config.hidden_size,
                                        output_size=config.output_size,Embedding_table=self.Embedding_table,keep_prob=config.keep_prob)
        # 将Encoder的输出经过线性层变换，作为ResponseDecoder的初始隐状态
        self.initHidden = nn.Linear(in_features=config.hidden_size*2,out_features=config.hidden_size)
        # MLPb
        self.MLP = nn.Linear(in_features=config.hidden_size*2,out_features=config.output_size)

    def forward(self,source,source_lens,target,target_lens,batch_size,descendingOrders,input_article_lens,teacher_forcing_ratio=0.5):
        '''
        :param source: [num*batch_size,max_time_step],句子集合,num为每篇文章中句子的最大个数，max_time_step为句子的最大长度,下同
        :param source_lens: [num*batch_size],是句子集合中各条数据未填充前的长度，倒排
        :param target: [batch_size,max_time_step],每篇文章对应的摘要集合,batch_size为当前批次文章的个数,max_time_step为摘要句子最大长度
                        预测时为None
        :param target_lens: [batch_size,] 每篇文章对应的摘要的实际长度，预测时为None
        :param batch_size: 当前批次的文章数量，注意并非句子数量(source.size(0))
        :param descendingOrders: 用于恢复句子集合input中每个句子原始顺序的索引列表
        :param input_article_lens: 由句子集合input构成的batch_size篇文章中，每篇文章的真实句子个数
        :param teacher_forcing_ratio: 使用TeacherForcing训练的数据比例
        :return:outputs: [target_len,batch_size,vocab_size] # 各个时刻的解码结果，保持概率分布的形式
        '''
        target_len = target.size(1) # target的序列长度
        # Encoder
        # hidden: [batch_size,1,hidden_size*2],作为ResponseDecoder初始时刻隐状态，但需要先经过一个线性层
        # KLD: D(q(z|x,c)||p(z|c))
        # flag: 取值为'Train' or 'Inference',用于判断当前是训练还是预测，以决定隐变量Z的采样方式
        hidden,KLD = self.Encoder(source,source_lens,target,target_lens,batch_size,descendingOrders,input_article_lens,flag='Train')

        # ------------------------------------  MLPb ------------------------------------------
        # 该网络用于避免出现KL散度消失问题，计算 BOW损失
        # The idea is to introduce an auxiliary loss that requires the decoder network
        # to predict the bag-of-words in the response x
        # 在实现上是将Encoder的输出hidden，经过一个线性层变换，然后生成词包中各个单词的概率。由于hidden第二个维度为1，
        # 我们将要预测的是一个序列，因此需要将softmax后的结果复制num份，num是当前batch中文章最大的句子个数，地位相当于
        # max_time_step,之后将各个时刻对应的最大单词的概率相乘，求取使序列概率最大的各个时刻的预测结果。显然，我们是通过
        # 直接复制多份第一个时刻的结果而得到的后续多个时刻的预测结果，因此各个时刻的预测结果事实上是一致的。然而，在训练过程中
        # 由于我们期望能获得各个时刻的真实预测结果，这会导致，这些时刻的真实值所对应的词包中相应位置的单词概率趋于一致
        # 即，假如target中有A,B,C三个单词，则mlp_output = torch.softmax(mlp_output,dim=1)中，A,B,C三个位置的单词的概率趋于
        # 相等，通过这种方式，可以迫使Encoder中，隐变量Z中包含X，不至于使隐变量Z与X相互独立，使得X的生成与Z无关，导致KL散度为0或消失
        mlp_output = self.MLP(hidden) # [batch_size,1,output_size]
        mlp_output = mlp_output.squeeze(1) # [batch_size,output_size]
        mlp_output = torch.softmax(mlp_output,dim=1) # [batch_size,output_size]
        mlp_output = mlp_output.unsqueeze(1) # [batch_size,1,output_size]
        mlp_outputs = mlp_output.repeat(1,target_len,1) # [batch_size,target_len,output_size]
        mlp_outputs = mlp_outputs.permute(1,0,2) # [target_len,batch_size,output_size]

        # Decoder
        initHidden = self.initHidden(hidden) # [batch_size,1,hidden_size]
        # 以输出的各单元概率的形式，存储预测序列，便于交叉熵计算
        outputs = torch.zeros(target_len,batch_size,self.output_size).cuda()
        # 输入解码起始符_GO开始解码,需要给batch中每条数据都输入
        GO_token = torch.tensor([[1]]*batch_size).cuda()
        decoder_input = GO_token
        # 解码长度为 batch中的序列长度
        for step in range(target_len):
            output, initHidden = self.Decoder(decoder_input,initHidden,flag=True)
            outputs[step] = output # 记录当前时刻的预测结果
            # 如果为True，则TeacherForcing
            TeacherForcing = random.random() < teacher_forcing_ratio # 随机生成[0,1)之间的数
            topv,topi = output.topk(1) # topk对应的具体值，topi对应相应的索引，即ID
            # target: [batch_size,max_time_step] ，取出当前时刻的一个batch的target
            decoder_input = target[:,step] if TeacherForcing else topi
            decoder_input = decoder_input.view(-1,1).contiguous() # 将维度设置为[batch_size,1]
        return outputs,KLD,mlp_outputs # [max_time_step,batch_size,vocab_size],scalar,[max_time_step,batch_size,vocab_size]

    def BatchSample(self,source,source_lens,batch_size,descendingOrders,input_article_lens):
        '''
        批量预测
        :param source: [num*batch_size,max_time_step],句子集合,num为每篇文章中句子的最大个数，max_time_step为句子的最大长度,下同
        :param source_lens: [num*batch_size],是句子集合中各条数据未填充前的长度，倒排
        :param batch_size: 当前批次的文章数量，注意并非句子数量(source.size(0))
        :param descendingOrders: 用于恢复句子集合input中每个句子原始顺序的索引列表
        :param input_article_lens: 由句子集合input构成的batch_size篇文章中，每篇文章的真实句子个数
        :return: 返回预测结果 和 注意力权重分布
        '''
        # Encoder
        # hiddens:[batch_size,1,hidden_size*2]
        # flag: 取值为'Train' or 'Inference',用于判断当前是训练还是预测，以决定隐变量Z的采样方式
        hiddens = self.Encoder(source,source_lens,None,None,batch_size,descendingOrders,input_article_lens,flag='Inference')

        # Decoder
        initHidden = self.initHidden(hiddens)  # [batch_size,1,hidden_size]
        # 记录batch中，每条数据的各个时刻的预测结果,解码的最大长度为self.MAX_LENGTH
        results = []
        # 输入解码起始符_GO开始解码
        GO_token = torch.tensor([1]).cuda()
        # 解码终止符
        EOS_token = torch.tensor(2).cuda()

        # 逐一对batch中的各条数据解码
        for i in range(batch_size):
            # 存储当前序列的解码结果
            result = []
            # 解码起始符
            decoder_input = GO_token
            hidden = initHidden[i] # 当前数据对应的Encoder最后时刻隐藏状态
            # 由于是单条数据，因此各个维度必须要符合Decoder的输入
            decoder_input = decoder_input.unsqueeze(0)
            hidden = hidden.unsqueeze(0)
            for j in range(self.MAX_LENGTH):
                # output: [1,output_size]
                output, hidden = self.Decoder(decoder_input, hidden,flag=False)
                topv,topi = output.topk(1)
                result.append(topi.item())
                decoder_input = topi.view(-1,1).contiguous()
                if topi == EOS_token:
                    break
            results.append(result)
        return results # [batch_size,some_time_step]

if __name__ == '__main__':
    from Config import Config
    config = Config()
    model = kgCVAE(config)
    print(model)