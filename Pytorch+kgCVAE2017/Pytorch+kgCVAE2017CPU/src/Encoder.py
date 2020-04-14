# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:42
# @Author  : Weiyang
# @File    : Encoder.py

# ---------------------------------------------------------------------------------------------------
# Encoder: 集成UtteranceEncoder、ContextEncoder和Recognition Network、Prior Network
# ---------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from UtteranceEncoder import UtteranceEncoder
from ContextEncoder import ContextEncoder
from recoveryOrder import recoveryOrder
from Gaussian_kld import Gaussian_kld

class Encoder(nn.Module):
    def __init__(self,embedding_size,hidden_size,Embedding_table):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size # Encoder的RNN计算单元维度
        self.Embedding_table = Embedding_table # 词向量表用于查询单词词向量
        # 句子编码器
        self.UtteranceEncoder = UtteranceEncoder(embedding_size=embedding_size,hidden_size=hidden_size)
        # 文章编码器
        self.ContextEncoder = ContextEncoder(hidden_size=hidden_size)

        # Recognition Network
        # 计算均值和log方差，结果的前一半为均值，后一半为log方差
        self.RecognitionNetwork = nn.Linear(in_features=hidden_size*3,out_features=hidden_size*2)

        # Prior Network
        # 计算均值和log方差，结果的前一半为均值，后一半为log方差
        self.PriorNetwork = nn.Linear(in_features=hidden_size,out_features=hidden_size*2)

    def forward(self,input,input_lens,target,target_lens,batch_size,descendingOrders,input_article_lens,flag):
        '''
        :param input: [num*batch_size,max_time_step] ,句子集合,num为每篇文章中句子的最大个数，max_time_step为句子的最大长度,下同
        :param input_lens: [num*batch_size],是句子集合中各条数据未填充前的长度，倒排
        :param target: [batch_size,max_time_step],每篇文章对应的摘要集合,batch_size为当前批次文章的个数,max_time_step为摘要句子最大长度
                        预测时为None
        :param target_lens: [batch_size,] 每篇文章对应的摘要的实际长度，预测时为None
        :param batch_size: 当前批次的文章数量，注意并非句子数量
        :param descendingOrders: 用于恢复句子集合input中每个句子原始顺序的索引列表
        :param input_article_lens: 由句子集合input构成的batch_size篇文章中，每篇文章的真实句子个数
        :param flag: 取值为'Train' or 'Inference',用于判断当前是训练还是预测，以决定隐变量Z的采样方式
        :return: 返回抽样结果z与文章编码c结果的拼接,作为解码器的输入;以及KL散度，预测时不返回该值
        '''
        # 查表获取词向量序列,维度为[batch_size,max_time_step,embedding_size]
        sentence_embeddings = self.Embedding_table(input)

        # ------------------------------------------   UtteranceEncoder  ----------------------------------------------
        # 对当前batch中每篇文章中每个句子进行编码,由于进行过倒排，因此enforce_sorted=True
        _,sentenceEncoding = self.UtteranceEncoder(sentence_embeddings,input_lens,enforce_sorted=True) # [num*batch_size,hidden_size*2]
        # 由于pack_padded_sequence时,打乱了句子顺序，因此需要恢复其原有顺序
        sentenceEncoding = recoveryOrder(sentenceEncoding,descendingOrders) # [num*batch_size,hidden_size*2]

        # ------------------------------------------   ContextEncoder  ------------------------------------------------
        # 调整句子编码结果的维度，将属于同一篇文章的句子划分到一起
        sentenceEncoding = sentenceEncoding.view(batch_size,-1,self.hidden_size*2) # [batch_size,max_sent_num,hidden_size*2]
        # 对当前batch中每篇文章进行编码,结果作为prior network 网络的输入，未进行过排序，enforce_sorted=False
        _,articleEncoding = self.ContextEncoder(sentenceEncoding,input_article_lens,enforce_sorted=False) # [batch_size,hidden_size]

        # ------------------------------------------  Train or Inference  --------------------------------------------
        if flag == 'Train':
            # 查询词向量表，获取target的词向量，维度为[batch_size,max_time_step,embedding_size]
            target_embeddings = self.Embedding_table(target)
            # 对target进行句子编码，未进行过排序，enforce_sorted=False
            _,targetEncoding = self.UtteranceEncoder(target_embeddings,target_lens,enforce_sorted=False) # [batch_size,hidden_size*2]

            # 构造Recognition Network的输入
            recogn_input = torch.cat((articleEncoding,targetEncoding),dim=1) # [batch_size,hidden_size*2+hidden_size]

            # --------------------------------------  Recognition Network  -------------------------------------------
            recogn_mu_logvar = self.RecognitionNetwork(recogn_input) # [batch_size,hidden_size*2]
            # 获取均值和log方差
            # [batch_size,hidden_size],[batch_size,hidden_size]
            recog_mu,recog_logvar = recogn_mu_logvar.split([self.hidden_size,self.hidden_size],dim=1)
            # 获取标准差
            recog_std = torch.exp(0.5 * recog_logvar) # [batch_size,hidden_size]

            # -------------------------------------   Prior Network   -----------------------------------------------
            prior_mu_logvar = self.PriorNetwork(articleEncoding) # [batch_size,hidden_size*2]
            # 获取均值和log方差
            # [batch_size,hidden_size],[batch_size,hidden_size]
            prior_mu,prior_logvar = prior_mu_logvar.split([self.hidden_size,self.hidden_size],dim=1)
            # 获取标准差
            # prior_std = torch.exp(0.5 * prior_logvar) # [batch_size,hidden_size]

            # -------------------------------------   sample from Recognition Network  -------------------------------
            epsilon = torch.rand_like(recog_std)
            # Reparameterization Trick
            sample_Z = recog_mu + epsilon * recog_std # [batch_size,hidden_size]

            # -------------------------------------   KL散度计算：D(Recognition||Prior)  -----------------------------
            KLD = Gaussian_kld(recog_mu,recog_logvar,prior_mu,prior_logvar)
            output = torch.cat((sample_Z, articleEncoding), dim=1)  # [batch_size,hidden_size+hidden_size]
            output = output.unsqueeze(1)  # [batch_size,1,hidden_size*2]
            return output,KLD # 最大化ELBO ，相当于最小化 crossEntropy + KLD
        elif flag == 'Inference':
            # -------------------------------------   Prior Network   -----------------------------------------------
            prior_mu_logvar = self.PriorNetwork(articleEncoding)  # [batch_size,hidden_size*2]
            # 获取均值和log方差
            # [batch_size,hidden_size],[batch_size,hidden_size]
            prior_mu, prior_logvar = prior_mu_logvar.split([self.hidden_size, self.hidden_size], dim=1)
            # 获取标准差
            prior_std = torch.exp(0.5 * prior_logvar)  # [batch_size,hidden_size]

            # -------------------------------------   sample from Prior Network  -------------------------------
            epsilon = torch.rand_like(prior_std)
            # Reparameterization Trick
            sample_Z = prior_mu + epsilon * prior_std  # [batch_size,hidden_size]
            output = torch.cat((sample_Z,articleEncoding),dim=1) # [batch_size,hidden_size+hidden_size]
            output = output.unsqueeze(1) # [batch_size,1,hidden_size*2]
            return output
        else:
            print("模型没有此过程，请设置flag='Train' or flag='Inference' !")
            exit()

if __name__ == '__main__':
    model = Encoder(100,150,None)
    print(model)