# -*- coding: utf-8 -*-
# @Time    : 2020/3/21 17:34
# @Author  : Weiyang
# @File    : NNLMDecoder.py

# ----------------------------------------------------------------------
# Decoder解码器: 局部注意力机制
# a standard feed-forward neural network language model (NNLM)

# 论文5 Extension: Extractive Tuning

# In particular the abstractive model does not have the capacity to find extractive word matches when necessary,
# for example transferring unseen proper noun phrases from the input

# To address this issue we experiment with tuning a very small set of additional features that tradeoff
# the abstractive/extractive tendency of the system. We do this by modifying our scoring function
# to directly estimate the probability of a summary using a log-linear model, as is standard in machine translation
# ----------------------------------------------------------------------

import torch
import torch.nn as nn

class NNLMDecoder(nn.Module):

    def __init__(self,output_size,vocab_size,embedding_size,hidden_size,C_size,alpha1,alpha2,alpha3,alpha4,alpha5):
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
        # 定义ABS+新增抽取特征的各个权重因子
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.alpha5 = alpha5


    def forward(self,encoder_output,predict_sequence,source,source_lens):
        '''
        返回当前时刻的预测结果,[batch_size,output_size],其中最后一维度是各个词对应的概率
        :param encoder_output: [batch_size,hidden_size]
        :param predict_sequence: 先前时刻已预测的单词序列,[batch_size,None]
        :param source: 输入序列,[batch_size,max_time_step]
        :param source_lens: 输入序列的真实长度,[batch_size]
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

        # ABS+
        # The function f is defined to combine
        # the local conditional probability with some additional indicator featrues

        # output: [batch_size,output_size]
        # 存储抽取出的特征 some additional indicator featrues
        # These features correspond to indicators of unigram, bigram, and trigram match with the input as
        # well as reordering of input words
        # 此处有点类似拷贝机制为生成模块的单词增加概率的方式，我们需要判断当前搜索空间中的单词的一些抽取特征：
        # unigram,bigram,trigram是否匹配输入序列,以及重排序输入序列
        # 1. unigram特征： 如果当前待预测的单词(unigram)存在输入序列中，则在output该单词的生成概率(未softmax)处加1;
        # 2. bigram特征：如果当前待预测单词与前面已预测出的单词组合(bigram)存在于输入序列中，即(later_word,current_word),
        #                则在该单词的生成概率处加1；
        # 3. trigram特征：如果当前待预测单词与前面已预测出的最近两个单词的组合(trigram)存在于输入序列中，
        #                 即(later_two_word,later_one_word,current_word),则在该单词的生成概率处加1；
        # 4. reordering of input words特征：如果当前已预测的结果中，上一时刻的预测结果存在于输入序列中，其在输入序列中的位置索引为k，
        # 且当前待预测的单词也存在输入序列中，其在输入序列中的位置索引为j,满足j<k，则在该单词的生成概率处加1

        unigram_feature = torch.zeros((batch_size, self.output_size))  # [batch_size,output_size]
        bigram_feature = torch.zeros((batch_size, self.output_size))  # [batch_size,output_size]
        trigram_feature = torch.zeros((batch_size, self.output_size))  # [batch_size,output_size]
        reordering_feature = torch.zeros((batch_size, self.output_size))  # [batch_size,output_size]

        # predict_sequence: 先前时刻已预测的单词序列,[batch_size,None]
        # source: 输入序列，[batch_size,max_time_step]
        # source_lens: 输入序列的真实长度，排除padding位，[batch_size]

        # 上一时刻的预测结果
        later_one_result = predict_sequence[:, -1]  # [batch_size]
        # 上两个时刻的预测结果
        later_two_result = predict_sequence[:, -2:]  # [batch_size,2]

        # 遍历每条数据
        for i in range(batch_size):
            # 当前输入序列数据的ID列表
            current_input_list = source[i].tolist()  # [1,2,3,...]
            # 当前输入序列数据的字符串形式
            current_input = ''.join([str(id) for id in source[i].tolist()])  # eg: '123456000'
            # 当前输入序列数据对应的预测单词ID
            later_one_word = later_one_result[i].item()  # id
            later_two_word = later_two_result[i].tolist()  # [id1,id2]

            # 遍历当前数据的真实单词序列，排除padding位
            for position in range(source_lens[i]):
                # 当前单词的id
                id = source[i, position].item()

                # unigram feature
                unigram_feature[i, id] = 1

                # bigram feature
                # 判断是否已经预测出一个结果
                if len([later_one_word]) == 1:
                    bigram = str(later_one_word) + str(id)
                    if bigram in current_input:
                        bigram_feature[i, id] = 1

                # trigram feature
                # 判断是否有两个已知的预测结果
                if len(later_two_word) == 2:
                    trigram = ''.join([str(e) for e in later_two_word]) + str(id)
                    if trigram in current_input:
                        trigram_feature[i, id] = 1

                # reordering feature
                # 判断上一时刻的预测结果是否在输入序列中
                if later_one_word in current_input_list:
                    # 上一时刻在输入序列中的索引
                    k = current_input_list.index(later_one_word)
                    # 当前待预测单词在输入序列中的索引
                    j = current_input_list.index(id)
                    # 判断是否满足条件
                    if j < k:
                        reordering_feature[i, id] = 1
        # unigram_feature: [batch_size,output_size]
        # bigram_feature: [batch_size,output_size]
        # trigram_feature: [batch_size,output_size]
        # reordering_feature: [batch_size,output_size]
        # output: [batch_size,output_size]
        combine_output = output * self.alpha1 + unigram_feature * self.alpha2 + bigram_feature * self.alpha3 \
                         + trigram_feature * self.alpha4 + reordering_feature * self.alpha5
        combine_output = torch.softmax(combine_output, dim=1)  # [batch_size,output_size]

        return combine_output


if __name__ == '__main__':
    from BowEncoder import BowEncoder

    encoder = BowEncoder(vocab_size=10,hidden_size=6)

    inputs = torch.tensor([
        [1,2,3,0],
        [5,2,0,0],
        [3,2,1,3]
    ]) # [3,4],batch_size=3,max_time_step=4
    input_lens = torch.tensor([3,2,4]) # [3]

    context = encoder(inputs,input_lens)

    y = torch.tensor([
        [7, 5, 2],
        [6, 2, 1],
        [4, 3, 2]
    ])  # batch_size=3,

    decoder = NNLMDecoder(output_size=10,vocab_size=10,embedding_size=12,hidden_size=6,C_size=2,
                          alpha1=0.6,alpha2=0.1,alpha3=0.1,alpha4=0.1,alpha5=0.1)

    output = decoder(context,y,inputs,input_lens)

    print(output.size())
    print(output)