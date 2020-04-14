# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 10:00
# @Author  : Weiyang
# @File    : Config.py

#-------------------------------------------------
# 超参
#-------------------------------------------------

class Config(object):
    vocab_size = 5004  #输入空间：词包大小还得加上这些特殊符号："_PAD": 0, "_GO": 1, "_EOS": 2, "_UNK": 3
    embedding_size = 128
    hidden_size = 128
    output_size = 5004 # 输出空间
    max_decoding_steps = 30
    batch_size = 8
    learning_rate = 0.0001
    num_Epochs = 20
    TeacherForcingRate = 0.8
    max_norm = 5 # 相应范数的最大值，梯度裁剪
    norm_type = 2 # 范数的类型，1表示范数L1，2表示范数L2
    C_size = 5 # 在解码时，利用已知解码序列的单词个数，比如在解码当前时刻时，要基于先前的C_size个单词解码，即马尔可夫假设的范围
    alpha1 = 0.6  # 传统由Decoder生成的概率值(未softmax)的权重，具体查看论文 5 .Extension: Extractive Tuning
    alpha2 = 0.1  # unigram_feature的权重
    alpha3 = 0.1  # bigram_feature的权重
    alpha4 = 0.1  # trigram_feature的权重
    alpha5 = 0.1  # reordering_feature的权重