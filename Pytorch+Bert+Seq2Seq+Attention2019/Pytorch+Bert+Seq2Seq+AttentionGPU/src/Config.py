# -*- coding: utf-8 -*-
# @Time    : 2020/3/11 10:00
# @Author  : Weiyang
# @File    : Config.py

#-------------------------------------------------
# 超参
#-------------------------------------------------

class Config(object):
    vocab_size = 21128  #输入空间：词包大小还得加上这些特殊符号："_PAD": 0, "_GO": 1, "_EOS": 2, "_UNK": 3
    embedding_size = 768 # bert词向量的维度
    hidden_size = 128
    output_size = 21128 # 输出空间,bert词包大小
    max_decoding_steps = 30
    batch_size = 8
    learning_rate = 0.0001
    num_Epochs = 20
    TeacherForcingRate = 0.8
    max_norm = 5 # 相应范数的最大值，梯度裁剪
    norm_type = 2 # 范数的类型，1表示范数L1，2表示范数L2
    use_drop_out = True # 是否使用Dropout层，此处该层设置在Decoder模块的Embedding层
                        # Dropout可放置在：可见层(输入层)、相邻隐藏层之间、隐藏层和输出层之间
    keep_prob = 0.5 # drop_out的比例通常在0.2-0.5之间，具体需要尝试