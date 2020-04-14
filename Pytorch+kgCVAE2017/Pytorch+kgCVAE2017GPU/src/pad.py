# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 16:40
# @Author  : Weiyang
# @File    : pad.py

#---------------------------------------------------
# torch.utils.data.DataLoader的collate_fn回调函数
# 用于对一个batch进行处理：
# 1. 填充到等长
# 2. 返回batch各条数据实际长度，并逆序(target随source排序)
# 3. 对应的是MyDataSet数据集

# 由于每个batch中每条数据均包含若干个句子，我们需要对每条数据：
# 1. 每条数据中每个句子扩充到等长，每条数据中的句子个数扩充到等长
# 2. 由于需要padding，padding的对象是这个batch中所有的句子，假设最大句子个数为num，则这个batch中共有num*batch_size个句子
#    需要padding，由于padding时，需要将句子个数按大到小倒排，这会打乱句子所属的数据的类别，即某个句子可能属于句子a,但顺序变了
# 3. 为保证句子的相对顺序，便于将属于同一条数据的句子划分到一组，我们在将所有句子按数据顺序排列在一起时，需要保留其原有
#    的句子索引。具体操作如下：当我们将句子按真实长度从大到小排列时，会输出相应的索引列表descendingOrders，我们保留这个
#    索引列表descendingOrders，当需要恢复原有的句子顺序时，只需执行recoveryOrder操作即可。比如：
#    >>> a = torch.tensor([1,2])
#    >>> descendingOrders = [1,0] # 即值1排序后的索引为1，值2排序后的索引为0
#    >>> a = a[descendingOrders]
#        tensor([2, 1])
#    >>> a = recoveryOrder(a,descendingOrders) # 执行recoveryOrder索引操作，便恢复到原来的顺序
#        tensor([1, 2])
# 4. 我们对num*batch_size个句子执行同样的操作，即Utterance Encoder
#---------------------------------------------------

from torch.nn.utils.rnn import pad_sequence
import torch
import numpy as np

def pad(batch):
    padding_value = 0 # 填充值索引
    source,target = [],[]
    source_article_lens = [] # 存储每篇文章句子的个数,是一个一维列表,[num,...]
    source_sent_lens = [] # 存储每篇文章中每个句子的长度,是一个二维列表,[[len,len,...],...]
    target_lens = [] # 存储每条摘要的长度，是一个一维列表,[len,...]
    for x,y in batch:
        source.append(x) # x:[[id,id,..],[id,..],..]
        target.append(y) # y:[id,id,...]
        source_article_lens.append(len(x))
        source_sent_lens.append([len(sent) for sent in x]) # [[len,len,...],..]
        target_lens.append(len(y))

    # 存储当前batch中，每篇文章最大的句子个数
    max_num_sent = max(source_article_lens)

    # 填充的句子
    padding_sent = [padding_value] # 之所以用一个padding_value，目的在于保持该填充句子长度最小，具体padding操作由pad_sequence执行
    # num*batch_size个句子的集合，其中num表示当前batch中最大的句子个数
    source_sents = [] # 存储句子集合，每个元素是一个列表，代表一个句子
    # num*batch_size个句子实际长度的集合
    source_per_sent_lens = [] # 存储句子的实际长度，每个元素是一个int值，表示句子的长度

    # 将当前batch中所有的句子扩充到等长,将每篇文章的句子个数扩充到一样
    # 遍历每条数据
    for article,sent_lens in zip(source,source_sent_lens):
        # 当前数据的句子个数
        num_sent = len(article)
        # 将真实的句子和填充的句子添加到当前batch的句子集合中
        source_sents.extend(article)
        # 将真实的句子长度和填充的句子长度添加到当前batch的句子长度集合中
        source_per_sent_lens.extend(sent_lens)
        # 判断句子个数是否为max_num_sent
        if num_sent < max_num_sent:
            padding_sents = [padding_sent] * (max_num_sent - num_sent)
            padding_sents_len = [1] * (max_num_sent - num_sent)
            # 将填充句子加入到当前batch的句子集合
            source_sents.extend(padding_sents)
            # 将填充句子的长度加入到当前batch的句子长度的集合
            source_per_sent_lens.extend(padding_sents_len)

    # 对source_sents长度倒排，target保持原有的排序不变
    source_per_sent_lens = np.array(source_per_sent_lens)
    descendingOrders = np.argsort(-source_per_sent_lens)
    source = np.array(source_sents)[descendingOrders].tolist() # source中共有max_num_sent*batch_size个句子
    source_per_sent_lens = torch.tensor(np.sort(source_per_sent_lens)[::-1].tolist()) # 相应的长度从大到小排序
    # 转为torch.tensor
    source = [torch.tensor(seq) for seq in source]
    target = [torch.tensor(seq) for seq in target]
    # 填充,(max_num_sent*batch_size) * max_time_step(句子的最大长度)
    source = pad_sequence(source,batch_first=True,padding_value=padding_value)
    target = pad_sequence(target,batch_first=True,padding_value=padding_value)
    batch_size = len(batch) # 当前batch大小
    # 文章中的句子集合,训练目标集合,文章中每个句子的实际长度集合,训练目标长度集合,batch_size大小,排序索引列表,文章中句子最大个数,每篇文章中真实句子个数
    return source,target,source_per_sent_lens,target_lens,batch_size,descendingOrders,source_article_lens