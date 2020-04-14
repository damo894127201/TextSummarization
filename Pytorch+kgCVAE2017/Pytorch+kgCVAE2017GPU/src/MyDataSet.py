# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 8:43
# @Author  : Weiyang
# @File    : MyDataSet.py

#----------------------------------------------------
# 自定义数据集，继承torch.utils.data.Dataset
# source_path: 每行一篇文档，每篇文档由若干个句子构成，句子之间以制表符分割
# target_path: 每行为一个摘要句子或标题
#----------------------------------------------------

from torch.utils.data import Dataset

class MyDataSet(Dataset):

    def __init__(self,source_path,target_path,tokenizer):
        self.tokenizer = tokenizer
        self.sourceData,self.targetData = self.load(source_path,target_path)

    def load(self,source_path,target_path):
        '''load source data'''
        with open(source_path, 'r', encoding='utf-8') as fs, open(target_path, 'r', encoding='utf-8') as ft:
            sourceData = []
            targetData = []
            EOS_id = self.tokenizer.convert_token_to_id('_EOS')
            for source, target in zip(fs, ft):
                # source,每行是一篇文章，每篇文章由若干个句子构成，句子间以制表符分割，句子内每个字符以空格分割
                sentences = [self.tokenizer.convert_tokens_to_ids(sent.split(' ')) for sent in source.strip().split('\t')]
                sourceData.append(sentences)
                #sourceData.append(self.tokenizer.convert_tokens_to_ids(source.strip().split(' ')))
                # target 末尾加上EOS结束符
                targetData.append(self.tokenizer.convert_tokens_to_ids(target.strip().split(' ')+[EOS_id]))
        return sourceData, targetData

    def __len__(self):
        '''要求：返回数据集大小'''
        return len(self.sourceData)

    def __getitem__(self,index):
        '''要求：传入index后，可按index单例或切片返回'''
        return self.sourceData[index],self.targetData[index]