# -*- coding: utf-8 -*-
# @Time    : 2020/3/14 10:36
# @Author  : Weiyang
# @File    : LoadBert.py

##########################
# 测试Pytorch版本的Bert
#########################

import torch
from transformers.modeling_bert import BertModel,BertConfig
from transformers.tokenization_bert import BertTokenizer

bert_path = '../../../../chinese_wwm_ext_pytorch/'

config = BertConfig.from_pretrained(bert_path + 'bert_config.json')
model = BertModel.from_pretrained(bert_path + 'pytorch_model.bin',config=config)
tokenizer = BertTokenizer.from_pretrained(bert_path + 'vocab.txt')

#input_ids = tokenizer.encode(['Hello','my','dog','is','cute'], add_special_tokens=True) # True,表示在句子前后加[CLS]和[SEP]符
input_ids = tokenizer.encode('Hello my dog is cute', add_special_tokens=True) # True,表示在句子前后加[CLS]和[SEP]符,False则不加
# 如果要获取句向量，则必须为True，这样获取的句向量才有意义. 因为BERT训练时就是这么训的. 建议直接输入空格分隔的字符串,而不是上述字符串列表
# 因为tokenizer直接输入字符串会调用tokenize()方法, 处理字符串中每个单词，大小改为小写，长单词分隔为子词
# 而直接输入字符串列表的话，就是个简单的查词表映射单词ID, 并不对字符串列表中的字符串做任何处理，这可能导致有的单词
# 只是大写了，却在词表中不存在，而用[UNK].例如，字符串'Hello my dog is cute'会被tokenize()处理为：
# ['[CLS]', 'hello', 'my', 'dog', 'is', 'cut', '##e', '[SEP]']
# 而不是 ['Hello','my','dog','is','cute']
# 在Seq2Seq+bert中，并不加特殊符号[CLS]和[SEP],
# 因为[SEP]是分句符，而模型是直接输入整个文档，并不需要分句


words = tokenizer.convert_ids_to_tokens(input_ids)
inputs = torch.tensor(input_ids).unsqueeze(0)
outputs = model(inputs)
last_hidden_states = outputs[0]
words_embeddings = last_hidden_states[:,1:-1,:] # [batch_size,max_time_step,hidden_size]
                                                # 表示去掉第一个符号[CLS]和最后一个符号[SEP]的所有词的向量
sentence_vector = outputs[1]


'''
print(input_ids)
print(words)
print(inputs)
print('*'*20)
print(outputs)
print(last_hidden_states)
print(last_hidden_states.size())
print()
print(words_embeddings)
print(words_embeddings.size())
print(sentence_vector)
'''

#print(tokenizer.convert_tokens_to_ids(['[UNK]','[unused1]'])+[555])

