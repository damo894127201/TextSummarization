# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 10:11
# @Author  : Weiyang
# @File    : Tokenizer.py

#---------------------------------------------------------
# 用于将单词映射到ID，ID映射到单词
#---------------------------------------------------------

class Tokenizer(object):

    def __init__(self,vocab_path):
        self.w2i,self.i2w = self.load(vocab_path)

    def load(self,vocab_path):
        with open(vocab_path,'r',encoding='utf-8') as fi:
            w2i = {"_PAD": 0, "_GO": 1, "_EOS": 2, "_UNK": 3}
            i2w = {0: "_PAD", 1: "_GO", 2: "_EOS", 3: "_UNK"}
            for line in fi:
                word = line.strip().split('\t')[0]
                if word not in w2i:
                    id = len(w2i)
                    w2i[word] = id
                    i2w[id] = word
            return w2i,i2w

    def convert_tokens_to_ids(self,tokens):
        '''将tokens序列转为id序列'''
        output = []
        unk_id = self.w2i['_UNK']
        for token in tokens:
            output.append(self.w2i.get(token,unk_id))
        return output

    def convert_token_to_id(self,token):
        unk_id = self.w2i['_UNK']
        return self.w2i.get(token,unk_id)

    def convert_ids_to_tokens(self,ids):
        '''将ids转为tokens'''
        output = []
        for id in ids:
            output.append(self.i2w[id])
        return output

    def convert_id_to_token(self,id):
        return self.i2w[id]

if __name__ == '__main__':
    vocab_path = '../data/vocab.txt'
    model_path = '../model/model.pth'
    tokenizer = Tokenizer(vocab_path)
    print(tokenizer.i2w)
    print(tokenizer.convert_id_to_token(5))