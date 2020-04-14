# -*- coding: utf-8 -*-
# @Time    : 2020/3/10 9:01
# @Author  : Weiyang
# @File    : Inference.py

#-------------------------------
# 基于训练好的模型，批量预测
#-------------------------------

import torch
from torch.utils.data import DataLoader
from MyDataSet import MyDataSet
from Tokenizer import Tokenizer
from pad import pad
from Config import Config
from Seq2Seq import Seq2Seq

if __name__ == '__main__':
    source_path = '../data/test/source.txt'
    target_path = '../data/test/target.txt'
    vocab_path = '../data/vocab.txt'
    model_path = '../model/model.pth'
    tokenizer = Tokenizer(vocab_path)
    config = Config()
    fr = open('../result/test.txt','w',encoding='utf-8-sig') # 存储预测结果

    loader = DataLoader(dataset=MyDataSet(source_path, target_path, tokenizer), batch_size=config.batch_size, shuffle=True,
                        num_workers=2,collate_fn=pad,drop_last=False) # 最后一个batch数据集不丢弃

    if not torch.cuda.is_available():
        print('No cuda is available!')
        exit()
    device = torch.device('cuda:0')
    model = Seq2Seq(config)
    model.to(device)
    # 加载模型
    checkpoint = torch.load(model_path,map_location=device)
    model.load_state_dict(checkpoint['model'])

    for iter, (batch_x, batch_y, batch_source_lens,batch_target_lens) in enumerate(loader):
        batch_x = batch_x.cuda()
        batch_source_lens = torch.as_tensor(batch_source_lens).cuda()
        # 预测结果和相应时刻的注意力权重
        results, attention_weights = model.BatchSample(batch_x,batch_source_lens)
        for i in range(len(results)):
            words = tokenizer.convert_ids_to_tokens(results[i])
            if i % 100 == 0:
                print(''.join(words))
            fr.write(''.join(words))
            fr.write('\n')
    fr.close()