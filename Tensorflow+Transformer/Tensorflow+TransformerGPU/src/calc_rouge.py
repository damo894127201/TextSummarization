# -*- coding: utf-8 -*-
# @Time    : 2020/4/8 15:02
# @Author  : Weiyang
# @File    : calc_rouge.py

from rouge import Rouge
import logging

def calc_rouge(ref,translation,name='eval',epoch=0):
    '''Calculates rouge score and appends the report to translation
    ref: reference file path
    translation: model output file path
    name: result file name prefix
    epoch: epoch number

    Returns
    translation that the bleu score is appended to'''
    with open(ref,'r',encoding='utf-8') as f_ref,open(translation,'r',encoding='utf-8') as f_trans:
        ref_lst = f_ref.read().split('\n')
        trans_lst = f_trans.read().split('\n')
    # 判断参考译文与模型生成译文的数量是否一致
    assert len(ref_lst) == len(trans_lst),"参考译文和模型生成译文的数量不一致"
    fi = open("../result/"+name+"_bleu.score", "a+",encoding='utf-8')
    Rouges_1 = [] # 存储每条摘要的Rouge-1值
    Rouges_2 = []  # 存储每条摘要的Rouge-2值
    Rouges_L = []  # 存储每条摘要的Rouge-L值
    rouge = Rouge()
    for ref,trans in zip(ref_lst,trans_lst):
        ref = [' '.join(list(ref))] # ['char char char ...']
        trans = [' '.join(list(trans))]
        rouge_score = rouge.get_scores(trans,ref)
        Rouges_1.append(rouge_score[0]['rouge-1']['r'])
        Rouges_2.append(rouge_score[0]['rouge-2']['r'])
        Rouges_L.append(rouge_score[0]['rouge-l']['r'])
    # 求bleu的平均值
    rouge_score_1 = sum(Rouges_1)/float(len(Rouges_1)) * 100
    rouge_score_2 = sum(Rouges_2) / float(len(Rouges_2)) * 100
    rouge_score_L = sum(Rouges_L) / float(len(Rouges_L)) * 100
    fi.write('Epoch '+str(epoch)+' Rouge_score 1 : '+str(rouge_score_1)+'\n')
    fi.write('Epoch '+str(epoch)+' Rouge_score 2 : '+str(rouge_score_2)+'\n')
    fi.write('Epoch '+str(epoch)+' Rouge_score L : '+str(rouge_score_L)+'\n')
    fi.write('\n')
    fi.close()
    logging.info('Epoch: %d  Rouge Score 1: %.2f'% (epoch,rouge_score_1))
    logging.info('Epoch: %d  Rouge Score 2: %.2f' % (epoch, rouge_score_2))
    logging.info('Epoch: %d  Rouge Score L: %.2f' % (epoch, rouge_score_L))