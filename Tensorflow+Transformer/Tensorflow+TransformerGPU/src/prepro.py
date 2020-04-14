# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 8:24
# @Author  : Weiyang
# @File    : prepro.py
'''
目标:
1. 利用原始source和target数据训练sentencepiece分词器，类型为bpe;或者直接用结巴分词
2. 分词器训练完毕后,同时生成分词器和词包
3. 用分词器将原始的source和target数据分词
4. 统计source和target长度分布
'''

import os
import errno
import sentencepiece as spm
import jieba as jb
from hparams import Hparams
import logging
import configparser
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO)

def prepro(hp):
    'hp: hyperparams. argparse.'
    # 读取默认参数
    cfg = configparser.ConfigParser()
    cfg.read('../config/config.ini',encoding='utf-8')

    logging.info("# Check if raw files exist")
    # 读取各数据集路径
    train_source = cfg.get('data','train_source_sequence')
    train_target = cfg.get('data','train_target_sequence')
    eval_source = cfg.get('data','eval_source_sequence')
    eval_target = cfg.get('data','eval_target_sequence')
    test_source = cfg.get('data','test_source_sequence')
    test_target = cfg.get('data','test_target_sequence')
    # 检查各数据集是否存在,不存在抛出异常
    for f in (train_source, train_target, eval_source, eval_target, test_source, test_target):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    # 判断是否训练分词器,需要足够的数据才能训练较好的分词器
    # 数据不足，则用jieba分词
    if cfg.getint('params','if_train_tokenizer'):
        logging.info("# Train a joint BPE model with sentencepiece")
        # 判断分词结果的存储路径是否存在
        os.makedirs(cfg.get('generate_data','segmented_root_path'), exist_ok=True)

        # 分词器参数配置
        # 用于训练分词器的数据路径,包括source和target的数据
        train_data = '%s,%s,%s,%s,%s,%s'%(
            cfg.get('data','train_source_sequence'),
            cfg.get('data', 'train_target_sequence'),
            cfg.get('data', 'eval_source_sequence'),
            cfg.get('data', 'eval_target_sequence'),
            cfg.get('data', 'test_source_sequence'),
            cfg.get('data', 'test_target_sequence')
        )
        # 分词器的存储路径和name前缀,格式为:path/name.model
        tokenizer_prefix = cfg.get('generate_data','tokenizer_prefix')
        train = '--input={}        \
                 --pad_id=0        \
                 --unk_id=1        \
                 --bos_id=2        \
                 --eos_id=3        \
                 --model_prefix={} \
                 --vocab_size={}   \
                 --model_type=bpe'.format(train_data,tokenizer_prefix,hp.vocab_size)
        # 开始训练分词器
        spm.SentencePieceTrainer.Train(train)
        logging.info('Tokenizer has trained ! ')
        # 加载分词器模型
        logging.info("# Load trained bpe model")
        sp = spm.SentencePieceProcessor()
        sp.Load(cfg.get('generate_data','tokenizer_model'))

    # 开始分词
    logging.info("# Segment")
    def _segment_and_write(finame, foutname,is_tokenizer=True):
        'finame: 原数据路径, foutname: 经过分词后的数据存储路径  is_tokenizer: 是否使用训练的分词器'
        '返回句子长度的列表'
        # 存储source和target序列的长度,即单词个数;
        lengths = []
        # 判断路径目录是否存在,不存在则创建
        foutpath = '/'.join(foutname.split('/')[:-1])
        Path(foutpath).mkdir(parents=True,exist_ok=True)
        with open(finame,'r',encoding='utf-8') as fi,open(foutname, "w",encoding='utf-8') as fout:
            for sent in fi:
                if is_tokenizer:
                    pieces = sp.EncodeAsPieces(sent)
                    lengths.append(len(pieces))
                    fout.write(" ".join(pieces) + "\n")
                else:
                    pieces = jb.cut(sent)
                    lengths.append(len(pieces))
                    fout.write(" ".join(pieces) + "\n")
        return lengths
    length1 = _segment_and_write(train_source, cfg.get('generate_data','train_source_sequence'),is_tokenizer=cfg.getint('params','if_train_tokenizer'))
    length2 = _segment_and_write(train_target, cfg.get('generate_data','train_target_sequence'),is_tokenizer=cfg.getint('params','if_train_tokenizer'))
    length3 = _segment_and_write(eval_source, cfg.get('generate_data','eval_source_sequence'),is_tokenizer=cfg.getint('params','if_train_tokenizer'))
    length4 = _segment_and_write(eval_target, cfg.get('generate_data','eval_target_sequence'),is_tokenizer=cfg.getint('params','if_train_tokenizer'))
    length5 = _segment_and_write(test_source, cfg.get('generate_data','test_source_sequence'),is_tokenizer=cfg.getint('params','if_train_tokenizer'))
    length6 = _segment_and_write(test_target, cfg.get('generate_data', 'test_target_sequence'),is_tokenizer=cfg.getint('params','if_train_tokenizer'))
    # source长度列表
    source_len = length1 + length3 + length5
    # target长度列表
    target_len = length2 + length4 + length6
    logging.info("source sequence length distribution")
    logging.info("source sequence max length: %d"%max(source_len))
    logging.info("source sequence min length: %d" % min(source_len))
    logging.info("source sequence average length: %d" % (sum(source_len)/float(len(source_len))))
    logging.info("source sequence most 100 length:")
    for i in Counter(source_len).most_common(100):
        logging.info('length: %d , number : %d'%(i))

    logging.info("target sequence length distribution")
    logging.info("target sequence max length: %d" % max(target_len))
    logging.info("target sequence min length: %d" % min(target_len))
    logging.info("target sequence average length: %d" % (sum(target_len) / float(len(target_len))))
    logging.info("source sequence most 100 length:")
    for i in Counter(source_len).most_common(100):
        logging.info('length: %d , number : %d' % (i))
    # 将source和target长度数据输出,用于进一步分析
    with open(cfg.get('generate_data','source_sequence_length'),'w',encoding='utf-8') as fi:
        fi.write(','.join([str(i) for i in source_len]))
    with open(cfg.get('generate_data','target_sequence_length'),'w',encoding='utf-8') as fi:
        fi.write(','.join([str(i) for i in source_len]))

if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")