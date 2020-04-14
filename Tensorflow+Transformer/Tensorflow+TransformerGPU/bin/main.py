# -*- coding: utf-8 -*-
# @Time    : 2019/5/19 19:02
# @Author  : Weiyang
# @File    : main.py
'''
用于单条数据预测
'''

import tensorflow as tf
from src.model import Transformer
from src.hparams import Hparams
from src.utils import postprocess, load_hparams
import sentencepiece as spm
import jieba as jb
import configparser
import logging


def main():
    # 读取默认参数
    cfg = configparser.ConfigParser()
    cfg.read('../config/config.ini',encoding='utf-8')
    # 判断是否加载分词器模型
    flag = cfg.getint('params','if_train_tokenizer')
    if flag:
        sp = spm.SentencePieceProcessor()
        sp.Load(cfg.get('generate_data','tokenizer_model'))
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    load_hparams(hp, hp.modeldir)
    # 创建模型结构
    m = Transformer(hp)
    # 定义几个占位符来表示模型输入
    x = tf.placeholder(shape=(1,None), dtype=tf.int32)
    x_seqlens = tf.placeholder(shape=(1,None), dtype=tf.int32)
    sents = tf.placeholder(shape=(1,None), dtype=tf.string)
    # 构造模型输入
    xs, ys = (x, x_seqlens, sents), (x, x, x_seqlens, sents)  # ys的内容是无用的,只是用作占位符
    # 输入模型
    y_hat, _ ,_= m.eval(xs, ys)

    with tf.Session() as sess:
        # 加载模型
        ckpt = tf.train.latest_checkpoint(hp.modeldir)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)
        while True:
            # 获取用户输入
            source = input("Me: ")
            if source == 'quit':
                break
            # 分词
            if flag:
                pieces = sp.EncodeAsPieces(source)
            else:
                pieces = jb.cut(source)
            # 截取指定长度内的内容
            if len(pieces) + 1 > hp.maxlen_source:
                logging.info('输入长度过大，我们只截取指定长度的内容')
                pieces = pieces[:hp.maxlen_source-1]
            # 将单词序列转为ID序列
            unk_id = m.token2idx['<unk>']
            temp = [m.token2idx.get(word,unk_id) for word in pieces]
            # 在ID序列末尾加上</s>表示结束符EOS的标记ID
            pieces = temp + [m.token2idx['</s>']]

            # 执行预测
            Y_ID = sess.run(y_hat,feed_dict={x:[pieces],x_seqlens:[[len(pieces)]],sents:[[' ']]})
            # 将预测的单词ID转为字符序列，并截取结束符</s>前面的内容作为输出
            hypotheses = postprocess(Y_ID, m.idx2token)
            print("AI: "+ hypotheses[0])

if __name__ == '__main__':
    main()