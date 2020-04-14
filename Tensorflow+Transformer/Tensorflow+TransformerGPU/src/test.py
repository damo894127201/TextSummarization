# -*- coding: utf-8 -*-
'''
用于批量数据预测
'''
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer

Inference
'''

import os

import tensorflow as tf

from data_load import get_batch
from model import Transformer
from hparams import Hparams
from utils import get_hypotheses, calc_bleu,load_hparams
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
load_hparams(hp, hp.modeldir)

logging.info("# Prepare test batches")
test_batches, num_test_batches, num_test_samples  = get_batch(hp.test_source, hp.test_target,
                                              100000, 100000,
                                              hp.vocab, hp.test_batch_size,
                                              shuffle=False)
iter = tf.data.Iterator.from_structure(test_batches.output_types, test_batches.output_shapes)
xs, ys = iter.get_next()

test_init_op = iter.make_initializer(test_batches)

logging.info("# Load model")
m = Transformer(hp)
y_hat, _ ,refs= m.eval(xs, ys)

logging.info("# Session")
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.modeldir)
    saver = tf.train.Saver()

    saver.restore(sess, ckpt)

    sess.run(test_init_op)

    logging.info("# get hypotheses")
    hypotheses,refs_result = get_hypotheses(num_test_batches, num_test_samples, sess, y_hat,refs, m.idx2token)

    # 将原始的结果写到本地
    logging.info("write references")
    result_output = "refs"
    if not os.path.exists(hp.test_result): os.makedirs(hp.test_result)
    ref_path = os.path.join(hp.test_result, result_output)
    with open(ref_path, 'w', encoding='utf-8') as fout:
        _refs = []
        for r in refs_result:
            words = r.decode('utf-8').split()
            s = [word.replace("▁", " ") for word in words]  # remove bpe symbols
            sent = ''.join(s)
            _refs.append(sent.strip())
        fout.write("\n".join(_refs))

    logging.info("# write results")
    result_output = "trans"
    if not os.path.exists(hp.test_result): os.makedirs(hp.test_result)
    translation = os.path.join(hp.test_result, result_output)
    with open(translation, 'w',encoding='utf-8') as fout:
        fout.write("\n".join(hypotheses))

    logging.info("# calc bleu score and write it to disk")
    calc_bleu(ref_path, translation,'test')