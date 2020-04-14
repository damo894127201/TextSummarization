# -*- coding: utf-8 -*-
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''
import tensorflow as tf

from model import Transformer
from tqdm import tqdm
from data_load import get_batch
from utils import save_hparams, save_variable_specs, get_hypotheses#, calc_bleu
from calc_rouge import calc_rouge
import os
from hparams import Hparams
import math
import logging

logging.basicConfig(level=logging.INFO)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


logging.info("# hparams")
hparams = Hparams()
parser = hparams.parser
hp = parser.parse_args()
save_hparams(hp, hp.modeldir)

logging.info("# Prepare train/eval batches")
train_batches, num_train_batches, num_train_samples = get_batch(hp.train_source, hp.train_target,
                                             hp.maxlen_source, hp.maxlen_target,
                                             hp.vocab, hp.batch_size,
                                             shuffle=True)
eval_batches, num_eval_batches, num_eval_samples = get_batch(hp.eval_source, hp.eval_target,
                                             hp.maxlen_source, hp.maxlen_target,
                                             hp.vocab, hp.eval_batch_size,
                                             shuffle=False)

# create a iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_batches.output_types, train_batches.output_shapes)
xs, ys = iter.get_next()

train_init_op = iter.make_initializer(train_batches)
eval_init_op = iter.make_initializer(eval_batches)

logging.info("# Load model")
m = Transformer(hp)
loss, train_op, global_step, train_summaries = m.train(xs, ys)
y_hat, eval_summaries,refs = m.eval(xs, ys)
# y_hat = m.infer(xs, ys)

logging.info("# Session")
saver = tf.train.Saver(max_to_keep=hp.max_to_keep)
with tf.Session() as sess:
    ckpt = tf.train.latest_checkpoint(hp.modeldir)
    if ckpt is None:
        logging.info("Initializing from scratch")
        sess.run(tf.global_variables_initializer())
        save_variable_specs(os.path.join(hp.modeldir, "specs"))
    else:
        saver.restore(sess, ckpt)
    # 存储日志信息
    summary_writer = tf.summary.FileWriter(hp.logdir, sess.graph)

    sess.run(train_init_op)
    total_steps = hp.num_epochs * num_train_batches
    _gs = sess.run(global_step)

    num_train_batches = 20000  # 每20000个step评估一次

    for i in tqdm(range(_gs, total_steps+1)):
        _, _gs, _summary = sess.run([train_op, global_step, train_summaries])
        epoch = math.ceil(_gs / num_train_batches)
        summary_writer.add_summary(_summary, _gs)

        if _gs and _gs % num_train_batches == 0:
            logging.info("epoch {} is done".format(epoch))
            _loss = sess.run(loss) # train loss

            # 将损失记录到本地
            temp = hp.loss_path.split('/')
            temp_path = '/'.join(temp[:-1])
            if not os.path.exists(temp_path): os.makedirs(temp_path)
            with open(hp.loss_path, 'a+', encoding='utf-8') as fi:
                fi.write(str(epoch) + ':' + str(_loss)+'\n')

            logging.info("# Starting evaluation")
            _, _eval_summaries = sess.run([eval_init_op, eval_summaries])
            summary_writer.add_summary(_eval_summaries, _gs)

            logging.info("# get hypotheses")
            hypotheses,refs_result = get_hypotheses(num_eval_batches, num_eval_samples, sess, y_hat,refs, m.idx2token)

            # 将原始的结果写到本地
            logging.info("write references")
            model_output = "refs_ The%02dL Epoch loss is %.2f" % (epoch, _loss)
            if not os.path.exists(hp.eval_result): os.makedirs(hp.eval_result)
            ref_path = os.path.join(hp.eval_result, model_output)
            with open(ref_path, 'w', encoding='utf-8') as fout:
                _refs = []
                for r in refs_result:
                    words = r.decode('utf-8').split()
                    s = [word.replace("▁", " ") for word in words]  # remove bpe symbols
                    sent = ''.join(s)
                    _refs.append(sent.strip())
                fout.write("\n".join(_refs))

            logging.info("# write results")
            model_output = "trans_ The%02dL Epoch loss is %.2f" % (epoch, _loss)
            if not os.path.exists(hp.eval_result): os.makedirs(hp.eval_result)
            translation_path = os.path.join(hp.eval_result, model_output)
            with open(translation_path, 'w',encoding='utf-8') as fout:
                fout.write("\n".join(hypotheses))

            logging.info("# calc bleu score and write it to disk")
            #calc_bleu(ref_path, translation_path,'eval',epoch)
            calc_rouge(ref_path, translation_path, 'eval', epoch)

            logging.info("# save models")
            model_output = "The%02dL Epoch loss is %.2f" % (epoch, _loss)
            ckpt_name = os.path.join(hp.modeldir, model_output)
            saver.save(sess, ckpt_name, global_step=_gs)
            logging.info("after training of {} epochs, {} has been saved.".format(epoch, ckpt_name))

            logging.info("# fall back to train mode")
            sess.run(train_init_op)
    summary_writer.close()


logging.info("Done")
