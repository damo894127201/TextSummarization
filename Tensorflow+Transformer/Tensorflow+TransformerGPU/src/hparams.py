# -*- coding: utf-8 -*-
# @Time    : 2019/5/15 8:24
# @Author  : Weiyang
# @File    : hparams.py
'''
参数配置：
默认是加载配置文件中的设置，也可以通过命令行给选项参数设置值的方式执行程序
'''
import argparse
import configparser

class Hparams:
    # 参数解析器
    parser = argparse.ArgumentParser()
    # 读取默认参数
    cfg = configparser.ConfigParser()
    cfg.read('../config/config.ini',encoding='utf-8')

    # prepro阶段
    # 词包大小,由用户指定sentencepiece生成的词包大小
    parser.add_argument('--vocab_size', default=cfg.getint('params','vocabulary_size'), type=int)

    # train阶段
    ## 训练集和评估集
    # 训练集: 编码器端输入--source sequence
    parser.add_argument('--train_source', default=cfg.get('generate_data','train_source_sequence'),
                             help="训练集: 编码器端输入,经过分词")
    # 训练集: 解码器端输出和解码器端输入--target sequence
    parser.add_argument('--train_target', default=cfg.get('generate_data','train_target_sequence'),
                             help="训练集: 解码器端输入和输出,经过分词")
    # 评估集: 编码器端输入--source sequence
    parser.add_argument('--eval_source', default=cfg.get('generate_data','eval_source_sequence'),
                             help="评估集: 编码器端输入,经过分词")
    # 评估集: 解码器端输出和解码器端输入--target sequence
    parser.add_argument('--eval_target', default=cfg.get('generate_data','eval_target_sequence'),
                             help="评估集: 解码器端输入和输出,经过分词")

    ## 加载prepro阶段生成的词包
    parser.add_argument('--vocab', default=cfg.get('generate_data','vocabulary'),
                        help="词包路径")

    # 模型参数
    parser.add_argument('--batch_size', default=cfg.getint('params','batch_size'), type=int)
    parser.add_argument('--eval_batch_size', default=cfg.getint('params','eval_batch_size'), type=int)

    parser.add_argument('--lr', default=cfg.getfloat('params','lr'), type=float, help="learning rate")
    parser.add_argument('--warmup_steps', default=cfg.getint('params','warmup_steps'), type=int)
    # 日志路径
    parser.add_argument('--logdir', default=cfg.get('params','logdir'), help="log directory")
    # 模型路径
    parser.add_argument('--modeldir', default=cfg.get('params', 'modeldir'), help="model directory")
    parser.add_argument('--num_epochs', default=cfg.getint('params','num_epochs'), type=int)
    # 评估集的预测结果
    parser.add_argument('--eval_result', default=cfg.get('params','eval_result'), help="evaluation dir")

    # model
    # 编码器和解码器的隐藏层维度: 注意力的维度
    parser.add_argument('--d_model', default=cfg.getint('params','attention_dimension'), type=int,
                        help="attention dimension and word2vec dimension")
    # 前馈神经网络隐藏层维度
    parser.add_argument('--d_ff', default=cfg.getint('params','feedforward_hidden_dimension'), type=int,
                        help="hidden dimension of feedforward layer")
    # 编码器端和解码器端各自堆叠的编码器和解码器的个数,paper中默认是6个
    parser.add_argument('--num_blocks', default=cfg.getint('params','num_blocks'), type=int,
                        help="number of encoder/decoder blocks")
    # 多头注意力的个数,paper默认是8个
    parser.add_argument('--num_heads', default=cfg.getint('params','num_heads'), type=int,
                        help="number of attention heads")
    # 编码器端输入序列的最大长度--source sequence
    parser.add_argument('--maxlen_source', default=cfg.getint('params','maxlen_source'), type=int,
                        help="maximum length of a source sequence")
    # 解码器端输入和输出序列的最大长度--target sequence
    parser.add_argument('--maxlen_target', default=cfg.getint('params','maxlen_target'), type=int,
                        help="maximum length of a target sequence")
    # 设置保存模型的个数
    parser.add_argument('--max_to_keep',default=cfg.get('params','max_to_keep'),type=int,
                        help="maximum number of trained model saved")

    # 训练损失存储路径
    parser.add_argument('--loss_path', default=cfg.get('generate_data', 'loss_path'), type=str,
                        help="train loss save to path")

    # dropout比率
    parser.add_argument('--dropout_rate', default=cfg.getfloat('params','dropout_rate'), type=float)
    # 对解码器端输出的target sequence的类别标签进行平滑,即假设某个单词的类别标签是[0,1,0,0,0]
    # 平滑后变为[0.1,0.8,0.05,0.04,0.01]
    parser.add_argument('--smoothing', default=cfg.getfloat('params','smoothing'), type=float,
                        help="label smoothing rate")

    # test阶段
    # 测试集: 编码器端输入--source sequence
    parser.add_argument('--test_source', default=cfg.get('generate_data','test_source_sequence'),
                        help="测试集: 编码器端输入,经过分词")
    parser.add_argument('--test_target', default=cfg.get('generate_data','test_target_sequence'),
                        help="测试集: 解码器端输入和输出,经过分词")
    parser.add_argument('--test_batch_size', default=cfg.getint('params','test_batch_size'), type=int)
    # 测试结果存储路径
    parser.add_argument('--test_result', default=cfg.get('params','test_result'), help="test result dir")