# -*- coding: utf-8 -*-
# @Time    : 2020/3/12 9:30
# @Author  : Weiyang
# @File    : sequence_mask_loss.py

#---------------------------------------------------------------------------------
# 处理sequence中填充位PAD的损失
# 将预测结果中，所有的PAD位的概率分布的预测结果，全部置为[1,1,..]形式
# 这样在计算交叉熵损失时，loss为0
# 假如
# 真实label为: [1,0,0]
# 预测label为: [0.7,0.2,0.1]
# 交叉熵损失为:
#              crossEntropyLoss = -(1 * log0.7 + 0 * log0.2 + 0 * log0.1)
# 为了使交叉熵损失为0，我们将预测label变为:[1,1,1]
# 这样交叉熵损失为：
#              crossEntropyLoss = -(1 * log1 + 0 * log1 + 0 * log1) = 0
#---------------------------------------------------------------------------------

import torch
from torch.autograd import Variable


def sequence_mask_loss(batch_outputs,batch_lens,output_size):
    '''
    :param batch_outputs: 模型预测的概率分布结果，shape为[max_time_step,batch_size,vocab_size]
    :param batch_lens: 当前batch中，各条数据实际长度,shape为[batch_size,]
    :param output_size: 预测结果的搜索空间，即分类的类别数
    :return: 将pad位的预测概率分布结果全部置为0
    '''
    batch_outputs = batch_outputs.permute(1,0,2) # [batch_size,max_time_step,vocab_size]
    batch_size = len(batch_lens)
    max_time_step = max(batch_lens)
    mask_outputs = []
    # 遍历每条数据
    for i in range(batch_size):
        outputs =  batch_outputs[i]  # 当前数据
        true_position_predict = outputs[:batch_lens[i]].tolist()
        pad_tokens = [[1]*output_size]*(max_time_step-batch_lens[i])
        mask_outputs.append(true_position_predict+pad_tokens)
    mask_tensor = Variable(torch.Tensor(mask_outputs),requires_grad=True) # 需用Variable包裹，并且requires_grad=True，
                                                                          # 表示需要对此求梯度，以使其参与反向传播
    # shape形状需转为原来的形式[max_time_step,batch_size,vocab_size]
    return mask_tensor.permute(1,0,2)

if __name__ == '__main__':
    # [max_time_step,batch_size,vocab_size]
    batch_outputs = torch.zeros((5,3,6)) # batch_size为3，max_time_step为5，output_size为6
    batch_lens = [5,4,3] # 每条数据去除pad位后的真实长度
    output_size = 6
    print(batch_outputs.size())
    batch_outputs = sequence_mask_loss(batch_outputs,batch_lens,output_size)
    print(batch_outputs.size())
    print(batch_outputs)