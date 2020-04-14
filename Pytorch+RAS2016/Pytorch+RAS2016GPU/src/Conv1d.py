# -*- coding: utf-8 -*-
# @Time    : 2020/3/17 20:54
# @Author  : Weiyang
# @File    : Conv1d.py

# --------------------------------------------------------------------------------------
# 卷积操作，论文公式(2)的实现，从公式可知，当前位置的单词位于卷积感受野中所有单词的中心
# 由论文可知，事实上有embedding_size个卷积核，且每个卷积核对于一个单词的所有embedding维度取值是一样的
# 故而无法对其直接调用torch.nn.Conv1d操作
# --------------------------------------------------------------------------------------

import torch

def Conv1d(input_embeddings,input_lens,B_matrix,q_size):
    '''
    返回卷积后的结果,且原序列中的padding位卷积的结果为0或0向量
    :param input_embeddings: [max_time_step,batch_size,embedding_size]
    :param input_lens: [batch_size]
    :param B_matrix: [num_filters,q_size,embedding_size]
    :param q_size: 卷积核的height
    :return: [batch_size,max_time_step,embedding_size]
    '''
    input_embeddings = input_embeddings.permute(1,0,2) # [batch_size,max_time_step,embedding_size]
    batch_size,max_time_step,embedding_size = input_embeddings.size(0),input_embeddings.size(1),input_embeddings.size(2)
    num_filters = B_matrix.size(0) # 卷积核的数量，同时也是卷积结果的维度
    outputs = torch.zeros((batch_size,max_time_step,num_filters)).cuda() # 卷积结果
    # 判断q_size是奇数还是偶数，以判断卷积时，当前时刻在局部感受野中的相对位置；若为奇数，则当前时刻处于中间，即(q_size-1)/2+1处
    # 若为偶数，则当前时刻处于q_size/2处
    # pre_spaces : 开始边界距当前时刻的距离,这里指相差几个单词的位置
    # end_spaces : 结束边界距当前时刻的距离,这里指相差几个单词的位置
    if q_size % 2 == 0:
        pre_spaces,end_spaces = int(q_size/2 - 1),int(q_size/2)
    else:
        pre_spaces,end_spaces = int((q_size-1)/2),int((q_size-1)/2)
    # 用0填充初始时刻之前和最后时刻之后的位置，便于卷积操作
    # [batch_size,pre_spaces,embedding_size]
    pre_start_position = torch.zeros((batch_size,pre_spaces,embedding_size),dtype=torch.float32).cuda()
    # [batch_size,end_spaces,embedding_size]
    end_position = torch.zeros((batch_size,end_spaces,embedding_size),dtype=torch.float32).cuda()
    # 当前输入前后填充padding后的结果，[batch_size,pre_spaces+max_time_step+end_spaces,embedding_size]
    convEmbeddings = torch.cat((pre_start_position,input_embeddings,end_position),dim=1)
    # 输入序列填充后的实际长度 padding_len = pre_spaces + max_time_step + end_spaces
    # 遍历每个时刻
    i = 0
    for current_position in range(pre_spaces,pre_spaces + max_time_step):
        # 上边界位置索引,下边界位置索引
        top_border,down_border = current_position - pre_spaces,current_position+end_spaces
        # 获取当前局部感受野
        ReceptiveField = convEmbeddings[:,top_border:down_border+1,:] # [batch_size,q_size,embedding_size]
        # 遍历每一个卷积核,获取当前感受野的卷积结果
        ConvResults = torch.zeros((batch_size,num_filters)).cuda() # [batch_size,num_filters]
        for j in range(num_filters):
            multiple_B = B_matrix[j].repeat(batch_size,1,1) # 复制当前卷积核便于一次性计算batch中所有数据,[batch_size,q_size,embedding_size]
                                                         # 改变形状为[batch_size,q_size,embedding_size]，便于它能与ReceptiveField的相乘
            convResult = ReceptiveField * multiple_B # [batch_size,q_size,embedding_size]
            convResult = torch.sum(torch.sum(convResult,dim=2),dim=1) # 累加当前数据内各个维度的值，而得到卷积值,[batch_size]
            ConvResults[:,j] += convResult
        # 存储当前位置的卷积结果
        outputs[:,i] += ConvResults
        i += 1
    # mask掉那些在输入序列中为保持batch长度一致而填充的padding位，不是为卷积而填充的padding位
    mask_vectors = torch.zeros((batch_size,max_time_step,num_filters)).cuda()# mask向量
    # 遍历每条数据
    for i in range(batch_size):
        # 遍历每条数据每个实际的值，出去padding位
        for j in range(input_lens[i]):
            mask_vectors[i,j] += torch.ones((num_filters)).cuda()
    # mask掉那些填充位
    outputs = outputs * mask_vectors

    return outputs # [batch_size,max_time_step,num_filters]

if __name__ == '__main__':
    input_embeddings = torch.tensor([
        [[1,2],[3,4],[0,0]],
        [[5,6],[0,0],[0,0]],
        [[7,8],[9,10],[11,12]]
    ],dtype=torch.float32) # torch.Size([3, 3, 2]),embedding_size=2
    input_lens = torch.tensor([2,1,3])
    embedding_size = 2
    q_size = 3
    num_filters = 2
    B_matrix = torch.Tensor(num_filters,q_size,embedding_size).uniform_(-1,1) # torch.Size([2, 2, 2])
    outputs = Conv1d(input_embeddings,input_lens,B_matrix,q_size) # [batch_size,max_time_step,embedding_size]
    print(outputs.size())
    print(outputs)