# -*- coding: utf-8 -*-
# @Time    : 2020/4/5 10:07
# @Author  : Weiyang
# @File    : convert_to_RougePattern.py

# -----------------------------------------------------------
# 将模型预测摘要的id序列结果，和 真实摘要转为ROUGE包指定的格式
# ------------------------------------------------------------

def convert_to_RougePattern(predict,true):
    '''
    :param predict: [[id,id,...],[id,..],...]
    :param true: tensor([[id,id,...],[id,..],...])
    :return: 返回预测摘要和真实摘要符合ROUGE的格式,['id id id ','id id ...',...]
             需要去除填充位PAD(id=0)和终止符_EOS(id=2),如果当前摘要为空，则用一个填充位PAD替代
    '''
    # predictPattern: ['id id id id','id id id',....],需要去除填充位PAD(id=0)和终止符_EOS(id=2)
    predictPattern = []
    for pre in predict:
        line = []
        for id in pre:
            if id != 0 and id != 2:
                line.append(str(id))
        # 判断是否为空，若为空，则添加一个填充位PAD作为当前预测的摘要，避免程序报错
        if len(line) == 0:
            line.append(str(0))
        predictPattern.append(' '.join(line))
    # truePattern: ['id id id id','id id id',....],需要去除填充位PAD(id=0)和终止符_EOS(id=2)
    truePattern = []
    for tru in true.tolist():
        line = []
        for id in tru:
            if id != 0 and id != 2:
                line.append(str(id))
        # 判断是否为空，若为空，则添加一个填充位PAD作为当前真实的摘要，避免程序报错
        if len(line) == 0:
            line.append(str(0))
        truePattern.append(' '.join(line))
    return predictPattern,truePattern