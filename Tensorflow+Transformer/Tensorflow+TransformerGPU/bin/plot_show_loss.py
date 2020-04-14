# -*- coding: utf-8 -*-
# @Time    : 2019/5/31 11:24
# @Author  : Weiyang
# @File    : plot_show_loss.py
'''
读取每epoch训练loss，并作图展示
'''

import matplotlib.pyplot as plt

# 记录loss
losses = []
with open('../result/train/loss.txt','r',encoding='utf-8') as fi:
    for line in fi:
        line = line.strip().split(':')
        losses.append(float(line[1]))
# 记录epoch
Epoches = list(range(1,len(losses)+1))
plt.plot(Epoches,losses,c='orange') # 传入数据
plt.xticks(rotation=45) # 选择X轴可读
plt.xlabel('Epoch') # 设置X轴标签
plt.ylabel('Cross Entropy Loss')
plt.title('Cross Entropy Loss Trends')
plt.show()