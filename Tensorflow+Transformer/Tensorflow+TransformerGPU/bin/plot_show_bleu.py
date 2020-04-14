# -*- coding: utf-8 -*-
# @Time    : 2019/5/31 11:24
# @Author  : Weiyang
# @File    : plot_show_bleu.py
'''
读取每个Epoch的BLEU值，并作图展示
'''

import matplotlib.pyplot as plt

# 记录BLEU值
bleus1 = []
bleus2 = []
bleus3 = []
bleus4 = []
with open('../result/eval_bleu.score','r',encoding='utf-8') as fi:
    count = 1
    for line in fi:
        line = line.strip().split(':')
        if count == 1:
            bleus1.append(float(line[1]))
            count += 1
            continue
        if count == 2:
            bleus2.append(float(line[1]))
            count += 1
            continue
        if count == 3:
            bleus3.append(float(line[1]))
            count += 1
            continue
        if count == 4:
            bleus4.append(float(line[1]))
            count += 1
            continue
        if count == 5:
            count = 1
# 记录epoch
Epoches = list(range(1,len(bleus1)+1))

plt.plot(Epoches,bleus1,c='orange',label='bleu-1') # 传入数据
plt.plot(Epoches,bleus2,c='green',label='bleu-2') # 传入数据
plt.plot(Epoches,bleus3,c='c',label='bleu-3') # 传入数据
plt.plot(Epoches,bleus4,c='red',label='bleu-4') # 传入数据
plt.legend(loc='upper right') # 每条线标签的位置
plt.xticks(rotation=45) # 选择X轴可读
plt.xlabel('Epoch') # 设置X轴标签
plt.ylabel('BLEU')
plt.title('BLEU Trends')
plt.show()