# -*- coding: utf-8 -*-
# @Time    : 2020/3/31 11:51
# @Author  : Weiyang
# @File    : Gaussian_kld.py

# --------------------------
# 条件变分编码器的高斯KL散度
# --------------------------

import torch

def Gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
    '''
    :param recog_mu: [batch_size,hidden_size]
    :param recog_logvar: [batch_size,hidden_size]
    :param prior_mu: [batch_size,hidden_size]
    :param prior_logvar: [batch_size,hidden_size]
    :return: 返回CVAE的KL散度值
    '''
    kld = -0.5 * torch.sum(torch.sum((1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow((prior_mu - recog_mu),2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar))), dim=1),dim=0)
    return kld

if __name__ == '__main__':
    recog_mu = torch.ones((3,4))
    recog_logvar = torch.ones((3,4)) * 0.5
    prior_mu = torch.ones((3, 4)) * 0.1
    prior_logvar = torch.ones((3, 4)) * 0.6

    kld = Gaussian_kld(recog_mu,recog_logvar,prior_mu,prior_logvar)
    print(kld)