import random
import numpy as np
from util import dirrnd

def random_model_uni(num_resources=None,
                     num_subparts=None,
                     trans_prior=None,
                     given_notknow_prior=None,
                     given_know_prior=None,
                     pi_0_prior=None,
                     prints=False):
    if prints: print('\n\033[1;30;47m    model_uni计算中......\n\033[0m')
    if num_resources is None:
        num_resources = 1
        if prints: print('num_resources未接收到传入参数，采用默认参数num_resources=%s' % num_resources)
    if prints: print('num_resources接收到传入参数num_resources=%s' % num_resources)

    if num_subparts is None:
        num_subparts = 1
        if prints: print('num_subparts未接收到传入参数，采用默认参数num_subparts=%s' % num_subparts)
    if prints: print('num_subparts接收到传入参数num_subparts=%s' % num_subparts)

    if trans_prior is None:
        """
        np.tile：将原矩阵横向、纵向地复制
        np.tile(np.transpose([[20, 4], [1, 20]]), (num_resources, 1))：
            就是将[[20, 4], [1, 20]]纵向的复制num_resources次，横向的复制1次
            并将复制铺展开的数组选择为num_resources个数字
        """
        trans_prior = np.tile(np.transpose([[20, 4], [1, 20]]), (num_resources, 1)).reshape((num_resources, 2, 2))
        if prints: print('trans_prior未接收到传入参数矩阵，采用默认参数矩阵trans_prior=\n%s' % trans_prior)
    else:
        trans_prior = trans_prior
        if prints: print('trans_prior接收到传入参数矩阵trans_prior=\n%s' % trans_prior)

    if given_notknow_prior is None:
        given_notknow_prior = np.tile([[5], [0.5]], (1, num_subparts))
        if prints: print('given_notknow_prior未接收到传入参数矩阵，采用默认参数矩阵given_notknow_prior=\n%s' % given_notknow_prior)
    else:
        given_notknow_prior = given_notknow_prior
        if prints: print('given_notknow_prior接收到传入参数矩阵given_notknow_prior=\n%s' % given_notknow_prior)

    if given_know_prior is None:
        given_know_prior = np.tile([[0.5], [5]], (1, num_subparts))
        if prints: print('given_know_prior未接收到传入参数矩阵，采用默认参数矩阵given_know_prior=\n%s' % given_know_prior)
    else:
        given_know_prior = given_know_prior
        if prints: print('given_know_prior接收到传入参数矩阵given_know_prior=\n%s' % given_know_prior)

    if pi_0_prior is None:
        pi_0_prior = np.array([[100], [1]])
        if prints: print('pi_0_prior未接收到传入参数矩阵，采用默认参数矩阵pi_0_prior=\n%s' % pi_0_prior)
    else:
        pi_0_prior = pi_0_prior
        if prints: print('pi_0_prior接收到传入参数矩阵pi_0_prior=\n%s' % pi_0_prior)

    As = dirrnd.dirrnd(trans_prior)
    given_notknow = dirrnd.dirrnd(given_notknow_prior)
    given_know = dirrnd.dirrnd(given_know_prior)
    emissions = np.stack((np.transpose(given_notknow.reshape((2, num_subparts))), np.transpose(given_know.reshape((2, num_subparts)))), axis=1)
    pi_0 = dirrnd.dirrnd(pi_0_prior)
    if prints:
        print('trans_prior计算后的伽马分布随机样本(As)为：\n%s' % As)
        print('given_notknow_prior计算后的伽马分布随机样本(given_notknow)为：\n%s' % given_notknow)
        print('given_know_prior算后的伽马分布随机样本(given_know)为：\n%s' % given_know)
        print('given_notknow和given_know经过num_subparts计算后的堆叠矩阵为：\n%s' % emissions)
        print('pi_0_prior算后的伽马分布随机样本(pi_0)为：\n%s' % pi_0)

    """
    prior = random.random()
    learns = 返回服从“0~1”均匀分布的随机样本(np.random.rand(num_resources)*0.4)
    """
    modelstruct = {}
    modelstruct['prior'] = random.random()
    As[:, 1, 0] = np.random.rand(num_resources) * 0.40
    As[:, 1, 1] = 1 - As[:, 1, 0]
    As[:, 0, 1] = 0
    As[:, 0, 0] = 1
    modelstruct['learns'] = As[:, 1, 0]
    modelstruct['forgets'] = As[:, 0, 1]
    given_notknow[1, :] = np.random.rand(num_subparts) * 0.40
    modelstruct['guesses'] = given_notknow[1, :]
    given_know[0, :] = np.random.rand(num_subparts) * 0.30
    modelstruct['slips'] = given_know[0, :]

    modelstruct['As'] = As
    modelstruct['emissions'] = emissions
    modelstruct['pi_0'] = pi_0
    if prints:
        print('\n\033[1;30;47m    model_uni经过伽马分布计算后响应的初始化参数为：\n\033[0m')
        print(modelstruct)
    return modelstruct



