import numpy as np

def run(model, trans_softcounts, emission_softcounts, init_softcounts):
    # print('M_step接收的参数分别是：\nall_trans_softcounts=%s,\nall_emission_softcounts=%s,\nall_initial_softcounts=%s'
    #       % (trans_softcounts, emission_softcounts, init_softcounts))
    # print('M_step接收的模型是：', model)
    z = np.sum(trans_softcounts, axis=1) == 0
    for i in range(len(z)):
        for j in range(len(z[0])):
            if z[i, j]:
                trans_softcounts[i, 0, j] = 0
                trans_softcounts[i, 1, j] = 1

    emission_softcounts[np.sum(emission_softcounts, axis=2) == 0, :] = 1
    assert (trans_softcounts.shape[1] == 2)
    assert (trans_softcounts.shape[2] == 2)

    # print(np.sum(trans_softcounts, axis=1))
    model['As'][:model['As'].shape[0]] = (trans_softcounts / np.sum(trans_softcounts, axis=1)[:model['As'].shape[0], None])
    # print('M_step根据传入的model中的As矩阵 / np.sum(As矩阵)得到新的As值为：\n', model['As'][:model['As'].shape[0]])

    model['learns'] = model['As'][:, 1, 0]
    model['forgets'] = model['As'][:, 0, 1]

    temp = np.sum(emission_softcounts, axis=2)

    model['emissions'] = emission_softcounts / temp[:, :, None]
    model['guesses'] = model['emissions'][:, 0, 1]
    model['slips'] = model['emissions'][:, 1, 0]

    model['pi_0'] = init_softcounts[:] / np.sum(init_softcounts[:])
    model['prior'] = model['pi_0'][1][0]
    # print('M_step响应的模型参数是:', model)

    return model
