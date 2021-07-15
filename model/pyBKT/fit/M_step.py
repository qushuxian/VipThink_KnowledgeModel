import numpy as np


def run(model, trans_softcounts, emission_softcounts, init_softcounts, prints=False):
    if prints:print('↓ '+'==' * 50)
    z = np.sum(trans_softcounts, axis=1) == 0
    for i in range(len(z)):
        for j in range(len(z[0])):
            if z[i, j]:
                trans_softcounts[i, 0, j] = 0
                trans_softcounts[i, 1, j] = 1
    emission_softcounts[np.sum(emission_softcounts, axis=2) == 0, :] = 1
    assert (trans_softcounts.shape[1] == 2)
    assert (trans_softcounts.shape[2] == 2)
    model['As'][:model['As'].shape[0]] = (trans_softcounts / np.sum(trans_softcounts, axis=1)[:model['As'].shape[0], None])
    if prints: print('M_step根据传入的all_trans_softcounts矩阵 / np.sum(all_trans_softcounts矩阵)得到新的As值为：\n', model['As'][:model['As'].shape[0]])

    model['learns'] = model['As'][:, 1, 0]
    model['forgets'] = model['As'][:, 0, 1]
    if prints:
        print('learns取自于新的As矩阵中二维的第一个值：', model['learns'])
        print('forgets取自于新的As矩阵中一维的第二个值：', model['forgets'])
        print()

    temp = np.sum(emission_softcounts, axis=2)

    model['emissions'] = emission_softcounts / temp[:, :, None]
    if prints:print('M_step根据传入的all_emission_softcounts矩阵 / np.sum(emission_softcounts矩阵)得到新的emissions值为：\n', model['emissions'])

    model['guesses'] = model['emissions'][:, 0, 1]
    model['slips'] = model['emissions'][:, 1, 0]
    if prints:
        print('guesses取自于新的emissions矩阵中一维的第二个值：', model['guesses'])
        print('slips取自于新的emissions矩阵中二维的第一个值：', model['slips'])
        print()

    model['pi_0'] = init_softcounts[:] / np.sum(init_softcounts[:])
    if prints:print('M_step根据传入的all_init_softcounts矩阵 / np.sum(init_softcounts矩阵)得到新的pi_0值为：\n', model['pi_0'])

    model['prior'] = model['pi_0'][1][0]
    if prints:
        print('prior取自于新的pi_0矩阵中二维的第一个值：', model['prior'])
        print('↑ '+'==' * 50)

    return model
