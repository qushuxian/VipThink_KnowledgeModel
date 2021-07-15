# """
# 原始代码结构
import numpy as np
from pyBKT.util import check_data
import os
from pyBKT.fit import M_step
from title.print_title import print_title
from multiprocessing import Pool, cpu_count

gs = globals()
np.random.seed(100)


def EM_fit(model, data, tol=None, maxiter=None, parallel=True, prints=False):
    if prints:
        print('\n\033[1;30;47m    EM_fit计算中......\n\033[0m')
    if tol is None:
        tol = 1e-3
        if prints:
            print('tol未接收到传入参数，采用默认参数tol=%s' % tol)
    else:
        tol = tol
        if prints:
            print('tol接收到传入参数tol=%s' % tol)
    if maxiter is None:
        maxiter = 100
        if prints:
            print('maxiter未接收到传入的迭代参数，采用默认参数maxiter=%s' % maxiter)
    else:
        maxiter = maxiter
        if prints:
            print('maxiter接收到传入参数maxiter=%s' % maxiter)

    check_data.check_data(data)
    num_subparts = data["data"].shape[0]
    num_resources = len(model["learns"])
    trans_softcounts = np.zeros((num_resources, 2, 2))
    emission_softcounts = np.zeros((num_subparts, 2, 2))
    init_softcounts = np.zeros((2, 1))
    log_likelihoods = np.zeros((maxiter, 1))
    result = {'all_trans_softcounts':   trans_softcounts, 'all_emission_softcounts': emission_softcounts,
              'all_initial_softcounts': init_softcounts}

    # 202106011039 新增将每一次迭代的EM参数返回
    model_list = []
    for i in range(maxiter):
        if prints:
            print(print_title('EM_fit根据maxiter={0}，循环第{1}次计算'.format(maxiter, i), colour='info'))

        if prints:
            print(print_title('传参进入E步(根据当前的参数值，计算样本隐藏变量的期望)计算......', colour='info', total_len=0))

        # 此处介入是否进行多线程C++程序处理
        if parallel:
            from pyBKT.fit import E_step
            result = E_step.run(data, model, 1, int(parallel))
            print(print_title('传参进入E步(根据当前的参数值，计算样本隐藏变量的期望)计算后得到的E步期望值', colour='info', total_len=0))
            print(result)
            print(print_title('这里需要all_trans_softcounts all_emission_softcounts all_initial_softcounts', colour='info', total_len=0))

            log_likelihoods[i][0] = result['total_loglike']
            if prints:
                print(print_title('第{0}次计算total_loglike={1}'.format(i, log_likelihoods[i][0]), colour='info', total_len=0))
                print(print_title('第{0}次计算total_loglike={1}'.format(i, log_likelihoods[i - 1][0]), colour='info', total_len=0))

            if i > 1 and abs(log_likelihoods[i][0] - log_likelihoods[i - 1][0]) < tol:
                if prints:
                    print(print_title('跳出迭代（循环次数{0} > 1，且最后一个likelihoods{1}-前一个likelihoods{2}≤{3}达到收敛条件、跳出循环）'.format(
                            i, log_likelihoods[i][0], log_likelihoods[i - 1][0], tol), colour='info', total_len=0))
                break
            if prints:
                print(print_title('将模型和得到的3个E步期望值，传参进入M步(根据当前样本的隐藏变量的期望，求解参数的最大似然估计)计算', colour='info', total_len=0))
            model = M_step.run(model,
                               result['all_trans_softcounts'],
                               result['all_emission_softcounts'],
                               result['all_initial_softcounts'],
                               )
            model_list.append({'iter': i, 'values': model})
            if prints:
                print(print_title('得到的M步最大似然估计', colour='info', total_len=0))
                print(model)
        else:
            result = run(data, model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'],
                         1, parallel, prints)
            if prints:
                print(print_title('传参进入E步(根据当前的参数值，计算样本隐藏变量的期望)计算后得到的E步期望值', colour='info', total_len=0))
                print(result)
                print(print_title('这里需要all_trans_softcounts all_emission_softcounts all_initial_softcounts', colour='info', total_len=0))

            for j in range(num_resources):
                result['all_trans_softcounts'][j] = result['all_trans_softcounts'][j].transpose()
            for j in range(num_subparts):
                result['all_emission_softcounts'][j] = result['all_emission_softcounts'][j].transpose()

            log_likelihoods[i][0] = result['total_loglike']
            if prints:
                print(print_title('第{0}次计算total_loglike={1}'.format(i, log_likelihoods[i][0]), colour='info', total_len=0))
                print(print_title('第{0}次计算total_loglike={1}'.format(i, log_likelihoods[i - 1][0]), colour='info', total_len=0))
            if i > 1 and abs(log_likelihoods[i][0] - log_likelihoods[i - 1][0]) <= tol:
                if prints:
                    print(print_title('跳出迭代（循环次数{0} > 1，且最后一个likelihoods{1}-前一个likelihoods{2}≤{3}达到收敛条件、跳出循环）'.format(
                            i, log_likelihoods[i][0], log_likelihoods[i - 1][0], tol), colour='info', total_len=0))
                    print()
                break

            if prints:
                print(print_title('将模型和得到的3个E步期望值，传参进入M步(根据当前样本的隐藏变量的期望，求解参数的最大似然估计)计算', colour='info', total_len=0))
            model = M_step.run(model, result['all_trans_softcounts'], result['all_emission_softcounts'], result['all_initial_softcounts'], prints)
            model_list.append({'iter': i, 'values': model})
            if prints:print(print_title('得到的M步最大似然估计', colour='info', total_len=0))
            if prints:print(model)
    if prints:
        print('\n\033[1;30;47m    EM(最大似然估计)响应的模型参数是：\n\033[0m',  model)

    # 20210601新增return model_list
    return model, log_likelihoods[:i + 1], model_list


def run(data, model, trans_softcounts, emission_softcounts, init_softcounts, num_outputs, parallel=True, prints=False):
    alldata = data["data"]
    bigT, num_subparts = len(alldata[0]), len(alldata)
    allresources, starts, learns, forgets, guesses, slips, lengths = \
            data["resources"], data["starts"], model["learns"], model["forgets"], model["guesses"], model["slips"], data["lengths"]

    prior, num_sequences, num_resources = model["prior"], len(starts), len(learns)
    normalizeLengths = False

    initial_distn = np.empty((2, ), dtype='float')
    initial_distn[0] = 1 - prior
    initial_distn[1] = prior

    As = np.empty((2, 2 * num_resources))
    interleave(As[0], 1 - learns, forgets.copy())
    interleave(As[1], learns.copy(), 1 - forgets)

    Bn = np.empty((2, 2 * num_subparts))
    interleave(Bn[0], 1 - guesses, guesses.copy())
    interleave(Bn[1], slips.copy(), 1 - slips)

    # Outputs
    all_trans_softcounts = np.zeros((2, 2 * num_resources))
    all_emission_softcounts = np.zeros((2, 2 * num_subparts))
    all_initial_softcounts = np.zeros((2, 1))

    alpha_out = np.zeros((2, bigT))

    total_loglike = np.empty((1,1))
    total_loglike.fill(0)

    input = {"As": As, "Bn": Bn, "initial_distn": initial_distn, 'allresources': allresources,
             'starts': starts, 'lengths': lengths, 'num_resources': num_resources, 'num_subparts': num_subparts,
             'alldata': alldata, 'normalizeLengths': normalizeLengths, 'alpha_out': alpha_out}

    num_threads = cpu_count() if parallel else 1
    thread_counts = [None for i in range(num_threads)]
    for thread_num in range(num_threads):
        blocklen = 1 + ((num_sequences - 1) // num_threads)
        sequence_idx_start = int(blocklen * thread_num)
        sequence_idx_end = min(sequence_idx_start+blocklen, num_sequences)
        thread_counts[thread_num] = {'sequence_idx_start': sequence_idx_start, 'sequence_idx_end': sequence_idx_end}
        thread_counts[thread_num].update(input)

    x = [inner(i, prints) for i in thread_counts]

    for i in x:
        total_loglike += i[3]
        all_trans_softcounts += i[0]
        all_emission_softcounts += i[1]
        all_initial_softcounts += i[2]
        for sequence_start, T, alpha in i[4]:
            alpha_out[:, sequence_start: sequence_start + T] += alpha
    all_trans_softcounts = all_trans_softcounts.flatten(order='F')
    all_emission_softcounts = all_emission_softcounts.flatten(order='F')

    result = {"total_loglike":           total_loglike,
              "all_trans_softcounts":    np.reshape(all_trans_softcounts, (num_resources, 2, 2), order='C'),
              "all_emission_softcounts": np.reshape(all_emission_softcounts, (num_subparts, 2, 2), order='C'),
              "all_initial_softcounts":  all_initial_softcounts,
              "alpha_out":               alpha_out.flatten(order='F').reshape(alpha_out.shape, order='C')}
    return result


def interleave(m, v1, v2):
    m[0::2], m[1::2] = v1, v2


def inner(x, prints=False):
    As, Bn, initial_distn, allresources, starts = x['As'], x['Bn'], x['initial_distn'], x['allresources'], x['starts']
    lengths, num_resources = x['lengths'], x['num_resources']
    num_subparts, alldata, normalizeLengths = x['num_subparts'], x['alldata'], x['normalizeLengths']
    alpha_out, sequence_idx_start, sequence_idx_end = x['alpha_out'], x['sequence_idx_start'], x['sequence_idx_end']

    N_R, N_S = 2 * num_resources, 2 * num_subparts
    trans_softcounts_temp = np.zeros((2, N_R))
    emission_softcounts_temp = np.zeros((2, N_S))
    init_softcounts_temp = np.zeros((2, 1))
    loglike = 0

    alphas = []
    dot, sum, log = np.dot, np.sum, np.log
    if prints:
        print(print_title('As'), As)
        print('As[0]=对应的是1-Learns')
        print('As[1]=对应的是1-As[0]')
        print(print_title('Bn'), Bn)
        print('Bn[:, 1]=对应的是Guesses和Slips')
        print('Bn[:, 0]=对应的是1-Guesses和1-Slips')
        print(print_title('initial_distn'), initial_distn)
        print('initial_distn[:, 1]=对应的是prior')
        print('initial_distn[:, 0]=对应的是1 - prior')

    for sequence_index in range(sequence_idx_start, sequence_idx_end):
        sequence_start = starts[sequence_index] - 1
        if prints: print(print_title('sequence_start'), sequence_start)
        T = lengths[sequence_index]
        if prints: print(print_title('T'), T)

        # 计算最大似然估计值
        likelihoods = np.ones((2, T))
        if prints: print(print_title('create_likelihoods_array'), likelihoods)
        alpha = np.empty((2, T))
        if prints: print(print_title('create_alpha_array'), alpha)
        for t in range(min(2, T)):
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    sl = Bn[:, 2 * n + int(data_temp == 2)]
                    # print('这里的数据取自Bn（Guesses和Slips）', sl)
                    likelihoods[:, t] *= np.where(sl == 0, 1, sl)
        if prints:
            print(print_title('likelihoods=Bn.T'), likelihoods)

            # 为 alpha 设置，包含在循环中以提高效率(保持它作为一个循环)
            print()
            print(print_title('这里是第0个答题可能性计算', title_format='/'))
            print(print_title('likelihoods'), likelihoods)
        alpha[:, 0] = initial_distn * likelihoods[:, 0]
        if prints:print(print_title('alpha=initial_distn * likelihoods[:, 0]'), alpha)
        norm = sum(alpha[:, 0])
        if prints:print(print_title('norm=sum(alpha[:, 0])'), norm)
        alpha[:, 0] /= norm
        if prints:print(print_title('alpha=alpha[:, 0]/norm'), alpha)
        contribution = log(norm) / (T if normalizeLengths else 1)
        loglike += contribution

        # 与 t 结合 = 2 为了效率，否则我们需要另一个循环
        if T >= 2:
            resources_temp = allresources[sequence_start]
            k = 2 * (resources_temp - 1)
            if prints:
                print()
                print(print_title('这里是第1个答题可能性计算', title_format='/'))
                print(print_title('likelihoods'), likelihoods)
            alpha[:, 1] = dot(As[0:2, k: k + 2], alpha[:, 0]) * likelihoods[:, 1]
            if prints:
                print('As[0:2, k: k + 2]:    ', As[0:2, k: k + 2])
                print('alpha[:, 0]:    ', alpha[:, 0])
                print('dot:    ', dot(As[0:2, k: k + 2], alpha[:, 0]))
                print(print_title('alpha[:, 1]=dot(As,alpha[:, 0])*likelihoods[:, 1]'), alpha)
            norm = sum(alpha[:, 1])
            if prints:print(print_title('norm=sum(alpha[:, 1])'), norm)
            alpha[:, 1] /= norm
            if prints:print(print_title('alpha=alpha[:, 1]/norm'), alpha)
            contribution = log(norm) / (T if normalizeLengths else 1)
            loglike += contribution

        for t in range(2, T):
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    sl = Bn[:, 2 * n + int(data_temp == 2)]
                    likelihoods[:, t] *= np.where(sl == 0, 1, sl)
            # general loop for alpha calculations
            # alpha 计算的通用循环
            if prints:
                print()
                print(print_title('这里是第{0}个答题可能性计算'.format(t), title_format='/'))
                print(print_title('likelihoods'), likelihoods)
            resources_temp = allresources[sequence_start + t - 1]
            k = 2 * (resources_temp - 1)
            alpha[:, t] = dot(As[0:2, k: k + 2], alpha[:, t - 1]) * likelihoods[:, t]
            if prints:print(print_title('alpha[:, {0}]=dot(As,alpha[:, {1}])*likelihoods[:, {0}]'.format(t, t - 1)), alpha)
            norm = sum(alpha[:, t])
            if prints:print(print_title('norm=sum(alpha[:, {0}])'.format(t)), norm)
            alpha[:, t] /= norm
            if prints:print(print_title('alpha=alpha[:, {0}]/norm'.format(t)), alpha)
            loglike += log(norm) / (T if normalizeLengths else 1)

        # 对计算结果反向统计计数
        if prints:
            print()
            print(print_title('开始对计算结果反向统计计算', title_format='#'))
        gamma = np.empty((2, T))
        gamma[:, (T - 1)] = alpha[:, (T - 1)].copy()
        if prints:print(print_title('创建gamma，并赋值alpha'), gamma)

        As_temp = As.copy()
        if prints:print(print_title('创建As_temp，并赋值As'), As_temp)
        f = True
        for t in range(T - 2, -1, -1):
            if prints:print(print_title('对alpha[:, {0}]进行反向计算'.format(t), title_format='/'))
            resources_temp = allresources[sequence_start + t]
            k = 2 * (resources_temp - 1)
            A = As_temp[0: 2, k: k + 2]
            pair = A.copy()  # don't want to modify original A
            if prints:print('copy As，避免对原来的As进行修改（pair）\n', pair)
            pair[0] *= alpha[:, t]
            pair[1] *= alpha[:, t]
            if prints:print('\n取pair[0]和pair[1] 乘 alpha[:, {0}], 得到新的As值（pair）\n'.format(t), pair)
            dotted = dot(A, alpha[:, t])
            if prints:
                print(A)
                print(alpha[:, t])
                print('\n取原有的As和alpha[:, {0}]进行乘积计算\n'.format(t), dotted)
            gamma_t = gamma[:, (t + 1)]
            if prints:print('\n创建gamma_t，并赋值gamma[:, {0}]（pair）\n'.format(t + 1), gamma_t)
            pair[:, 0] = (pair[:, 0] * gamma_t) / dotted
            pair[:, 1] = (pair[:, 1] * gamma_t) / dotted
            if prints:print('\n将得到copy的As值 * gamma_t / 乘积计算结果 = pair\n', pair)
            np.nan_to_num(pair, copy=False)
            trans_softcounts_temp[0: 2, k: k + 2] += pair
            if prints:
                print('\n输出结果【【trans_softcounts_temp】】= pair\n', trans_softcounts_temp)
                print(pair)
            gamma[:, t] = sum(pair, axis=0)
            if prints:print('\ngamma=sum(pair, axis=0)（更新alpha）\n', gamma)
            for n in range(num_subparts):
                data_temp = alldata[n][sequence_start + t]
                if data_temp:
                    emission_softcounts_temp[:, (2 * n + int(data_temp == 2))] = emission_softcounts_temp[:,
                                                                                 (2 * n + int(data_temp == 2))] + gamma[:, t]
                    if prints:print('\n将更新的alpha进行整理\n', emission_softcounts_temp)
                if f:
                    data_temp_p = alldata[n][sequence_start + (T - 1)]
                    if data_temp_p:
                        emission_softcounts_temp[:, (2 * n + int(data_temp_p == 2))] += gamma[:, (T - 1)]
                        if prints:
                            print('\n获取gamma[:, {0}]值）\n'.format(t + 1), gamma[:, (T - 1)])
                            print('\n将整理好的alpha + 获取gamma[:, {0}]值），输出结果【【emission_softcounts_temp】】\n'.format(t + 1), emission_softcounts_temp)
            f = False
        if prints:
            print()
            print(print_title('输出正向统计计算结果', title_format='#'))
        init_softcounts_temp += gamma[:, 0].reshape((2, 1))
        if prints:print(print_title('输出结果【【init_softcounts_temp】】 = 更新alpha的第一列'), init_softcounts_temp)
        alphas.append((sequence_start, T, alpha))
        if prints:print(print_title('输出结果【【正向计算的alphas】】 前两个数字表示的是原始数据索引'), alphas)
    return [trans_softcounts_temp, emission_softcounts_temp, init_softcounts_temp, loglike, alphas]