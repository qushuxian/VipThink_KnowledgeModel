# @Author : 曲书贤
# @File : BKT_TEST.py
# @CreateTime : 2021/6/3 16:18
# @Software : PyCharm
# @Comment : 知识点多重验证


import joblib
from sklearn.metrics import mean_absolute_error
import pandas as pd
import warnings
import datetime
import os
from pyBKT.models import Model
import json
from ML_methd import Plt
from Transform import Check, Get
from excel import import_excel
from multiprocessing.dummy import Pool as ThreadPool
warnings.filterwarnings('ignore')
plt = Plt()


def model_info(path_list):
    model_big_coef = {'forgets': [],
                      'skills': [],
                      'num_fits': [],
                      'coef': [],
                      'coef_prior': [],
                      'coef_learns': [],
                      'coef_guesses': [],
                      'coef_slips': [],
                      'coef_forgets': [],
                      'fit_model': [],
                      'fit_model_as': [],
                      'fit_model_resource_names': [],
                      'fit_em_list': [],
                      }
    path_lists = Check.check_path_list(path_list)
    for m_path in path_lists:
        models = m_path.split('/')[2]
        chapter = m_path.split('_')[-3:]
        model = joblib.load(m_path)
        if (chapter[0] != model.skills[0]) | (int(chapter[-1].split('.')[0]) != model.num_fits):
            print('传入的文件path与实际解析的模型skills或num_fits不一致，请检查！！')
            continue
        else:
            model_big_coef['forgets'].append(models)
            model_big_coef['skills'].append(model.skills[0])
            model_big_coef['num_fits'].append(model.num_fits)
            model_big_coef['coef'].append(model.coef_)
            model_big_coef['coef_prior'].append(model.coef_[model.skills[0]]['prior'])
            model_big_coef['coef_learns'].append(model.coef_[model.skills[0]]['learns'])
            model_big_coef['coef_guesses'].append(model.coef_[model.skills[0]]['guesses'])
            model_big_coef['coef_slips'].append(model.coef_[model.skills[0]]['slips'])
            model_big_coef['coef_forgets'].append(model.coef_[model.skills[0]]['forgets'])

            model_big_coef['fit_model'].append(model.fit_model)
            model_big_coef['fit_model_as'].append(model.fit_model[model.skills[0]]['As'])
            model_big_coef['fit_model_resource_names'].append(model.fit_model[model.skills[0]]['resource_names'].keys())

            model_big_coef['fit_em_list'].append(model.fit_em_list)
    return pd.DataFrame(model_big_coef)


def model_evaluate(evaluate_list):
    df = {'chapter_id': [], 'chapter_data_cnt': [], 'num_fits': [], 'model': [],
          'auc': [], 'rmse': [], 'accuracy': [], 'mae': []}
    evaluate_lists = Check.check_path_list(evaluate_list)
    for line_list in evaluate_lists:
        models = line_list.split('/')[1]
        evaluate_df = open(line_list, 'rb')
        for line in evaluate_df.readlines():
            json_loads = json.loads(line)
            for i in json_loads:
                dicts = dict(i)
                df['chapter_id'].append(dicts['chapter'][0])
                df['chapter_data_cnt'].append(dicts['chapter_data_cnt'][0])
                df['num_fits'].append(dicts['fits'][0])
                df['model'].append(models)
                df['auc'].append(dicts['metric'][0])
                df['rmse'].append(dicts['metric'][1])
                df['accuracy'].append(dicts['metric'][2])
                df['mae'].append(dicts['metric'][3])
        evaluate_df.close
    df = pd.DataFrame(df)
    return df


def write(path, data):
    with open(path, 'w') as dfs:
        dfs.write(json.dumps(data))


# 获取文件夹下全部文件的名称，并对文件数据进行模型拟合
file_path = Get.get_file_path('/Users/vipthink/Downloads/train_data/')
file_path_10 = file_path[:10]
print(datetime.datetime.now(), '运行开始......')
start_time = datetime.datetime.now()
for num in range(len(file_path_10)):
    num_start_time = datetime.datetime.now()
    file_name = file_path[num].split('/')[-1].split('.')[0]
    chapter_df = pd.read_csv(file_path[num])
    print('     正在针对第%s个知识点[%s]进行模型拟合，该知识点数据量共计%s行......' % (num, file_name, chapter_df.shape[0]), datetime.datetime.now())
    defaults = {'skill_name': 'chapter_id',
                'correct': 'status', 'user_id': 'user_id',
                'multigs': 'level', 'multilearn': 'level'}

    evaluate_forgets_n = []
    evaluate_forgets_n_multigs = []
    evaluate_forgets_y_one = []
    evaluate_forgets_y_two = []
    evaluate_forgets_y_three = []
    evaluate_forgets_y_four = []
    evaluate_forgets_y_five = []

    for fits in range(1, 31):
        # ###########################拟合标准模型
        model_start_time = datetime.datetime.now()
        print('         forgets_n_%s_fits_%s is running......' % (file_name, fits), datetime.datetime.now())
        forgets_n = Model(seed=0, num_fits=int(fits), print=False)
        forgets_n.fit(data=chapter_df, defaults=defaults, print=False)
        evaluate_forgets_n.append({'chapter': [file_name], 'chapter_data_cnt': [chapter_df.shape[0]],
                                   'fits': [fits], 'model': ['forgets_n'],
                                   'metric': forgets_n.evaluate(data=chapter_df,
                                                                metric=["auc", "rmse", "accuracy", mean_absolute_error])})
        if fits < 10:
            forgets_n_model_name = './train_pkl/model_1/forgets_n_%s_fits_0%s.pkl' % (str(file_name), str(fits))
        else:
            forgets_n_model_name = './train_pkl/model_1/forgets_n_%s_fits_%s.pkl' % (str(file_name), str(fits))
        joblib.dump(forgets_n, forgets_n_model_name)
        model_end_time = datetime.datetime.now()
        print('         forgets_n_%s_fits_%s is end，time:' % (file_name, fits), model_end_time-model_start_time)

        # ###########################拟合标准模型_知识点按用户生成表现参数（guess、slip）
        model_start_time1 = datetime.datetime.now()
        print('\n         forgets_n_%s_multigs_fits_%s is running......' % (file_name, fits), datetime.datetime.now())
        forgets_n_multigs = Model(seed=0, num_fits=int(fits), print=False)
        forgets_n_multigs.fit(data=chapter_df, defaults=defaults, multilearn='user_id', print=False)
        evaluate_forgets_n_multigs.append({'chapter': [file_name], 'chapter_data_cnt': [chapter_df.shape[0]],
                                           'fits': [fits], 'model': ['forgets_n_multigs'],
                                           'metric': forgets_n_multigs.evaluate(data=chapter_df,
                                                                                metric=["auc", "rmse", "accuracy",
                                                                                        mean_absolute_error])})
        if fits < 10:
            forgets_n_multigs_model_name = './train_pkl/model_2/forgets_n_multigs_%s_fits_0%s.pkl' % (str(file_name), str(fits))
        else:
            forgets_n_multigs_model_name = './train_pkl/model_2/forgets_n_multigs_%s_fits_%s.pkl' % (str(file_name), str(fits))
        joblib.dump(forgets_n_multigs, forgets_n_multigs_model_name)
        model_end_time1 = datetime.datetime.now()
        print('         forgets_n_%s_multigs_fits_%s is end，time:' % (file_name, fits),
              model_end_time1 - model_start_time1)

        # ###########################拟合遗忘模型one
        model2_start_time = datetime.datetime.now()
        print('\n         forgets_y_%s_one_fits_%s is running......' % (file_name, fits), datetime.datetime.now())
        forgets_y_one = Model(seed=0, num_fits=int(fits), print=False)
        forgets_y_one.fit(data=chapter_df, defaults=defaults, forgets=True, print=False)
        evaluate_forgets_y_one.append({'chapter': [file_name], 'chapter_data_cnt': [chapter_df.shape[0]],
                                       'fits': [fits], 'model': ['forgets_y_one'],
                                       'metric': forgets_y_one.evaluate(data=chapter_df,
                                                                        metric=["auc", "rmse", "accuracy",
                                                                                mean_absolute_error])})
        if fits < 10:
            forgets_y_one_model_name = './train_pkl/model_3/forgets_y_one_%s_fits_0%s.pkl' % (str(file_name), str(fits))
        else:
            forgets_y_one_model_name = './train_pkl/model_3/forgets_y_one_%s_fits_%s.pkl' % (str(file_name), str(fits))
        joblib.dump(forgets_y_one, forgets_y_one_model_name)
        model2_end_time = datetime.datetime.now()
        print('         forgets_y_%s_one_fits_%s is end，time:' % (file_name, fits), model2_end_time - model2_start_time)

        # ###########################拟合遗忘模型two
        model3_start_time = datetime.datetime.now()
        print('\n         forgets_y_%s_two_fits_%s is running......' % (file_name, fits), datetime.datetime.now())
        forgets_y_two = Model(seed=0, num_fits=int(fits), print=False)
        forgets_y_two.fit(data=chapter_df, defaults=defaults, forgets=True, multigs=True, print=False)
        evaluate_forgets_y_two.append({'chapter': [file_name], 'chapter_data_cnt': [chapter_df.shape[0]],
                                       'fits': [fits], 'model': ['forgets_y'],
                                       'metric': forgets_y_two.evaluate(data=chapter_df,
                                                                        metric=["auc", "rmse", "accuracy",
                                                                                mean_absolute_error])})
        if fits < 10:
            forgets_y_two_model_name = './train_pkl/model_4/forgets_y_two_%s_fits_0%s.pkl' % (str(file_name), str(fits))
        else:
            forgets_y_two_model_name = './train_pkl/model_4/forgets_y_two_%s_fits_%s.pkl' % (str(file_name), str(fits))
        joblib.dump(forgets_y_two, forgets_y_two_model_name)
        model3_end_time = datetime.datetime.now()
        print('         forgets_y_%s_two_fits_%s is end，time:' % (file_name, fits), model3_end_time - model3_start_time)

        # ###########################拟合遗忘模型three
        model4_start_time = datetime.datetime.now()
        print('\n         forgets_y_%s_three_fits_%s is running......' % (file_name, fits), datetime.datetime.now())
        forgets_y_three = Model(seed=0, num_fits=int(fits), print=False)
        forgets_y_three.fit(data=chapter_df, defaults=defaults, forgets=True, multigs=True, print=False)
        evaluate_forgets_y_three.append({'chapter': [file_name], 'chapter_data_cnt': [chapter_df.shape[0]],
                                         'fits': [fits], 'model': ['forgets_y'],
                                         'metric': forgets_y_three.evaluate(data=chapter_df,
                                                                            metric=["auc", "rmse", "accuracy",
                                                                                    mean_absolute_error])})
        if fits < 10:
            forgets_y_three_model_name = './train_pkl/model_5/forgets_y_three_%s_fits_0%s.pkl' % (str(file_name), str(fits))
        else:
            forgets_y_three_model_name = './train_pkl/model_5/forgets_y_three_%s_fits_%s.pkl' % (str(file_name), str(fits))
        joblib.dump(forgets_y_three, forgets_y_three_model_name)
        model4_end_time = datetime.datetime.now()
        print('         forgets_y_%s_three_fits_%s is end，time:' % (file_name, fits), model4_end_time - model4_start_time)

        # ###########################拟合遗忘模型four
        model5_start_time = datetime.datetime.now()
        print('\n         forgets_y_%s_four_fits_%s is running......' % (file_name, fits), datetime.datetime.now())
        forgets_y_four = Model(seed=0, num_fits=int(fits), print=False)
        forgets_y_four.fit(data=chapter_df, defaults=defaults, forgets=True, multigs=True, multilearn=True, print=False)
        evaluate_forgets_y_four.append(
            {'chapter': [file_name], 'chapter_data_cnt': [chapter_df.shape[0]], 'fits': [fits], 'model': ['forgets_y'],
             'metric': forgets_y_four.evaluate(data=chapter_df, metric=["auc", "rmse", "accuracy", mean_absolute_error])})
        if fits < 10:
            forgets_y_four_model_name = './train_pkl/model_6/forgets_y_four_%s_fits_0%s.pkl' % (str(file_name), str(fits))
        else:
            forgets_y_four_model_name = './train_pkl/model_6/forgets_y_four_%s_fits_%s.pkl' % (str(file_name), str(fits))
        joblib.dump(forgets_y_four, forgets_y_four_model_name)
        model5_end_time = datetime.datetime.now()
        print('         forgets_y_%s_four_fits_%s is end，time:' % (file_name, fits),
              model5_end_time - model5_start_time)

        # ###########################拟合遗忘模型five
        model6_start_time = datetime.datetime.now()
        print('\n         forgets_y_%s_five_fits_%s is running......' % (file_name, fits), datetime.datetime.now())
        forgets_y_five = Model(seed=0, num_fits=int(fits), print=False)
        forgets_y_five.fit(data=chapter_df, defaults=defaults, forgets=True, multigs=True, multilearn='user_id', print=False)
        evaluate_forgets_y_five.append(
            {'chapter': [file_name], 'chapter_data_cnt': [chapter_df.shape[0]], 'fits': [fits], 'model': ['forgets_y'],
             'metric': forgets_y_five.evaluate(data=chapter_df,
                                               metric=["auc", "rmse", "accuracy", mean_absolute_error])})
        if fits < 10:
            forgets_y_four_model_name = './train_pkl/model_7/forgets_y_five_%s_fits_0%s.pkl' % (str(file_name), str(fits))
        else:
            forgets_y_four_model_name = './train_pkl/model_7/forgets_y_five_%s_fits_%s.pkl' % (str(file_name), str(fits))
        joblib.dump(forgets_y_four, forgets_y_four_model_name)
        model6_end_time = datetime.datetime.now()
        print('         forgets_y_%s_five_fits_%s is end，time:' % (file_name, fits),
              model6_end_time - model6_start_time)
        print('')

    # 每个模型拟合结束后，将评估结果保存
    write('./train_evaluate_info/model_1/%s_evaluate_forgets_n.txt' % str(file_name), evaluate_forgets_n)
    write('./train_evaluate_info/model_2/%s_evaluate_forgets_n_multigs.txt' % str(file_name), evaluate_forgets_n_multigs)
    write('./train_evaluate_info/model_3/%s_evaluate_forgets_y_one.txt' % str(file_name), evaluate_forgets_y_one)
    write('./train_evaluate_info/model_4/%s_evaluate_forgets_y_two.txt' % str(file_name), evaluate_forgets_y_two)
    write('./train_evaluate_info/model_5/%s_evaluate_forgets_y_three.txt' % str(file_name), evaluate_forgets_y_three)
    write('./train_evaluate_info/model_6/%s_evaluate_forgets_y_four.txt' % str(file_name), evaluate_forgets_y_four)
    write('./train_evaluate_info/model_7/%s_evaluate_forgets_y_five.txt' % str(file_name), evaluate_forgets_y_five)
    num_end_time = datetime.datetime.now()
    print('     第%s个知识点[%s]模型拟合完成，用时：' % (num, file_name), num_end_time-num_start_time)
    print('\n\n')
end_time = datetime.datetime.now()
print('知识点模型拟合完成，总用时：', end_time-start_time)


# 将多个train_pkl参数进行合并
# model_coef = model_info(Get.get_file_path('train_pkl/'))
# model_evaluates = model_evaluate(Get.get_file_path('train_evaluate_info/'))
# model_evaluates.sort_values(by=['chapter_id', 'model', 'num_fits'], ascending=[True, True, True], inplace=True)




# 解析模型评估文件
# model_evaluate_list = Get.get_file_path('./train_evaluate_info/')
# model_big_evaluate = model_evaluate(model_evaluate_list)
# model_big_evaluate_n = model_big_evaluate[model_big_evaluate['model'] == 'forgets_n'][['num_fits', 'auc']]
# model_big_evaluate_y = model_big_evaluate[model_big_evaluate['model'] == 'forgets_y'][['num_fits', 'auc']]
# plt.line(model_big_evaluate_n['num_fits'], model_big_evaluate_n['auc'])
# plt.line(model_big_evaluate_y['num_fits'], model_big_evaluate_y['auc'])


# 开启多线程，进行模型评估
# def main(p):
#     model = joblib.load(p)
#     evaluate_list = model.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error])
#     print({'paths': [str(p)], 'evaluate': evaluate_list, 'datetime': [datetime.datetime.now()]})
#     print(datetime.datetime.now(), '\n')
# pool = ThreadPool(8)
# results = pool.map(main, _model2)
# pool.close()
# pool.join()


