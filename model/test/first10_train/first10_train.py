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
first10_df = import_excel('/Users/vipthink/Downloads/first10/', 'csv')
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
    print('=============================================', fits)
    # ###########################拟合标准模型
    model_start_time = datetime.datetime.now()
    print('model1 is running......', datetime.datetime.now())
    forgets_n = Model(seed=0, num_fits=int(fits), print=False)
    forgets_n.fit(data=first10_df, defaults=defaults, print=False)
    evaluate_forgets_n.append({'chapter': ['model1'], 'chapter_data_cnt': [first10_df.shape[0]],
                               'fits': [fits], 'model': ['forgets_n'],
                               'metric': forgets_n.evaluate(data=first10_df,
                                                            metric=["auc", "rmse", "accuracy", mean_absolute_error])})
    model_end_time = datetime.datetime.now()
    print('model1 is end，time:', model_end_time-model_start_time)

    # ###########################拟合标准模型_知识点按用户生成表现参数（guess、slip）
    model_start_time1 = datetime.datetime.now()
    print('\n model2 is running......', datetime.datetime.now())
    forgets_n_multigs = Model(seed=0, num_fits=int(fits), print=False)
    forgets_n_multigs.fit(data=first10_df, defaults=defaults, multilearn='user_id', print=False)
    evaluate_forgets_n_multigs.append({'chapter': ['model2'], 'chapter_data_cnt': [first10_df.shape[0]],
                                       'fits': [fits], 'model': ['forgets_n_multigs'],
                                       'metric': forgets_n_multigs.evaluate(data=first10_df,
                                                                            metric=["auc", "rmse", "accuracy",
                                                                                    mean_absolute_error])})
    model_end_time1 = datetime.datetime.now()
    print('model2 is end，time:', model_end_time1 - model_start_time1)

    # ###########################拟合遗忘模型one
    model2_start_time = datetime.datetime.now()
    print('\n model3 is running......', datetime.datetime.now())
    forgets_y_one = Model(seed=0, num_fits=int(fits), print=False)
    forgets_y_one.fit(data=first10_df, defaults=defaults, forgets=True, print=False)
    evaluate_forgets_y_one.append({'chapter': ['model3'], 'chapter_data_cnt': [first10_df.shape[0]],
                                   'fits': [fits], 'model': ['forgets_y_one'],
                                   'metric': forgets_y_one.evaluate(data=first10_df,
                                                                    metric=["auc", "rmse", "accuracy",
                                                                            mean_absolute_error])})
    model2_end_time = datetime.datetime.now()
    print('model3 is end，time:', model2_end_time - model2_start_time)

    # ###########################拟合遗忘模型two
    model3_start_time = datetime.datetime.now()
    print('\n model4 is running......', datetime.datetime.now())
    forgets_y_two = Model(seed=0, num_fits=int(fits), print=False)
    forgets_y_two.fit(data=first10_df, defaults=defaults, forgets=True, multigs=True, print=False)
    evaluate_forgets_y_two.append({'chapter': ['model4'], 'chapter_data_cnt': [first10_df.shape[0]],
                                   'fits': [fits], 'model': ['forgets_y'],
                                   'metric': forgets_y_two.evaluate(data=first10_df,
                                                                    metric=["auc", "rmse", "accuracy",
                                                                            mean_absolute_error])})
    model3_end_time = datetime.datetime.now()
    print('model4 is end，time:', model3_end_time - model3_start_time)

    # ###########################拟合遗忘模型three
    model4_start_time = datetime.datetime.now()
    print('\n model5 is running......', datetime.datetime.now())
    forgets_y_three = Model(seed=0, num_fits=int(fits), print=False)
    forgets_y_three.fit(data=first10_df, defaults=defaults, forgets=True, multigs=True, print=False)
    evaluate_forgets_y_three.append({'chapter': ['model5'], 'chapter_data_cnt': [first10_df.shape[0]],
                                     'fits': [fits], 'model': ['forgets_y'],
                                     'metric': forgets_y_three.evaluate(data=first10_df,
                                                                        metric=["auc", "rmse", "accuracy",
                                                                                mean_absolute_error])})
    model4_end_time = datetime.datetime.now()
    print('model5 is end，time:', model4_end_time - model4_start_time)

    # ###########################拟合遗忘模型four
    model5_start_time = datetime.datetime.now()
    print('\n model6 is running......', datetime.datetime.now())
    forgets_y_four = Model(seed=0, num_fits=int(fits), print=False)
    forgets_y_four.fit(data=first10_df, defaults=defaults, forgets=True, multigs=True, multilearn=True, print=False)
    evaluate_forgets_y_four.append(
        {'chapter': ['model6'], 'chapter_data_cnt': [first10_df.shape[0]], 'fits': [fits], 'model': ['forgets_y'],
         'metric': forgets_y_four.evaluate(data=first10_df, metric=["auc", "rmse", "accuracy", mean_absolute_error])})
    model5_end_time = datetime.datetime.now()
    print('model6 is end，time:', model5_end_time - model5_start_time)

    # ###########################拟合遗忘模型five
    model6_start_time = datetime.datetime.now()
    print('\n model7 is running......', datetime.datetime.now())
    forgets_y_five = Model(seed=0, num_fits=int(fits), print=False)
    forgets_y_five.fit(data=first10_df, defaults=defaults, forgets=True, multigs=True, multilearn='user_id', print=False)
    evaluate_forgets_y_five.append(
        {'chapter': ['model7'], 'chapter_data_cnt': [first10_df.shape[0]], 'fits': [fits], 'model': ['forgets_y'],
         'metric': forgets_y_five.evaluate(data=first10_df,
                                           metric=["auc", "rmse", "accuracy", mean_absolute_error])})
    model6_end_time = datetime.datetime.now()
    print('model7 is end，time:', model6_end_time - model6_start_time)
    print('')

# 每个模型拟合结束后，将评估结果保存
write('first10_model1.txt', evaluate_forgets_n)
write('first10_model2.txt', evaluate_forgets_n_multigs)
write('first10_model3.txt', evaluate_forgets_y_one)
write('first10_model4.txt', evaluate_forgets_y_two)
write('first10_model5.txt', evaluate_forgets_y_three)
write('first10_model6.txt', evaluate_forgets_y_four)
write('first10_model7.txt', evaluate_forgets_y_five)


# 将多个model_evaluate评估结果合并
# model_evaluates = model_evaluate(Get.get_file_path('aa/'))
# model_evaluates.sort_values(by=['chapter_id', 'model', 'num_fits'], ascending=[True, True, True], inplace=True)

