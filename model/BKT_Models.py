# @Author : 曲书贤
# @File : BKT_Models.py
# @CreateTime : 2021/6/2 17:05
# @Software : PyCharm
# @Comment : 说明脚本的用处


import joblib
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from pyBKT.models import Model
import pandas as pd
import numpy as np
import warnings
import datetime
from functools import reduce
from collections import defaultdict
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


"""
# default column names for assistments
as_default={'order_id': 'order_id',
             'skill_name': 'skill_name',
             'correct': 'correct',
             'user_id': 'user_id',
             'multilearn': 'template_id',
             'multiprior': 'correct',
             'multipair': 'template_id',
             'multigs': 'template_id',
             'folds': 'user_id',
             }

# default column names for cognitive tutors
ct_default={'order_id': 'Row',
            'skill_name': 'KC(Default)',
            'correct': 'Correct First Attempt',
            'user_id': 'Anon Student Id',
            'multilearn': 'Problem Name',
            'multiprior': 'Correct First Attempt',
            'multipair': 'Problem Name',
            'multigs': 'Problem Name',
            'folds': 'Anon Student Id',
            }

模型传参说明：
    seed = 随机种子，默认20094805
    num_fits = 默认5

模型拟合参数说明：
    skills = 是否针对不同的知识点进行拟合，类型为list
    multigs | multipair = 是否要针对不同的题目(Problem Name)训练数据，类型为bool
    forgets = 是否计算遗忘概率，类型为bool
    multilearn = 如果forgets=True的情况下设置multilearn=True该参数，表示知识点下满足计算结果后，未计算的题目仍然计算。同时可以指定user_id字段计算用户的遗忘概率
    multiprior = correct， 简单理解下来为就是讲真实的先验P（L0）设置为0

"""

# 获取数据
q_df = pd.read_csv('test/questions_chapter_top5.csv')

defaults = {'skill_name': 'chapter_id', 'correct': 'status', 'user_id': 'user_id', 'multigs': 'level', 'multilearn': 'level'}
# 对数据进行标准模型拟合
print(datetime.datetime.now(), '运行开始......')
start_time = datetime.datetime.now()

model = Model(seed=0, num_fits=10)
model.fit(data=q_df, defaults=defaults, multigs=True)
print("\n\n\033[1;30;47mmodel学习参数:\n\033[0m", model.params())
print('\n\n\033[1;30;47mmodel拟合效果评估:\n\033[0m')
print(model.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))
model_evaluate = model.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error])

model1 = Model(seed=0, num_fits=10)
model1.fit(data=q_df, defaults=defaults, multigs=True, forgets=True, multilearn=True)
print("\n\n\033[1;30;47mmodel1学习参数:\n\033[0m", model1.params())
print('\n\n\033[1;30;47mmodel1拟合效果评估:\n\033[0m')
print(model1.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))
model1_evaluate = model1.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error])

end_time = datetime.datetime.now()
print(datetime.datetime.now(), '运行结束，总耗时：', end_time-start_time)

# 保存模型
# joblib.dump(model, '2fits_prior_2.pkl')
#
#
# # 加载模型,并进行效果评估
# model_2fits = joblib.load('2fits.pkl')
# model_2fits_prior_2 = joblib.load('2fits_prior_2.pkl')
# model_2fits_prior_5 = joblib.load('2fits_prior_5.pkl')
# model_5fits = joblib.load('5fits.pkl')
# model_5fits_prior_5 = joblib.load('5fits_prior_5.pkl')
# print('model_2fits拟合效果评估:', model_2fits.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))
# print('model_2fits_prior_2拟合效果评估:', model_2fits_prior_2.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))
# print('model_2fits_prior_5拟合效果评估:', model_2fits_prior_5.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))
# print('model_5fits拟合效果评估:', model_5fits.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))
# print('model_5fits_prior_5拟合效果评估:', model_5fits_prior_5.evaluate(data=q_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))

"""
评估指标：["auc", "rmse", "accuracy", "mae"]

标准模型
2fits拟合效果评估:            [0.6137861209676047, 0.25724153311560033, 0.936235989393484, 0.1744524963648551]
2fits_prior=0.2拟合效果评估:  [0.6130683480409825, 0.25724896098093275, 0.9362365969521456, 0.17424458943350285]
2fits_prior=0.5拟合效果评估:  [0.6106234599594745, 0.25707299043618026, 0.936230339097931, 0.17280988293181612]
5fits拟合效果评估:            [0.6137242767897104, 0.25722944602222675, 0.9362366577080117, 0.17441369914773075]
5fits_prior=0.5拟合效果评估:  [0.613598929422136, 0.25723437586214104, 0.9362365969521456, 0.17434754866028648]

多重交叉+遗忘模型
2fits_seed=0拟合效果评估:                                 [0.593462650329493, 0.2699997862297355, 0.9358379169583961, 0.1824543150697113]
2fits_seed=0_forgets=True_multilearn=True拟合效果评估:    [0.6766493413831067, 0.24335751157461105, 0.9338934862177748, 0.08755568594826958]
"""



