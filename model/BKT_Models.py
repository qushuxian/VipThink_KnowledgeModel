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



