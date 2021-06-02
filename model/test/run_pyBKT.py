# @Author : 曲书贤
# @File : run_pyBKT.py
# @CreateTime : 2021/5/20 16:58
# @Software : PyCharm
# @Comment : pyBKT实验
import joblib
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from model.pyBKT.models import Model
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


def rolling(data, rolling_index, index_drop=True):
    order_count = defaultdict(int)

    def rolling_number(column_name):
        order_count[column_name] += 1
        return order_count[column_name]
    data['row_number'] = data.apply(lambda x: rolling_number(x[rolling_index]), axis=1)
    if index_drop:
        data.drop([rolling_index], axis=1, inplace=True)
    return data


def group(data, group_list, agg_dict=None, group_rename=None, reset_index=True, merge=True):
    """ DataFrame.group """
    if not isinstance(group_list, list) and not isinstance(data, pd.DataFrame):
        raise ValueError("no data specified")
    else:
        group_dt = data.groupby(group_list).agg(agg_dict)
        for i in group_list:
            group_dt[i] = group_dt.index.get_level_values(i)
        group_dt.rename(columns=group_rename, inplace=True)
        if reset_index:
            group_dt.reset_index(drop=True, inplace=True)
        if merge:
            group_dt = reduce(lambda left, right: pd.merge(left, right, how='left', on=group_list), [data, group_dt])
    return group_dt


def state_sift(data, group_list, seed=0.85):
    data['sift_number'] = data.apply(lambda x: x.row_number if x.state_predictions >= seed else 0, axis=1)
    group_dt = data.groupby(group_list, as_index=False).apply(lambda t: t[t.sift_number > 0].min())
    for i in group_list:
        group_dt[i] = group_dt.index.get_level_values(i)
    group_dt.reset_index(drop=True, inplace=True)
    group_list.extend(['state_predictions', 'questions_cnt', 'sift_number'])
    print(group_list)
    group_dt = group_dt[group_list]
    return group_dt


# 获取数据
ct_df = pd.read_excel('ct_df.xlsx')
ct_df_1kc = ct_df.loc[ct_df['KC(Default)'] == 'Calculate unit rate']


# 对数据进行标准模型拟合
defaults = {'skill_name': 'chapter_id', 'correct': 'status', 'user_id': 'user_id', 'multigs': 'gs_classes'}
model = Model(num_fits=5)
# model.coef_ = {'Calculate unit rate': {'prior': 0.5}}
model.fit(data=ct_df, defaults=defaults)
print("\n\n\033[1;30;47mmodel学习参数:\n\033[0m", model.params())
print('\n\n\033[1;30;47mmodel拟合效果评估:\n\033[0m')
print(model.evaluate(data=ct_df, metric=["auc", "rmse", "accuracy", mean_absolute_error]))

# print('\n\n\033[1;30;47mmodel对数据进行预测:\n\033[0m')
# model.predict(data=ct_df)
#
# # 保存模型，加载模型
# joblib.dump(model, 'bkt.pkl')
# model = joblib.load('bkt.pkl')


# # 统计用户知识点下实际答题数量,计算答题序列号
# ct_df_1kc = group(ct_df_1kc, ['Anon Student Id', 'KC(Default)'], {'Row': 'count'}, {'Row': 'questions_cnt'})
# ct_df_1kc['rolling_index'] = ct_df_1kc["Anon Student Id"].map(str) + ct_df_1kc["KC(Default)"]
# ct_df_1kc = rolling(ct_df_1kc, 'rolling_index')
#
#
# # 求解用户知识点的掌握情况
# ct_df_1kc_sift = state_sift(ct_df_1kc, ['Anon Student Id', 'KC(Default)'], seed=0.7)





