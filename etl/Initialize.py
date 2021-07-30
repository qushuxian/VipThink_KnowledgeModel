# @Author : 曲书贤
# @File : Initialize.py
# @CreateTime : 2021/5/13 11:23
# @Software : PyCharm
# @Comment : 用于数据的初始化

from sql import PostgreSQL
import datetime
import re
from config import class_room_sql, chapter_class_online_work_sql, chapter_class_online_answer_sql, knowledge_config_sql, \
    knowledge_should_questions_sql, questions_room_sql, questions_online_work_sql, subject_testing_sql


# 查询数据，并将数据写入数据库，请主要创建表结构
print('开始查询......', datetime.datetime.now())
run_start_time = datetime.datetime.now()
df = PostgreSQL().select(subject_testing_sql, is_local=False)
# df['knowledge_name'] = [re.sub('\n', '', name, 1) for name in df['knowledge_name']]
run_end_time = datetime.datetime.now()
print('查询成功', datetime.datetime.now(), '总耗时：', run_end_time - run_start_time, '数据量：', df.shape)

insert_start_time = datetime.datetime.now()
print('\n开始写入数据库......', datetime.datetime.now())
insert_states = PostgreSQL().insert(df, table_name='ods.subject_testing', is_local=True)
insert_end_time = datetime.datetime.now()
if insert_states == 1:
    print('写入成功', datetime.datetime.now(), '总耗时：', insert_end_time - insert_start_time)
else:
    print('写入失败，请检查！！！！！！！！！！！！！！')


