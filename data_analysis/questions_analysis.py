# @Author : 曲书贤
# @File : questions_analysis.py
# @CreateTime : 2021/5/17 16:31
# @Software : PyCharm
# @Comment : 签到用户实际答题探索分析

from sql import PostgreSQL
from functools import reduce
import pandas as pd


def _class_questions(q_df, c_df):
    c_df['user_id'] = c_df['user_id'].astype('str')
    c_df['chapter_id'] = c_df['chapter_id'].astype('str')
    q_df['user_id'] = q_df['user_id'].astype('str')
    q_df['chapter_id'] = q_df['chapter_id'].astype('str')
    q_df['status'] = q_df['status'].astype('str')
    df_group = q_df.groupby(['user_id', 'chapter_id']).agg(
        {'status': lambda x: ','.join(x), 'level': 'count', 'create_time_min': 'min', 'create_time_max': 'max'})
    df_group_types = q_df.groupby(['user_id', 'chapter_id', 'types']).agg(
        {'status': lambda x: ','.join(x), 'level': 'count', 'create_time_min': 'min', 'create_time_max': 'max'})

    df_group['user_id'] = df_group.index.get_level_values('user_id')
    df_group['chapter_id'] = df_group.index.get_level_values('chapter_id')
    df_group.rename(columns={'level': 'questions_cnt'}, inplace=True)
    df_group.reset_index(drop=True, inplace=True)

    df_group_types['user_id'] = df_group_types.index.get_level_values('user_id')
    df_group_types['chapter_id'] = df_group_types.index.get_level_values('chapter_id')
    df_group_types['types'] = df_group_types.index.get_level_values('types')
    df_group_types.rename(columns={'level': 'questions_cnt'}, inplace=True)
    df_group_types.reset_index(drop=True, inplace=True)

    df_group_types_qr = df_group_types[df_group_types['types'] == 'qr'].copy()
    df_group_types_qr.rename(columns={'status': 'qr_status', 'questions_cnt': 'qr_questions_cnt',
                                      'create_time_min': 'qr_create_time_min', 'create_time_max': 'qr_create_time_max'},
                             inplace=True)
    df_group_types_qr.drop(['types'], axis=1, inplace=True)
    df_group_types_qw = df_group_types[df_group_types['types'] == 'qw'].copy()
    df_group_types_qw.rename(columns={'status': 'qw_status', 'questions_cnt': 'qw_questions_cnt',
                                      'create_time_min': 'qw_create_time_min', 'create_time_max': 'qw_create_time_max'},
                             inplace=True)
    df_group_types_qw.drop(['types'], axis=1, inplace=True)

    crq_df = reduce(lambda left, right: pd.merge(left, right, how='left', on=['user_id', 'chapter_id']),
                    [c_df, df_group, df_group_types_qr, df_group_types_qw])
    crq_df['user_id'] = crq_df['user_id'].astype('int')
    crq_df['chapter_id'] = crq_df['chapter_id'].astype('int')
    return crq_df


sql_cr = """
    select user_id,live_id,chapter_id,class_start_time,cn_numbers
    from(
        select  cr.user_id
                ,cr.class_start_time
                ,cr.cn_numbers
                ,cr.live_id
                ,cr.chapter_id
            ,count(live_id) over (partition by cr.user_id,cr.chapter_id) check_num
        from ods.class_room cr
        where   cr.course_type in(1,6) 
                and cr.live_status=2 
                and cr.check_status=1 
                and cr.cn_numbers>=42
    )a
    where a.check_num=1 -- 剔除一个知识点上课两节及以上
"""
sql_ksq = """
    select  chapter_id,knowledge_id,knowledge_name,knowledge_level,knowledge_tree
            ,count(case when questions_type=1 then knowledge_subject_level end) kz_cnt
            ,count(case when questions_type=0 then knowledge_subject_level end) kh_cnt
    from dwd.knowledge_should_questions
    group by chapter_id,knowledge_id,knowledge_name,knowledge_level,knowledge_tree
"""
sql_questions = """
    select 'qr' as types,a.user_id,a.chapter_id,a.level,a.status,a.create_time
    from ods.questions_room a
    inner join ods.class_room b on a.user_id=b.user_id and a.chapter_id=b.chapter_id
    where   b.course_type in(1,6) 
            and b.live_status=2 
            and b.check_status=1 
            and b.cn_numbers>=42
    union all
    select 'qw' as types,a.user_id,b.chapter_id,cast(a.level as varchar) levels,a.question_status status,a.create_time
    from ods.questions_online_work a
    inner join ods.class_room b on a.user_id=b.user_id and a.live_id=b.live_id
    where   b.course_type in(1,6) 
        and b.live_status=2 
        and b.check_status=1 
        and b.cn_numbers>=42
        and a.question_status in(0,1)
"""
sql_testing = """
    select user_id,knowledge_id,knowledge_is_grasp,testing_cnt,testing_success_cnt
    from(
    select 	a.user_id,a.knowledge_id,a.knowledge_is_grasp,a.knowledge_qa_cnt testing_cnt,a.knowledge_qa_success_cnt testing_success_cnt
            ,count(1) over (partition by a.user_id,a.knowledge_id order by a.first_question_time) knowledge_qa_asc
    from ods.subject_testing a
    )testing
    where testing.knowledge_qa_asc=1
"""
sql_list = [sql_cr, sql_ksq, sql_questions, sql_testing]

# 获取数据并进行数据排序
cr_df, ksq_df, questions_df, testing_df = (PostgreSQL().select(i, is_local=True) for i in sql_list)
questions_df["status"] = questions_df["status"].map({2: 0, 1: 1, 0: 0})
questions_df.sort_values(by=['user_id', 'chapter_id', 'create_time'], ascending=[True, True, True], inplace=True)
questions_df.reset_index(drop=True, inplace=True)
questions_df["create_time_max"] = questions_df.create_time
questions_df.rename(columns={'create_time': 'create_time_min'}, inplace=True)

# 处理答题数据并匹配排课数据和知识图谱数据
class_room_questions = _class_questions(questions_df, cr_df)
class_room_questions = pd.merge(class_room_questions,
                                pd.DataFrame(ksq_df['chapter_id'].unique(), columns=['chapter_id']),
                                how='inner', on='chapter_id')

# 再匹配专题测评数据
# class_room_questions = pd.merge(class_room_questions, ksq_df[['chapter_id', 'knowledge_id']],
#                                 how='left', on='chapter_id')
class_room_questions = pd.merge(class_room_questions, testing_df, how='left', on=['user_id', 'knowledge_id'])

# class_room_questions表字段说明
"""
字段	                    说明
user_id	                用户id
live_id	                直播课id
chapter_id	            课件id
class_start_time	    开始上课日期
cn_numbers	            讲次

status	                课中+课后实际答题数据数据，按答题时间顺序排列
questions_cnt	        课中+课后实际答题总量
create_time_min	        课中+课后实际答题第一题答题时间
create_time_max	        课中+课后实际答题最后一题答题时间
qr_status	            课中实际答题数据数据，按答题时间顺序排列
qr_questions_cnt	    课中实际答题总量
qr_create_time_min	    课中实际答题第一题答题时间
qr_create_time_max	    课中实际答题最后一题答题时间
qw_status	            课后实际答题数据数据，按答题时间顺序排列
qw_questions_cnt	    课后实际答题总量
qw_create_time_min	    课后实际答题第一题答题时间
qw_create_time_max	    课后实际答题最后一题答题时间

knowledge_id	        直播课对应的末级知识点id
knowledge_is_grasp	    直播课对应的末级知识点专题测评是否全部答对
testing_cnt	            直播课对应的末级知识点专题测评答题总量
testing_success_cnt	    直播课对应的末级知识点专题测评答题正确总量
"""

# 检查一下知识图谱中，chapter_id是否有对应2个及以上的knowledge_id
ksq_dfs = pd.DataFrame(ksq_df.groupby(['chapter_id', 'knowledge_id'])['knowledge_id'].nunique())
ksq_dfs.rename(columns={'knowledge_id': 'knowledge_cnt'}, inplace=True)
ksq_dfs['chapter_id'] = ksq_dfs.index.get_level_values('chapter_id')
ksq_dfs['knowledge_id'] = ksq_dfs.index.get_level_values('knowledge_id')
ksq_dfs.reset_index(drop=True, inplace=True)
if ksq_dfs[ksq_dfs['knowledge_cnt'] > 1].shape[0] > 0:
    print('发现chapter_id对应2个及以上的knowledge_id，请检查！')
    print(ksq_dfs[ksq_dfs['knowledge_cnt'] > 0].to_dict(orient='records'))
else:
    pass

# 保存数据到本地的excel文件
class_room_questions.to_csv('用户实际答题数据.csv')
ksq_df.to_csv('知识点图谱及知识点课中和课后应答题数据.csv')


# # 探索不同末级知识点下实际答题数量的用户数
# class_room_questions = pd.read_csv('用户实际答题数据.csv')
# # ksq_df = pd.read_csv('知识点图谱及知识点课中和课后应答题数据.csv')
# class_room_questions['status'] = class_room_questions['status'].str.replace(',', "")
# class_room_questions['status_cnt'] = class_room_questions['status'].str.len()
#
# questions_status_cnt = class_room_questions.groupby(['chapter_id', 'status_cnt']).agg({'user_id': 'count'})
# questions_status_cnt.rename(columns={'user_id': '实际答题人次'}, inplace=True)
# questions_status_cnt['chapter_id'] = questions_status_cnt.index.get_level_values('chapter_id')
# questions_status_cnt['status_cnt'] = questions_status_cnt.index.get_level_values('status_cnt')
# questions_status_cnt.reset_index(drop=True, inplace=True)
# questions_status_cnt.rename(columns={'chapter_id': '末级知识点名称', 'status_cnt': '答题数量'}, inplace=True)
