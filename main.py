# @Author : 曲书贤
# @File : main.py
# @CreateTime : 2021/5/20 15:04
# @Software : PyCharm
# @Comment : 项目运行


"""
http://wangxin123.com/2019/05/21/论文阅读/知识追踪模型调研/
https://www.163.com/dy/article/ELJ8TI1Q0516QHFP.html
https://www.cnblogs.com/GuoJiaSheng/p/7099724.html
https://www.cnblogs.com/davidwang456/articles/8926997.html
"""
from etl.Incremental import IncrementalData
from Transform import DateTime


# 获取时间
start_date, end_date, start_unix, end_unix = DateTime().days(int(-7))

# 增量同步
icm_dt = IncrementalData(start_date, end_date, start_unix, end_unix)
icm_dt.class_room(db_table_name='ods.class_room', is_local=True)
icm_dt.chapter_class_online_work(db_table_name='ods.chapter_class_online_work', is_local=True)
icm_dt.chapter_class_online_answer(db_table_name='ods.chapter_class_online_answer', is_local=True)
icm_dt.knowledge_config(db_table_name='ods.knowledge_config', is_local=True)
icm_dt.knowledge_should_questions(db_table_name='dwd.knowledge_should_questions', is_local=True)
icm_dt.questions_room(db_table_name='ods.questions_room', is_local=True,
                      incremental_unix_start=int(str(start_unix)+'000'),
                      incremental_unix_end=int(str(end_unix)+'000'))
icm_dt.questions_online_work(db_table_name='ods.questions_online_work', is_local=True)

# 获取待预测数据


# 结果预测


# 结果写入

