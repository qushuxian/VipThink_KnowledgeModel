# @Author : 曲书贤
# @File : Initialize.py
# @CreateTime : 2021/5/13 11:23
# @Software : PyCharm
# @Comment : 用于数据的初始化

from sql import PostgreSQL
import datetime
import re


# 初始化数据的SQL
class_room_sql = """
select  to_char(to_timestamp(ol.start_time),'yyyy-mm-dd HH24:mi:ss') class_start_time, -- 上课时间
            to_char(to_timestamp(ol.end_time),'yyyy-mm-dd HH24:mi:ss') class_end_time, -- 下课时间
            to_char(to_timestamp(ol.create_time),'yyyy-mm-dd HH24:mi:ss') create_time, -- 数据写入时间 
            to_char(to_timestamp(ol.update_time),'yyyy-mm-dd HH24:mi:ss') update_time, -- 数据更新时间 
            ol.id live_id, -- 课程ID
            ls.student_id user_id, -- 用户ID
            cc2.name type_name, -- 课类名称
            ol.cate_pid, -- 课类ID
            cc.name step_name, --阶段名称
            dc.cn_number, -- 章节编号(讲次)
            case when split_part(dc.cn_number,'_',2) is null then 0
                when split_part(dc.cn_number,'_',2)='A' then 0
                else cast(split_part(dc.cn_number,'_',2) as int) end cn_numbers, -- 第几讲
            dc.game_url, -- 课件编码
            dc.chapter_id, -- 课件ID
            dc.chapter_name, -- 课件名称
            ol.course_type, -- 课类1:正课；2:试听课；3:活动课；6:补课
            ol.live_status, -- 上课状态 0=待上课，1=上课中，2=已下课 3=已取消
            ls.check_status, -- 考勤状态1:签到用户；2:请假用户；3:旷课用户；4:取消用户,该取消为老师主动取消不计算学生课次
            case when right(dc.game_url,1)='a' then 'A' when right(dc.game_url,1)='b' then 'B' else null end chapter_level -- 课件难度
    from odl_online.ol_live ol
    inner join odl_online.ol_live_student ls on ol.id=ls.live_id
    inner join( select t.*
                from(
                    select  admin_id,name,nickname,groupname1,groupname2,groupname3,groupname4
                            ,new_leave_time leave_time -- 离职时间
                            ,(groupname3||groupname4||groupname5||groupname6) adds
                            ,ROW_NUMBER() over(partition by admin_id order by create_time asc) num 
                    from bdl_online.dim_auth_admin_user -- 公司内部组织结构表，一个员工可能会对应多个部门
                    where   (groupname2 not like '%研发中心%' or groupname2 is null) -- 排除公司内部测试账号
                            and nickname not like '%测试%' -- 排除公司内部测试账号
                            and name not like '%测试%' -- 排除公司内部测试账号
                )t where t.num=1
        ) au on au.admin_id=ol.teacher_id
    inner join (select user_id,first_pay_time
                from adl_online.adl_tr_user_info -- user表
                where user_status in(0,1) and is_test_user=0
            ) u on ls.student_id=u.user_id
    inner join odl_online.ol_chapter_reb rb on ol.id=rb.live_id
    inner join bdl_online.dim_chapter dc on rb.slave_cn_id = dc.cn_id
    left join odl_online.ol_course_category cc2 on ol.cate_pid=cc2.id
    left join odl_online.ol_course_category cc on ol.cate_sid=cc.id
    left join adl_online.adl_tr_user_info fr on fr.user_id = ls.student_id
    where   ol.status = 1 
            and ol.live_status in (0,1,2)
            and ol.course_type in (1,6)
            and ls.check_status in (1,2,3)
            and ol.cate_pid=748 -- 仅选择数学思维V3课程
            and dc.chapter_id is not null
"""
chapter_class_online_work_sql = """
        select  ow.live_chapter_id, -- 直播课课件ID
                oc2.chapter_name live_chapter_name, -- 直播课课件名称,
                oc2.game_url live_game_url, -- 直播课课件编码
                oc2.difficulty live_difficulty, -- 直播课课件难度
    
                ow.chapter_id work_chapter_id, -- 在线作业题目对应的课件ID
                oc1.game_url work_game_url, -- 在线作业题目对应的课件编码
                ow.id work_id, -- 在线作业题目ID
                ow.status work_status, -- 在线作业题目是否禁用 -- 0为禁用，1为启用
                ow.difficulty work_difficulty, -- 在线作业题目难度
    
                wq.content_module_id content_module_id, -- 在线作业题目内容模块id
                cast(wq.knowledge_id as int) knowledge_id, -- 知识图谱 ID
                wq.level subject_level, -- 在线作业题目题号（关卡）与online_work_answer_log表的level保持一致
                wq.status subject_status, -- 0下架，1上架，2删除
                wq.type subject_type, -- 题目类型 0 普通题 1 挑战题 2 小老师作业
                wq.difficulty subject_difficulty -- 难度系数，1-10 
        from odl_online.ol_online_work ow
        left join odl_online.ol_chapter oc1 on oc1.id=ow.chapter_id
        left join odl_online.ol_chapter oc2 on oc2.id=ow.live_chapter_id
        left join realtime_jy_work.ol_ow_subject wq on wq.online_work_id=ow.id
        where ow.status=1 and wq.status=1
"""
chapter_class_online_answer_sql = """
    select  cast(knowledge_id as int) knowledge_id -- 知识点id
            ,chapter_id -- 课件id
            ,concat(case when stage>0 then stage-1 else 0 end,'_',
                    case when item>0 then item-1 else 0 end
                    ) new_knowledge_id -- 课件下的不同关卡ID,-1是因为直播间答题关卡和题号数据错位
            ,link_type new_knowledge_type -- 课件课件下的不同答题类型 5例题,6练习,9挑战题
            ,row_number() over(partition by chapter_id,link_type order by stage,item asc) new_knowledge_subject_level -- 不同题型的题目ID
            ,difficulty new_knowledge_difficulty -- 难度
    from realtime_jy_work.ol_ck_point
    where   status=1  -- 状态 1正常 0禁用（同一个chapter先有学生使用，但后期对chapter下的题更新了，第一次使用的会被禁用）
            and point_type=1  --1知识点 2课前预习(=2无数据)
            and level_type=3 -- 知识点类型 0无,1图文,2动画3题目,4游戏
            and link_type in(5,6,9) -- 课件环节 0无,1预习,2封面,3开场,4小游戏,5例题,6练习,7过场,8神秘任务,9挑战题,10结尾,11总结,12思维导图,13小老师任务
            -- 增量字段，datatime
            -- and create_time
            -- and update_time
    order by chapter_id,new_knowledge_type,new_knowledge_subject_level
"""
knowledge_config_sql = """
select  id knowledge_id -- 知识点ID
        ,pid knowledge_pid -- 父级id
        ,name knowledge_name -- 知识点名称
        ,level knowledge_level -- 知识点层级
        ,tree knowledge_tree -- 知识点树结构
        ,tags knowledge_tags-- 知识点标签
from realtime_jy_work.ol_knowledge_config -- 知识点配置表
where subject_type=0 --数学,1为语文
        and isdeleted=1 --是否删除,2为删除，注：可能存在用后再删除的情况
        and id not in (61,1515,1522)
        and split_part(tree,',',1) not in ('61','1515','1522')
"""
knowledge_should_questions_sql = """
    select distinct a.*
    from(
        -- 实际上课数据剔除V3版本42讲之前的课程
        -- 实际上课数据剔除chapter_id拆分为多节课且课程签到数据
        -- 课件和课后作业关联，获得用户课件对应的应答题数量
        -- 课后作业和知识图谱关联，获得用户在某个知识图谱（末级）下的应答题数量
        select  0 as questions_type -- 知识点题目应用场景，1课中答题、0课后答题
                ,cr.chapter_id -- 课件id
                ,ow.knowledge_id -- 末级知识点ID
                ,kc.knowledge_name -- 末级知识点名称
                ,kc.knowledge_level -- 末级知识点层级
                ,kc.knowledge_tree -- 末级知识点的上层树结构
                ,kc.knowledge_pid -- 末级知识点的父级id
                ,kc.knowledge_tags -- 末级知识点标签
                -- ,ow.work_id knowledge_subject_id -- 末级知识点下的题目ID
                ,ow.subject_level knowledge_subject_level -- 末级知识点下的题目题号（关卡）
                ,ow.subject_type knowledge_subject_type -- 题目类型 0 普通题 1 挑战题 2 小老师作业
                ,ow.subject_difficulty knowledge_subject_difficulty -- 末级知识点下的题目难度系数，1-10
        from(
            select  distinct chapter_id,count(live_id) over (partition by user_id,chapter_id) check_in_cnt
            from ods.class_room
            where course_type in(1,6) and live_status=2 and check_status=1 and cn_numbers>=42
        ) cr
        inner join ods.chapter_class_online_work ow on cr.chapter_id=ow.live_chapter_id
        inner join ods.knowledge_config kc on ow.knowledge_id=kc.knowledge_id
        where cr.check_in_cnt=1
        union all
        -- 新课件知识点（V3-42讲之前的使用两级知识点，42讲之后用的新知识图谱包含五级知识点）
        select  1 as questions_type -- 知识点题目应用场景，1课中答题、0课后答题
                ,cr.chapter_id -- 课件id
                ,coa.knowledge_id -- 末级知识点ID
                ,kc.knowledge_name -- 末级知识点名称
                ,kc.knowledge_level -- 末级知识点层级
                ,kc.knowledge_tree -- 末级知识点的上层树结构
                ,kc.knowledge_pid -- 末级知识点的父级id
                ,kc.knowledge_tags -- 末级知识点标签
                -- ,coa.new_knowledge_id knowledge_subject_id -- 末级知识点下的题目ID
                ,coa.new_knowledge_subject_level knowledge_subject_level -- 末级知识点下的题目题号（关卡）
                ,coa.new_knowledge_type knowledge_subject_type -- 末级知识点下的题目类型 5例题,6练习,9挑战题
                ,coa.new_knowledge_difficulty knowledge_subject_difficulty -- 末级知识点下的题目难度系数，1-10
        from(
            select  distinct chapter_id,count(live_id) over (partition by user_id,chapter_id) check_in_cnt
            from ods.class_room
            where course_type in(1,6) and live_status=2 and check_status=1 and cn_numbers>=42
        ) cr
        inner join ods.chapter_class_online_answer coa on cr.chapter_id=coa.chapter_id
        inner join ods.knowledge_config kc on coa.knowledge_id=kc.knowledge_id
        where cr.check_in_cnt=1
    ) a
    order by chapter_id,questions_type,knowledge_id,knowledge_subject_type,knowledge_subject_level
"""
questions_room_sql = """
    select  create_time -- 关卡完成答题上报时间
            ,user_id -- 学员id
            ,chapter_id -- 课件id
            ,level -- 关卡ID，对应课中应答题表new_knowledge_id字段
            ,status  -- 答题状态-- 1是答对 2是答错 3是跳过
            ,ave_time -- 答题时长
            ,ave_level_time -- 关卡时长
    from (
        select  to_char(to_timestamp(e.finish_event_time/1000),'yyyy-mm-dd HH24:mi:ss') create_time -- 关卡完成答题上报时间
                ,e.user_id,cast(e.courseware_id as int) chapter_id,e.level,e.status
                ,case when e.ave_time is null then 0 else  e.ave_time end ave_time
                ,case when e.ave_level_time is null then 0 else  e.ave_level_time end ave_level_time
                ,row_number()over(partition by e.courseware_id,e.level,e.user_id order by e.finish_time) rank 
        from adl_online.adl_sensors_study_info_mission e
        inner join(
                select  ls.student_id user_id,cast(dc.chapter_id as varchar) chapter_id
                from odl_online.ol_live ol
                inner join odl_online.ol_live_student ls on ol.id=ls.live_id
                inner join odl_online.ol_chapter_reb rb on ol.id=rb.live_id
                inner join bdl_online.dim_chapter dc on rb.slave_cn_id = dc.cn_id
                where   ol.status=1 -- 是否启用
                        and ol.live_status in (2)
                        and ol.course_type in (1,6)
                        and ls.check_status in (1)
                        and ol.cate_pid=748
                        and dc.chapter_id is not null
                group by 1,2
            ) live on e.user_id=live.user_id and e.courseware_id=live.chapter_id
        where e.study_stage=2 -- 学习环节,课中答题
            and e.status in(1,2) -- 答题状态 1是答对 2是答错 3是跳过（默认是老师点击切关跳过）
    ) sensors
    where rank = 1 -- 首次答题
"""
questions_online_work_sql = """
    select create_time,user_id,live_id,question_id,level,answer_img_id,question_status,question_time
    from(
        select  to_char(to_timestamp(owa.create_time),'yyyy-mm-dd HH24:mi:ss') create_time -- 数据写入时间 
                ,owa.user_id
                ,owa.live_id -- 直播ID(live表自增ID)
                ,owa.question_id -- 作业关联题目ID
                ,owa.level -- 题号（关卡）与online_work_answer_log表的level保持一致
                ,owa.answer_img_id -- 答题截图图片ID(admin_attachment 表自增ID）
                ,owa.status question_status -- 答题状态 0=错，1=对
                ,owa.time question_time -- 答题完成时长
                ,row_number() over (partition by owa.user_id,owa.live_id,owa.question_id order by owa.create_time) ascs
        from odl_online.ol_online_work_answer_log owa -- 线上作业答题记录
        inner join(
                select  ls.student_id user_id,ls.live_id
                from odl_online.ol_live ol
                inner join odl_online.ol_live_student ls on ol.id=ls.live_id
                inner join odl_online.ol_chapter_reb rb on ol.id=rb.live_id
                inner join bdl_online.dim_chapter dc on rb.slave_cn_id = dc.cn_id
                where   ol.status=1 -- 是否启用
                        and ol.live_status in (2)
                        and ol.course_type in (1,6)
                        and ls.check_status in (1)
                        and ol.cate_pid=748
                        and dc.chapter_id is not null
                group by 1,2
            ) live on owa.user_id=live.user_id and owa.live_id=live.live_id
        where   owa.live_id>0 and owa.question_id>0
                -- and owa.create_time -- 增量字段,时间戳
        order by user_id,live_id,level
    ) kh
    where kh.ascs=1 -- question_id取首次答题结果
"""
subject_testing_sql = """
    select  t.user_id,t.eva_id,t.eva_code,t.eva_name,t.knowledge_id,t.question_degree
            ,t.question_create_time first_question_time,t.knowledge_qa_cnt,t.knowledge_qa_success_cnt
            ,case when t.knowledge_qa_cnt=t.knowledge_qa_success_cnt and t.knowledge_qa_cnt>0 then 1 else 0 end knowledge_is_grasp
    from(
        select  a.user_id
                ,a.eva_id -- 测评课件ID
                ,b.code eva_code -- 课件码
                ,b.name  eva_name -- 测评课件名称
                ,acq.knowledge_id -- 测评的知识点ID
                ,acq.difficulty_degree question_degree -- 测评难度系数，1-10
                ,aca.question_index -- 测评答题题号
                ,to_char(to_timestamp(aca.create_time),'yyyy-mm-dd HH24:mi:ss') question_create_time -- 测评答题题号创建时间
                ,aca.answer question_answer -- 测评答题是否正确 
                ,aca.time_long question_time_long -- 测评答题时长
                ,count(1) over (partition by a.user_id,a.eva_id,acq.knowledge_id) knowledge_qa_cnt -- 专题测评末级知识点测评题目数量
                ,sum(aca.answer) over (partition by a.user_id,a.eva_id,acq.knowledge_id) knowledge_qa_success_cnt -- 专题测评末级知识点测评题目数量
                ,row_number() over (partition by a.user_id,a.eva_id,acq.knowledge_id order by aca.question_index) testing_number -- 专题测评末级知识点测评顺序
        from bdl_online.fact_assign_class_eva_log a -- 评测完成情况事实表
        inner join odl_online.ol_assign_class_answer aca on aca.eva_id=a.eva_id and aca.user_id=a.user_id -- 评测模块2.0-答题表-使用中
        inner join odl_online.ol_assign_class_question acq on aca.question_id=acq.id -- 测评模块2.0-问题表-使用中
        inner join (select cast(id as varchar) as knowledge_id
                    from realtime_jy_work.ol_knowledge_config -- 知识点配置表
                    where subject_type=0 --数学,1为语文
                            and isdeleted=1 --是否删除,2为删除，注：可能存在用后再删除的情况
                            and id not in (61,1515,1522)
                            and split_part(tree,',',1) not in ('61','1515','1522')
                            and id>0
                    group by 1
                ) kc on acq.knowledge_id=kc.knowledge_id
        left join bdl_online.fact_assign_class_evaluation b on a.eva_id=b.eva_id  
        where   a.status in(0,1,2) --'状态：0下架，1上架，2删除'
                and a.category_id=2 -- 测评分类ID 1=分班测评；2=专题测评；3=结营测评；4=定级测评；5=阶段测评；6=直播结营测评
                and acq.status=1 -- 测评状态 1=完成；0=未完成
    ) t
    where t.testing_number=1
    order by user_id,eva_id,knowledge_id
"""


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


