# @Author : 曲书贤
# @File : knowledge_atlas.py
# @CreateTime : 2021/5/17 15:29
# @Software : PyCharm
# @Comment : 说明脚本的用处

from sql import PostgreSQL
import pandas as pd
import re


def regexp_split_to_table(dict_list: list, columns: str):
    """
    :param dict_list:
    :param columns: 多个值拆解多变量名称前戳
    :return:
    对列表中的字典进行，如果列表变量长度不一样，自动创建列名称
    """
    tree_values = []
    tree_columns = []
    df = []

    for i in dict_list:
        tree_values.append([v for v in i.values()])
        tree_columns.append([columns + '_' + str(k + 1) for k in range(len(i.keys()))])

    for v, c in zip(tree_values, tree_columns):
        dfs = pd.DataFrame(data=pd.DataFrame(v).T)
        dfs.columns = c
        df.append(dfs)
    df = pd.concat(df, axis=0)
    return df


def knowledge_join(join_data=None, from_df_tree=None, from_df_tags=None):
    """
    :param join_data: 知识结构要匹配的数据框
    :param from_df_tree: 知识点维度表
    :param from_df_tags: 题库标签表
    :return: tree_df, tags_df
    知识点和标签匹配
    """
    from_data_clm = join_data.columns.to_list()
    tree_dfs = []
    tags_dfs = []
    for f in from_data_clm:
        if f[0:4] == 'tree':
            tree_merge_df = pd.merge(join_data[f], from_df_tree, left_on=f, right_on='knowledge_id', how='left')
            tree_merge_df = pd.DataFrame(tree_merge_df['knowledge_name'])
            tree_merge_df.rename(columns={"knowledge_name": f + "_name"},  inplace=True)
            tree_dfs.append(tree_merge_df)
        elif f[0:4] == 'tags':
            tags_merge_df = pd.merge(join_data[f], from_df_tags, left_on=f, right_on='tags_id', how='left')
            tags_merge_df = pd.DataFrame(tags_merge_df['tags_name'])
            tags_merge_df.rename(columns={"tags_name": f + "_name"}, inplace=True)
            tags_dfs.append(tags_merge_df)
    tree_df = pd.concat(tree_dfs, axis=1)
    tags_df = pd.concat(tags_dfs, axis=1)
    return tree_df, tags_df


# ====================================  获取原始数据
# 知识点配置表
config_sql = """
    select 	id knowledge_id -- 知识点ID
            ,name knowledge_name -- 知识点名称
            ,(level+1) knowledge_level -- 知识点层级数量
            ,case when level=0 then cast(id as varchar) else tree end as tree -- 知识点树结构
            ,tags -- 知识点标签
            ,create_time
            ,update_time
    from realtime_jy_work.ol_knowledge_config -- 知识点配置表
    where subject_type=0 --数学,1为语文
            and isdeleted=1 --是否删除,2为删除，注：可能存在用后再删除的情况
            and id not in (61,1515,1522)
            and split_part(tree,',',1) not in ('61','1515','1522')
"""
config_data = PostgreSQL().select(config_sql)

# 题库标签表
tags_sql = """
    select 	id tags_id
            ,name tags_name
            ,(level+1) tags_level
            ,create_time
            ,update_time
    from realtime_jy_work.ol_qb_tags -- 题库标签表
    where isdeleted=1
"""
tags_data = PostgreSQL().select(tags_sql)

# 知识图谱，列式表结构
atlas_sql = """
    select 	a.tree_id as knowledge_id -- 知识点id
            ,b.name as knowledge_name -- 知识点名称
            ,a.knowledge_level -- 知识点层级
            ,a.tags_id as knowledge_tags_id -- 知识点标签id
            ,c.name as knowledge_tags_name -- 知识点标签名称
    from(
        select 	id knowledge_id -- 知识点ID
                ,name knowledge_name -- 知识点名称
                ,(level+1) knowledge_level -- 知识点层级数量
                ,case when level=0 then id else cast(regexp_split_to_table(tree,',') as int) end tree_id -- 知识点树结构
                ,case when length(tags)=0 then null else cast(regexp_split_to_table(tags,',') as int) end tags_id -- 知识点标签
        from realtime_jy_work.ol_knowledge_config -- 知识点配置表
        where subject_type=0 --数学,1为语文
                and isdeleted=1 --是否删除,2为删除，注：可能存在用后再删除的情况
                and id not in (61,1515,1522)
                and split_part(tree,',',1) not in ('61','1515','1522')
    ) a
    left join realtime_jy_work.ol_knowledge_config b on a.tree_id=b.id
    left join realtime_jy_work.ol_qb_tags c on a.tags_id=c.id and c.isdeleted=1
    group by 1,2,3,4,5
    order by knowledge_id,knowledge_tags_id
"""
atlas_data = PostgreSQL().select(atlas_sql)


# ====================================  数据清洗
# 原始数据tree、tags拆分为多列
config_tree_dt = [dict([(re.sub('^\d+x\s+', "", y), y) for y in x.split(',')]) for x in config_data['tree']]
config_tags_dt = [dict([(re.sub('^\d+x\s+', "", y), y) for y in x.split(',')]) for x in config_data['tags']]
config_tree_dt = regexp_split_to_table(config_tree_dt, 'tree').reset_index(drop=True)
config_tags_dt = regexp_split_to_table(config_tags_dt, 'tags').reset_index(drop=True)
config_data = pd.concat([config_data, config_tree_dt, config_tags_dt], axis=1)

# 把拆分好的treeID匹配name并合并到config_data
config_knowledge = config_data[['knowledge_id', 'knowledge_name']].copy()
config_knowledge['knowledge_id'] = config_knowledge['knowledge_id'].astype('str')
tags_knowledge = tags_data[['tags_id', 'tags_name']].copy()
tags_knowledge['tags_id'] = tags_data['tags_id'].astype('str')

config_df, tags_df = knowledge_join(join_data=config_data.iloc[:, 7:], from_df_tree=config_knowledge, from_df_tags=tags_knowledge)
config_data = pd.concat([config_data, config_df, tags_df], axis=1)
config_data.drop(['knowledge_id', 'knowledge_name', 'tree', 'tags', 'create_time', 'update_time'], axis=1, inplace=True)
config_data.drop_duplicates(inplace=True)


# ====================================  存储
writer = pd.ExcelWriter('知识图谱_20200508.xlsx', engine='openpyxl')
config_data.to_excel(excel_writer=writer, sheet_name="知识结构_行", index=False)
atlas_data.to_excel(excel_writer=writer, sheet_name="知识结构_列", index=False)
writer.save()

