from py2neo import Graph,NodeMatcher,cypher,Node,Relationship,RelationshipMatcher
import json
import pandas as pd
import numpy as np
import re

usrgraph = Graph("http://localhost:7474", auth=("neo4j", "ZYDzyd917917"), name='neo4j')
# hrgraph = Graph("http://localhost:7475", auth=("neo4j", "ZYDzyd917917"), name='system')


def usrgraph_create(graph):
    # 图谱构建函数
    graph.delete_all()
    datas = pd.read_csv('data_all.csv')
    for index, data in datas.iterrows():
        print(data['title'], data['description'])
        print('------------')
        print(data['address'], data["education"])

        source_node = Node()
        source_node.add_label('岗位')
        source_node['name'] = str(data['title'])
        source_node['address'] = str(data['address'])
        source_node['education'] = str(data['education'])
        source_node['description'] = str(data['description'])
        source_node['link'] = str(data['link'])
        source_node['salary'] = str(data['salary'])
        source_node['company'] = str(data['company'])
        # if not Node.__hash__(source_node):
        # source_match = NodeMatcher(graph).match('岗位', name=str(data['title']))
        # print(source_match.count())
        # if source_match.__len__() == 0:
        graph.create(source_node)

        source_node = NodeMatcher(graph).match('岗位', name=str(data['title'])).first()

    return graph
        # target_node = Node()
        # target_node.add_label('要求')
        # target_node['name'] = str(data['description'])
        #
        # target_match = NodeMatcher(graph).match('要求', name=str(data['description']))
        # if target_match.__len__() == 0:
        #     graph.create(target_node)
        #
        # target_node = NodeMatcher(graph).match('要求', name=str(data['description'])).first()
        #
        # rel_match = RelationshipMatcher(graph)
        # print(source_node, target_node)
        # rel = Relationship(source_node, '岗位要求', target_node)
        # if len(rel_match.match([source_node, target_node], r_type='岗位要求')) == 0:
        #     graph.create(rel)


def query_entity_by_id(graph=usrgraph, entity_id=None):

    """
    根据实体的ID查询实体的属性，并返回其address、education和name。

    Parameters:
    — entity_id:要查询的实体ID。

    Returns:
    — dict:包含实体的address、education和name的字典。
    """

    # 找到对应ID的节点
    entity_node = graph.nodes.get(entity_id)
    if entity_node:
        result = {
            'address': entity_node.get('address', None),  # 获取address属性
            'education': entity_node.get('education', None),  # 获取education属性
            'name': entity_node.get('name', None),  # 获取name属性
            'salary': entity_node.get('salary', None),
            'company': entity_node.get('company', None),
            'link': entity_node.get('link', None),
        }
        return result
    else:
        # 如果ID不对应任何实体，返回空字典
        return {}


def query_entity_ids_by_conditions(graph=usrgraph, entity_label="岗位", property_key="name", property_value=None,
                                   max_salary=None,
                                   min_salary=None):
    """
    根据特定条件查询实体id，包括实体标签、属性、以及薪资条件。
    """
    match_clause = f"MATCH(entity:{entity_label})"
    where_clauses = []

    if property_key and property_value is not None:
        where_clauses = f"WHERE entity.{property_key} contains '{property_value}'"

    # 查询所有实体，然后在Python中过滤
    query = f"{match_clause}{where_clauses} RETURN entity.salary, id(entity)"
    result = graph.run(query)
    # 过滤符合薪资条件的实体
    entity_ids = []
    for record in result:
        salary = record[0]
        salary_range = parse_salary(salary)
        if min_salary is not None and salary_range[0] < min_salary:
            continue
        if max_salary is not None and salary_range[1] > max_salary:
            continue
        entity_ids.append(record[1])
    return entity_ids


def parse_salary(salary_str):
    """
    解析薪资范围字符串，返回范围内的最高工资。
    Args:
        salary_str (str): 薪资字符串，如 '2-3K' 或 '10K以上'。

    Returns:
        int: 范围内的最高工资，单位为元。
    """
    numbers = re.findall(r'(\d+)', salary_str)
    if numbers:
        if '以上' in salary_str:
            # 假设 '10K以上' 取 10K 作为最低值
            min_salary = int(numbers[0]) * 1000
            return min_salary, 1e9
        else:
            # 提取 '2-3K' 中的最高值 3
            max_salary = max(map(int, numbers))*1000
            min_salary = min(map(int, numbers))*1000
            return min_salary, max_salary
    else:
        return -1, -1


# usrgraph_create(usrgraph)


# # 根据id查询节点属性
# entity_id = 11365
# result = query_entity_by_id(entity_id)
# print(result)

# 根据条件查询节点id
# entity_ids = query_entity_ids_by_conditions(
#     usrgraph,
#     entity_label="岗位",
#     property_key="name",
#     property_value="算法",
#     # relationship_type="要求"
#     # max_salary=10000,
#     # min_salary=1000,
# )
# #
# print("Entity IDs:", entity_ids)
#
# for entity_id in entity_ids:
#     result = query_entity_by_id(entity_id)
#     print(result)



