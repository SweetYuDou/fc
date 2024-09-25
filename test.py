from model import *
from zhipuai import ZhipuAI
from flask import Flask, request, render_template, redirect, jsonify
import openai
import os
import pandas as pd
from resume_process import *

df = pd.read_csv('./data_all.csv')

openai.api_key = "sk-"
openai.api_base = "https://api.openai-proxy.org/v1"
# app = Flask(__name__)
# # 绑定访问地址127.0.0.1:5000/user

recall_model = UsrRecallModel()
rerank_model = RerankingModel()
rank_model = RankingModel()

load_path = "./encoder.ckpt"
if load_path and load_path != "":
    rerank_model.encoder.load_state_dict(torch.load(load_path))


def tuijian():
    text = pdf_to_text("./resume.pdf")
    # anal = Analysis()
    # info = Analysis.get_request(anal, text)

    job_example = ['前端', '后端', '算法', '测试', '运维', '编程', '计算机', '安全', '技术', '工程师', '技术员', '运营', '硬件']

    recall_list = recall_model.recall(text, job_example)
    print(f'召回结果:{recall_list}')
    rerank_list = rerank_model.find_most_similar(text, recall_list, topk=10)
    print(f'粗排结果:{rerank_list}')
    rank_list = rank_model.get_request(text, rerank_list)
    print(f'精排结果:{rank_list}')
    print(rank_list)
    final_ids = []
    for job in rank_list:
        for entity_id in query_entity_ids_by_conditions(property_value=job):
            final_ids.append(entity_id)
    final_ids = list(set(final_ids))
    rm_job = {
            'jobs': [
                {
                    'positionName': query_entity_by_id(entity_id=rm_id).get('name'),  # 岗位名称
                    'base': query_entity_by_id(entity_id=rm_id).get('address'),  # 岗位地点
                    'educationalRequirements': query_entity_by_id(entity_id=rm_id).get('education'),  # 学历要求
                    'salary': query_entity_by_id(entity_id=rm_id).get('salary'),  # 薪资
                    'companyName': query_entity_by_id(entity_id=rm_id).get('company'),  # 公司名称
                    'detailUrl': query_entity_by_id(entity_id=rm_id).get('link')  # 详情页链接
                }
                for rm_id in final_ids
            ]
    }
    return rm_job


def pingjia():
    text = pdf_to_text('./resume.pdf')
    dp = '前端'
    anal = Analysis()
    info = Analysis.get_request(anal, text)
    processed_info = {
        "简历信息": info,
        "意向岗位": dp,
    }
    pj = request_for_enhancer(processed_info)
    print(pj)

    overall_review = re.search(r'总体建议[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)
    base_info = re.search(r'基本信息意见[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)
    self_evaluation = re.search(r'自我评价意见[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)
    work_experience_suggest = re.search(r'工作经历意见[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)
    work_experience_optimize = re.search(r'工作经历优化[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)
    school_experience = re.search(r'在校经历意见[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)
    project_experience_suggest = re.search(r'项目经历意见[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)
    project_experience = re.search(r'项目经历优化[:：]\s*(.*?)(?=\n\S+#|$)', pj, re.DOTALL)

    overall_review = overall_review.group(1).strip() if overall_review else ""
    base_info = base_info.group(1).strip() if base_info else ""
    self_evaluation = self_evaluation.group(1).strip() if self_evaluation else ""
    work_experience_suggest = work_experience_suggest.group(1).strip() if work_experience_suggest else ""
    work_experience_optimize = work_experience_optimize.group(1).strip() if work_experience_optimize else ""
    school_experience = school_experience.group(1).strip() if school_experience else ""
    project_experience_suggest = project_experience_suggest.group(1).strip() if project_experience_suggest else ""
    project_experience = project_experience.group(1).strip() if project_experience else ""

    data = {
        "overallReview": overall_review,
        "baseInfo": base_info,
        "selfEvaluation": self_evaluation,
        "workExperience": {
            "suggest": work_experience_suggest,
            "postoptimality": work_experience_optimize,
        },
        "schoolExperience": school_experience,
        "projectExperienceInfo": {
            "experience": project_experience,
            "suggest": project_experience_suggest,
        }
    }

    print(data)
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return data


def request_for_enhancer(rule):
    prompt = f"""
    你是一个能力评价专家，
    给定你一组用户简介和期望职位，请生成对输入的每一小点提出意见并给出优化后的内容,
    再对用户与意向岗位间的契合度打分(1-100)：
    要求\
    -返回结果格式为：
    ## 基本信息意见：
    ## 自我评价意见：
    ## 工作经历意见：
    ## 在校经历意见：
    ## 项目经历意见：
    ## 工作经历优化：
    ## 项目经历优化：[
                "## 项目名称":
                "## 开始时间":
                "## 结束时间":
                "## 项目描述":    
                ]  
    ## 总体建议:
    <input>:{rule}
    """
    # print(prompt)
    # print(rule)
    # 创建一个 GPT-3 请求
    completion = openai.ChatCompletion.create(
      model="gpt-3.5-turbo-16k",
      # model="gpt-3.5-turbo-0301",
      messages=[
        {"role": "user", "content": prompt}
      ],
      request_timeout=30,
      top_p=0.0,
      frequency_penalty=0.1,
      presence_penalty=0.1
    )

    return completion.choices[0].message.content


def hr():
    position = '前端'
    ask = '学历本科及以上'
    salary = '低于8k'
    require = position + "," + ask + "," + salary

    usrsql = []
    recall_list = recall_model.recall(position, usrsql)

    print(f'召回结果:{recall_list}')
    rerank_list = rerank_model.find_most_similar(require, recall_list, topk=10)
    print(f'粗排结果:{rerank_list}')
    rank_list = rank_model.get_request(require, rerank_list)
    print(f'精排结果:{rank_list}')
    print(rank_list)
    return {rank_list}


if __name__ == '__main__':
    # rm = tuijian()
    # print(rm)
    pingjia()


