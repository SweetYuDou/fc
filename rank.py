from model import *
from flask import Flask, request, render_template, redirect, jsonify
import openai
import os
import pandas as pd
from resume_process import *
from kg import *
from flask_cors import CORS

df = pd.read_csv('./data_all.csv')

openai.api_key = "sk-"
openai.api_base = "https://api.openai-proxy.org/v1"
app = Flask(__name__)
CORS(app)

recall_model = UsrRecallModel()
# hr_model = HrRecallModel()
rerank_model = RerankingModel()
rank_model = RankingModel()

load_path = "./encoder.ckpt"
if load_path and load_path != "":
    rerank_model.encoder.load_state_dict(torch.load(load_path))


@app.route("/hr", methods=['GET', 'POST'])
def hr():
    if request.method == 'POST':
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        position = data.get('position')
        ask = data.get('ask')
        salary = data.get('salary')

        require = position + ask + salary

        usrsql = []

        # recall_list = recall_model.recall(require, usrsql)
        # print(f'召回结果:{recall_list}')
        # rerank_list = rerank_model.find_most_similar(require, recall_list, topk=10)
        # print(f'粗排结果:{rerank_list}')
        # rank_list = rank_model.get_request(require, rerank_list)
        # print(f'精排结果:{rank_list}')
        # print(rank_list)
        # return {rank_list}
        return {'lyy'}


@app.route("/recommend", methods=['GET', 'POST'])
def comment():
    # 能力评价
    # if request.method == 'POST':
    #     x = request.form['detail']
    # else:
    #     x = request.args['detail']
    # return request_for_enhancer(x)
    if 'resumePdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['resumePdf']
    file.save(os.path.join('pdf_file/', file.filename))

    text = pdf_to_text(os.path.join('pdf_file/', file.filename))

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    dp = request.form.getlist('desiredPosition')
    dp = ",".join(dp)

    anal = Analysis()
    info = Analysis.get_request(anal, text)
    processed_info = {
        "简历信息": info,
        "意向岗位": dp,
    }
    # os.remove(os.path.join('pdf_file/', file.filename))
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


@app.route("/comment", methods=['GET', 'POST'])
def recommend():
    '''
    请求：127.0.0.1:5000/recommend
    参数:
        -user:请求者的求职信息
        -gangwei:所有岗位描述信息的列表
    返回:岗位的排序结果
    '''
    # if request.method == 'POST':
    #     x = request.form['user']
    #     y = request.form['gangwei']
    # else:
    #     if 'user' in request.args.keys():
    #         x = request.args['user']
    #     else:
    #         x = None
    #     if 'gangwei' in request.args.keys():
    #         y = request.args['gangwei']
    #     else:
    #         y = None
    if 'resumePdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['resumePdf']
    file.save(os.path.join('pdf_file/', file.filename))

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    text = pdf_to_text(os.path.join('pdf_file/', file.filename))
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

    # return {rank_list}
    # return jsonify({
    #        'jobs': [
    #             {
    #                 "positionName": "【初级】web前端开发工程师",
    #                 "base": "中山广东倾云科技有限公司一层",
    #                 "educationalRequirements": "大专",
    #                 "salary": "2-4K·13薪",
    #                 "companyName": "广东倾云科技有限公司",
    #                 "detailUrl": "https://www.zhipin.com/job_detail/2c08c111e75221011n1539S8EVVQ.html?lid=6GPZx3II9Bv.search.31&securityId=f1xZgMo04Kywv-u1CqKcTn5qoCHek_nDfH_HMLOZDR2U7UD6hrtSKuU33BsHCW7ByiFvb2dqc6C5mDevs78kfFXp601IBEUpA0Rr43wDm5o5vDLg5UeiSXzoNHZYpCYCdkPCHw5liZ3l&sessionId=",
    #             },
    #             {
    #                 "positionName": "IT运维工程师",
    #                 "base": "衡水桃城区众成大厦2407",
    #                 "educationalRequirements": "大专",
    #                 "salary": "2-4K",
    #                 "companyName": "火眼科技（天津）有限公司",
    #                 "detailUrl": "https://www.zhipin.com/job_detail/8095129b424286831Xx53961FVZU.html?lid=6GPZx3II9Bv.search.32&securityId=nsR9uHW5KATNR-B1MwopcU4Cox09HgFQN4uLvMVSQcN4wzu3H8vtcumS5hrbqfvoWm6DJdxOUTlhqcVqfzOanS4FMI2XFoeSiNwmBQj1wYKcT2pOs9SMz_DXoK6K34YEnC3ZEG8RtcZC&sessionId="
    #             },
    #         ]
    #     })


def after_request(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Content-Type'] = 'application/json'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,Accept,Origin,Referer,User-Agent'

    return resp


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


def cv_evaluate(desc):
    """生成简历评估
    Args:\
        input: dict
            “基本信息”：string，
            “自我评价”：string，
            “工作经历”：string，
            “在校经历”：string，
            “项目经历”：string
    """
    basic_info = desc['基本信息']
    self_desc = desc['自我评价']
    work_exp = desc['工作经历']
    school_exp = desc['在校经历']
    pro_exp = desc['项目经历']


if __name__ == '__main__':
    app.after_request(after_request) 
    app.run(host='10.252.64.12', port=5000, debug=True)
