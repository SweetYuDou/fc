import json
import re
import os
import fitz  # PyMuPDF
from langchain_community.llms import OpenAI
from zhipuai import ZhipuAI


def pdf_to_text(pdf_path):
    """
    将PDF文件内容转换为文本。
    参数:
        pdf_path: PDF文件的路径。
    返回:
        包含PDF文本内容的字符串。
    """
    # 打开PDF文件
    doc = fitz.open(pdf_path)
    text = ''

    # 遍历每一页
    for page in doc:
        text += page.get_text()

    # 关闭文档
    doc.close()
    return text


class Analysis():
    def __init__(self) -> None:
        os.environ['OPENAI_API_KEY'] = "sk-xxxxxxxxxxx"
        os.environ['OPENAI_API_BASE'] = "https://api.openai-proxy.org/v1"

    def get_prompt(self, resume_text):
        prompt = f"""\
                 你是一个简历分析师。给你[面试者简历]，提取信息：\
                 要求：\
                -返回结果格式为：
                 ## 基本信息：
                 ## 自我评价：
                 ## 工作经历：
                 ## 在校经历：
                 ## 项目经历：
                 -项目经历包含开始时间和结束时间
                 [面试者简历]：{resume_text}
                 """
        return prompt

    def get_request(self, resume_text):
        llm = OpenAI(model_name='gpt-3.5-turbo', max_tokens=1024)
        prompt = self.get_prompt(resume_text)
        rsp = llm(prompt)

        sections = {
            '基本信息': r'基本信息：\s*(.*?)(?=\n\S+：|$)',
            '自我评价': r'自我评价：\s*(.*?)(?=\n\S+：|$)',
            '工作经历': r'工作经历：\s*(.*?)(?=\n\S+：|$)',
            '在校经历': r'在校经历：\s*(.*?)(?=\n\S+：|$)',
            '项目经历': r'项目经历：\s*(.*?)(?=\n\S+：|$)',
        }

        words_list = []
        # print(f'模型返回结果:{rsp}')
        for key, pattern in sections.items():
            match = re.search(pattern, rsp, re.DOTALL)
            if match:
                extracted_text = match.group(1).strip()
                words_list.append(f"{key}:{extracted_text}")
            else:
                words_list.append(f"{key}:""")
        return words_list


#
# def analysis_resume(resume_text, client):
#     """
#     使用智谱AI进行简历解析。
#     参数:
#         resume_text: 简历文本内容。
#         client: 智谱AI客户端实例。
#     返回:
#         智谱AI解析结果。
#     """
#     response = client.chat.completions.create(
#         model="GLM-4",
#         messages=[
#             {
#                 "role": "user",
#                 "prompt": (
#                             f"""\
#                             你是一个简历分析师。给你[面试者简历]，提取信息：\
#                             要求：\
#                             -返回结果格式为：
#                             ## 基本信息：
#                             ## 自我评价：
#                             ## 工作经历：
#                             ## 在校经历：
#                             ## 项目经历：
#                             [面试者简历]：{resume_text}
#                             """),
#                 "content": resume_text
#             }
#         ],
#         top_p=0.7,
#         temperature=0.95,
#         max_tokens=1024,
#         stream=False,
#     )
#
#     pattern_info = r'基本信息：\s*(.*)'
#     pattern_eval = r'自我评价：\s*(.*)'
#     pattern_work = r'工作经历：\s*(.*)'
#     pattern_school = r'在校经历：\s*(.*)'
#     pattern_project = r'项目经历：\s*(.*)'
#
#     match_info = re.search(pattern_info, response.choices[0].message.content)
#     match_eval = re.search(pattern_eval, response.choices[0].message.content)
#     match_work = re.search(pattern_work, response.choices[0].message.content)
#     match_school = re.search(pattern_school, response.choices[0].message.content)
#     match_project = re.search(pattern_project, response.choices[0].message.content)
#     matches = [match_info, match_eval, match_work, match_school, match_project]
#
#     words_list = []
#     # print(f'模型返回结果:{response.choices[0].message.content}')
#     for match in matches:
#         if match:
#             extracted_words = match.group(1)
#             words_list.append(re.findall(r'\b(\w+)\b', extracted_words))
#     # return words_list


if __name__ == "__main__":
    resume_path = './resume.pdf'
    resume_text = pdf_to_text(resume_path)
    Analyst = Analysis()
    resume_analysis = Analysis.get_request(Analyst, resume_text)
    resume_info = {
        "data": resume_analysis
    }
    # resume_info = analysis_resume(resume_text=resume_text, client=ZhipuAI(api_key="f7241adf0a0e9da8f06ee7e75595d548.GdloUbUm6c7y4Iaf"))
    with open('./resume_info.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(resume_info, indent=2, ensure_ascii=False))
    print(resume_info)



