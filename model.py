# -*- coding:utf-8 -*-
import os
import transformers
from transformers import DebertaTokenizer
from transformers import DebertaConfig, DebertaModel
import torch
from torch import nn
import jieba
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai, langchain
from langchain_community.llms import OpenAI
import re
from kg import *
from resume_process import *

# hrgraph = Graph("http://localhost:7475", auth=("neo4j", "ZYDzyd917917"), name='system')

UNCASED = './deberta-base'
token = DebertaTokenizer.from_pretrained(UNCASED)


class Deberta(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        configuration = DebertaConfig()
        # Initializing a model (with random weights) from the microsoft/deberta-base style configuration
        self.encoder = DebertaModel.from_pretrained(UNCASED)

    def forward(self, x):
        output = self.encoder(**x)
        return torch.sum(output.last_hidden_state, dim=1)


class DCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = nn.LayerNorm(768)
        self.bn2 = nn.LayerNorm(768)
        self.l1 = nn.Linear(768, 768)
        self.l2 = nn.Linear(768, 768)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        x1 = self.bn1(self.leaky_relu(self.l1(x)))
        x1 = x1 * x + x
        x2 = self.bn2(self.leaky_relu(self.l2(x)))
        x2 = x2 * x + x1
        return x2


class EncodingModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Deberta()
        self.dcn = DCN()

    def forward(self, x):
        output = self.encoder(x)
        return self.dcn(output)

    def encode(self, x):
        output = token(
            text=x,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors='pt'
        )
        output = self.encoder(output)
        output = self.dcn(output)
        return output


class UsrRecallModel():
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def recall(self, x, y):
        """
        基于词匹配进行召回
        Args:
            x (string): 用户职业
            y (list): 职业实体
        """
        jieba.load_userdict('./word.txt')
        word_list = jieba.cut(x)
        id_list = []
        with open('./word.txt', 'r', encoding='utf-8') as file:
            keywords = set(file.read().split())
        matches = [word for word in word_list if word in keywords]

        for word in matches:
            for entity_id in query_entity_ids_by_conditions(property_value=word):
                id_list.append(entity_id)
        id_list = list(set(id_list))

        job_list = []
        for job_id in id_list:
            job_list.append(query_entity_by_id(entity_id=job_id).get('name'))
        return job_list



        # for word in word_list:
        #     if word not in ["。", "，", "；", ".", ',', ';' ':', '：', '！', '!', '?', '？']:
        #         word_list_.append(word)
        # word_list = word_list_
        # waiting_idlist = []
        # waiting_list = []
        # for entity in y:
        #     # 遍历待选岗位
        #     for word in word_list:
        #         # 遍历简历内容
        #         if word in entity:
        #             # 若简历中的关键词与待选岗位匹配，将该岗位纳入候选名单
        #             waiting_list.append(entity)
        #             break
        #             # 跳转到下一个岗位
        #         return waiting_list

#
# class HrRecallModel():
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#
#     def recall(self, position, y):
#         """
#         基于词匹配进行召回
#         Args:
#             position (string): 岗位
#             y (list): 职业实体
#         """
#         jieba.load_userdict('./word.txt')
#         job_list = []
#         ids = query_entity_ids_by_conditions(graph=hrgraph, property_value=position)
#         for job_id in ids:
#             job_list.append(query_entity_by_id(graph=hrgraph, entity_id=job_id))
#
#         return job_list


class RerankingModel():
    def __init__(self) -> None:
        self.encoder = EncodingModel()

    def encoding(self,x):
        if type(x) is list:
            return torch.cat([self.encoding(x_) for x_ in x], dim=0)
        return self.encoder.encode(x).detach()

    def find_most_similar(self, x, y, topk=5):
        if topk > len(y):
            topk = len(y)
        # 转换x和y为向量表示
        x_vec = np.array(self.encoding(x))
        y_vec = np.array(self.encoding(y))
        # 计算余弦相似度
        similarity_scores = cosine_similarity(x_vec, y_vec)[0]
        # print(cosine_similarity(x_vec, y_vec)[0])
        # 根据相似度得分排序
        # print(np.argsort(similarity_scores))
        
        top_indices = np.argsort(similarity_scores)[::-1][0:topk]
        # print(top_indices)
        top_y = [y[i] for i in top_indices]
        return top_y


class RankingModel():        
    def __init__(self) -> None:
        os.environ['OPENAI_API_KEY'] = "sk-mQs4S4qxCL9fpWuLvT53jd8yMUJfM1MtS62GxaFMqrHUf1hB"
        os.environ['OPENAI_API_BASE'] = "https://api.openai-proxy.org/v1"
        
    def get_prompt(self, x, y, k=3):
        prompt = f"""\
        你是一个职位推荐官，给你[面试者简历],[职位列表],返回最适合该面试者的{k}个职位。\
        要求:\
        -返回结果必须完整保留[职位列表]中的名称
        -只能返回{k}个职位
        -需要简短的说明理由，用 ##结果: 开头输出推荐列表，每一项用,隔开
        [面试者简历]:{x}
        [职位列表]:{",".join(y)}
        """
        return prompt

    def get_request(self, x, y, k=3):
        llm = OpenAI(model_name='gpt-3.5-turbo', max_tokens=1024)
        prompt = self.get_prompt(x, y, k)
        rsp = llm(prompt)
        pattern = r'结果:\s*(.*)'  # 匹配以"结果:"开头，后面跟着任意字符的模式
        match = re.search(pattern, rsp)
        print(f'模型返回结果:{rsp}')
        if match:
            extracted_words = match.group(1)
            words_list = re.findall(r'\b(\w+)\b', extracted_words)
        else:
            words_list = y
        return words_list


if __name__ == '__main__':
    model = EncodingModel()
    y = ['前端工具人', '后端工具人', '网络开发工程师', "前端高级工程师"]
    x = '岗位职责：1、独立承担Web前端开发任务，根据工作安排高效、高质地完成代码编写，确保符合规范的前端代码规范；2、利用HTML5/CSS3/JavaScript/jQuery5/Ajax等各种Web技术进行产品的界面开发；3、负责公司现有项目和新项目的前端修改调试和开发工作；4、与设计团队紧密配合，能够实现实现设计师的设计想法；5、与后端开发团队紧密配合，确保代码有效对接，优化网站前端性能；6、页面通过标准校验，兼容各主流浏览器。岗位要求：1、熟练掌握JavaScript，熟悉HTML5/XML/JSON前端开发技术，熟悉DIV CSS布局；2、能使用原生的js或jQuery制作出页面常用的表现层动态效果,有node.js经验者优先；3、对浏览器兼容性、代码可维护性、前端性能优化等有深入研究；4、为人诚实正直，做事认真负责，具有良好的沟通和团队协作能力；5、有大型网站前端或移动web开发经验者优先。'
    # resume_path = './resume.pdf'
    # x = str(analysis_resume(pdf_to_text(resume_path), client=ZhipuAI(api_key="f7241adf0a0e9da8f06ee7e75595d548.GdloUbUm6c7y4Iaf")))
    recall = UsrRecallModel()
    rank_model = RerankingModel()
    print(rank_model.find_most_similar(x, y, 1))
    # model = RankingModel()
    # model.get_request(x,y)
    