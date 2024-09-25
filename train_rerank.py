import transformers
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader, Dataset
import random
import jieba
import torch.nn as nn
from model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_model_path = './encoder.ckpt'
df = pd.read_csv('./data_all.csv')

with open('./word.txt','r', encoding='utf-8') as f:
    word_list = f.readlines()


def get_data():
    titles = [str(title) for title in df['title'].tolist()]
    descriptions = [str(description) for description in df['description'].tolist()]
    # 正样本
    true_x = descriptions  # 类似用户技术描述
    true_y = titles  # 题目
    true_labels = [1] * len(descriptions)
    # 负样本
    false_x = random.choices(titles, k=len(titles))
    false_y = random.choices(descriptions, k=len(titles))
    for x, y in zip(false_x, false_y):
        for word in word_list:
            if word in x and word in y:
                false_x.remove(x)
                false_y.remove(y)
                break
    false_labels = [0] * len(false_x)
    return true_x, true_y, true_labels, false_x, false_y, false_labels


class HRDataset(Dataset):
    def __init__(self,train=True) -> None:
        super().__init__()
        true_x, true_y, true_labels, false_x, false_y, false_labels = get_data()
        self.x = true_x + false_x
        self.y = true_y + false_y
        random.shuffle(self.x)
        random.shuffle(self.y)
        split = int(0.8*len(self.x))
        if train:
            self.x = self.x[:split]
            self.y = self.y[:split]
        else:
            self.x = self.x[split:]
            self.y = self.y[split:]
        self.labels = true_labels + false_labels

    def __getitem__(self, index) :
        return self.x[index], self.y[index], self.labels[index]

    def __len__(self):
        return len(self.x)


def collate_fn(input):
    x, y, labels = [], [], []
    for item in input:
        x.append(item[0])
        y.append(item[1])
        labels.append(item[2])
    x_data = token(
        text=x,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='pt'
    )
    y_data = token(
        text=y,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors='pt'
    )
    # input_ids:编码之后的数字
    # attention_mask:是补零的位置是0,其他位置是1
    return x_data, y_data, torch.tensor(labels, dtype=torch.float32)


train_data = HRDataset()
val_data = HRDataset(False)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=16,
    collate_fn=collate_fn,
    shuffle=True
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=16,
    collate_fn=collate_fn,
    shuffle=True
)

criterion = nn.MSELoss()
model = EncodingModel()

# print(help(nn.CrossEntropyLoss))
optimizer = torch.optim.Adam(params=model.parameters(),lr=1e-4, weight_decay=0.2)
epochs = 10
pre_min_loss = 1e9
# for epoch in range(epochs):
#     step = 0
#     # train
#     print("======train======")
#     for x, y, label in train_loader:
#         # print(x)
#         x = x.to(device)
#         y = y.to(device)
#         label = label.to(device)
#         model = model.to(device)
#         optimizer.zero_grad()
#         x_embs = model(x)
#         y_embs = model(y)
#         logits = (x_embs * y_embs).sum(dim=-1) / (torch.sqrt(torch.sum(x_embs ** 2, dim=-1)) * torch.sqrt(torch.sum(y_embs ** 2, dim=-1)))
#         # logits = torch.nn.functional.cosine_similarity(x_embs,y_embs,dim=-1)
#         loss = criterion(logits, label)
#         loss.backward()
#         optimizer.step()
#         print(f'epoch:{epoch},step:{step},training loss:{loss.item()}')
#         step += 1
#     # validation
#     print("======val=======")
#     with torch.no_grad():
#         total_loss = 0
#         total_cnt = 0
#         pre_min_loss = 1e9
#         for x, y, label in train_loader:
#             x.to(device)
#             y.to(device)
#             label.to(device)
#             model.to(device)
#             x_embs = model(x)
#             y_embs = model(y)
#             logits = (x_embs * y_embs).sum(dim=-1) / (torch.sqrt(torch.sum(x_embs ** 2, dim=-1)) * torch.sqrt(torch.sum(y_embs ** 2, dim=-1)))
#             loss = criterion(logits, label)
#             total_loss += loss.item()
#             total_cnt += x.shape()[0]
#         print(f'Validation,total_loss:{total_loss},total_cnt:{total_cnt},avg_loss:{total_loss/total_cnt}')
#         if total_loss / total_cnt < pre_min_loss:
#             pre_min_loss = total_loss / total_cnt
#             torch.save(model.state_dict(), save_model_path)

for epoch in range(epochs):
    step = 0
    print("======train======")
    for x, y, label in train_loader:
        x = x.to(device)
        y = y.to(device)
        label = label.to(device)
        model = model.to(device)

        optimizer.zero_grad()
        x_embs = model(x)
        y_embs = model(y)
        logits = (x_embs * y_embs).sum(dim=-1) / (
                    torch.sqrt(torch.sum(x_embs ** 2, dim=-1)) * torch.sqrt(torch.sum(y_embs ** 2, dim=-1)))
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        print(f'epoch:{epoch},step:{step},training loss:{loss.item()}')
        step += 1

    print("======val=======")
    with torch.no_grad():
        total_loss = 0
        total_cnt = 0
        for x, y, label in val_loader:
            x = x.to(device)
            y = y.to(device)
            label = label.to(device)
            model = model.to(device)

            x_embs = model(x)
            y_embs = model(y)
            logits = (x_embs * y_embs).sum(dim=-1) / (
                        torch.sqrt(torch.sum(x_embs ** 2, dim=-1)) * torch.sqrt(torch.sum(y_embs ** 2, dim=-1)))
            loss = criterion(logits, label)
            total_loss += loss.item()
            total_cnt += x_embs.size(0)

        avg_loss = total_loss / total_cnt
        print(f'Validation,total_loss:{total_loss},total_cnt:{total_cnt},avg_loss:{avg_loss}')
        if avg_loss < pre_min_loss:
            pre_min_loss = avg_loss
            torch.save(model.state_dict(), save_model_path)
