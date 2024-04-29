# 导包初始化
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch import nn
# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_name = "./my_sst2_model"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 数据处理
filename = "./test1821.csv"
fo = open(filename, "r", encoding="utf-8")
print("文件名为: ", fo.name)
line = fo.readline()
lines = []
while line:
    lines.append(line)
    line = fo.readline()
fo.close()
print("数据大小：", len(lines))

count = 0 # 计数器
for item in lines:
    sentence = item.strip().split('\t')[0] # 句子评论
    label = int(item.strip().split('\t')[1]) # 标签（0：消极；1：积极）
    # print(sentence, label)
    # 处理开始
    inputs = tokenizer(sentence, return_tensors='pt')
    pt_outputs = pt_model(**inputs)
    pt_predictions = nn.functional.softmax(pt_outputs.logits, dim=-1)
    # 处理结束
    #print(pt_predictions[0])
    # 预测的结果
    neg = pt_predictions[0][0].item()
    pos = pt_predictions[0][1].item()
    # print(neg, pos)
    # 预测和标签比较
    flag = 1
    if neg > pos:
        flag = 0
    if label == flag:
        count = count + 1

print("匹配数：", count)
print("正确率为：", count / len(lines))
