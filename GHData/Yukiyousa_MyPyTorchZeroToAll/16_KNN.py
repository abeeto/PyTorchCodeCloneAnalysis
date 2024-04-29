import math

movie_data = {
    "宝贝当家": [45, 2, 10, "喜剧片"],
    "功夫熊猫3": [40, 5, 35, "喜剧片"],
    "举起手来": [50, 2, 20., "喜剧片"],
    "碟中谍2": [5, 2, 60, "动作片"],
    "叶问3": [4, 3, 65, "动作片"],
    "空中营救": [1, 2, 63, "动作片"],
    "怦然心动": [5, 30, 1, "爱情片"],
    "时空恋旅人": [6, 35, 1, "爱情片"],
    "恋恋笔记本": [10, 40, 1, "爱情片"]
}

# 测试样本: 加勒比海盗1": [15, 3, 60, "？片"]

x = [15, 3, 60]
KNN = []
for key, v in movie_data.items():
    # 计算距离
    d = math.sqrt((x[0] - v[0]) ** 2 + (x[1] - v[1]) ** 2 + (x[2] - v[2]) ** 2)
    # 返回两位小数
    KNN.append([key, round(d, 2)])

# 输出所用电影到加勒比海盗1的距离
print("KNN =", KNN)

# 按照距离大小进行递增排序 对列表的第二个值进行排序
KNN.sort(key=lambda dis: dis[1])

# 选取距离最小的k个样本，这里取k=3；
KNN = KNN[0:3]
print("nearest5 KNN:", KNN)

# 确定前k个样本所在类别出现的频率，并输出出现频率最高的类别
labels = {"喜剧片": 0, "动作片": 0, "爱情片": 0}
for s in KNN:
    label = movie_data[s[0]]
    labels[label[3]] += 1
labels = sorted(labels.items(), key=lambda l: l[1], reverse=True)
print("labels:", labels)
print("prediction result:", labels[0][0])
