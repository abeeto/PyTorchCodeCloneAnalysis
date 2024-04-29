import os

# print([x.path for x in os.scandir('./allset/anime') if x.name.endswith(".jpg")]) # 可以解析 x.path 以获取 label
imagepath = []
for dir in os.listdir('./allset/'):  # 目录里的所有文件
    temps = os.listdir(os.path.join('./allset/', dir))
    for i in range(len(temps)):
        temps[i] = dir + '/' + temps[i]
    imagepath += temps
print(imagepath)
img_path = os.path.join('./allset/', imagepath[0])  # 获取索引为0的图片的路径名
print(img_path)
label = img_path.split('/')[2]
print(label)
