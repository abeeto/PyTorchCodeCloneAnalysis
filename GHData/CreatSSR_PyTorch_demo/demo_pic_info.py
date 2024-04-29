#分析数据
import cv2
import os

size = []

def read_img(img_path):

    max_height = 0
    max_width =0

    files = os.listdir(r'./'+img_path)
    files.sort(key=lambda x: int(x[:-4]))

    for filename in files:
        img = cv2.imread(img_path + '/' + filename)
        shape = img.shape
        size.append(shape)

        height = shape[0]
        width = shape[1]
        if height >= max_height:
            max_height = height

        if width >= max_width:
            max_width = width

        with open("train_img_info.txt", "a+") as f:  # 打开文件并追加写，没有就创建
            f.write(str(filename)) #可以写入文件名
            f.write(" ")  # 写入空格
            f.write(str(height))  # 写入height
            f.write(" ")  # 写入空格
            f.write(str(width))  # 写入width
            f.write("\n")  # 写入换行

    with open("train_img_info.txt", "a+") as f:
        f.write("max_height:    ")
        f.write(str(max_height))
        f.write("max_width:     ")
        f.write(str(max_width))

    print("最大高：", max_height, "最大宽：", max_width)

if __name__=="__main__":
    path1 = 'data/train/calling_images'
    path2 = 'data/train/normal_images'
    path3 = 'data/train/smoking_images'

    read_img(path1)
    read_img(path2)
    read_img(path3)






