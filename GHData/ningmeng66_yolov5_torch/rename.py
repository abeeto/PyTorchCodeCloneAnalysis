import os

path = r'C:\Users\admin\Downloads\拼接堇青石图像采集\n'

fileType = '.jpg'
count = 61
startNumber = 1
name = 'img_train'
for file in os.listdir(path):
    oldF = os.path.join(path, file)

    if os.path.isfile(oldF) and os.path.splitext(oldF)[1] == fileType:
        newF = os.path.join(
            path, name + str(count + int(startNumber)) + fileType)
        os.rename(oldF, newF)
    else:
        continue
    count += 1