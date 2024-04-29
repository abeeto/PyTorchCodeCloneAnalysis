import glob
import os
import random
import shutil

rootdir = '/path/to/dataset-root-dir'
os.mkdir(f'{rootdir}-split')
os.mkdir(f'{rootdir}-split/train')
os.mkdir(f'{rootdir}-split/test')

for i in range(26):
    imglist = glob.glob(f"{rootdir}/{(chr(i + ord('A')))}/*")
    os.mkdir(f"{rootdir}-split/train/{(chr(i + ord('A')))}")
    os.mkdir(f"{rootdir}-split/test/{(chr(i + ord('A')))}")
    
    random.shuffle(imglist)
    trainlist = imglist[:int(len(imglist)*0.9)]
    testlist = imglist[int(len(imglist)*0.9):]
    print(len(imglist), len(trainlist), len(testlist))

    for train in trainlist:
        shutil.copy(train, train.replace(f"/{chr(i + ord('A'))}/", f"-split/train/{chr(i + ord('A'))}/"))
    for test in testlist:
        shutil.copy(test, test.replace(f"/{chr(i + ord('A'))}/", f"-split/test/{chr(i + ord('A'))}/"))
    

