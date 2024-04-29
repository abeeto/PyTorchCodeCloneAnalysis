import os
import sys

dir = 'paper-png'
files = os.listdir(dir)
str = ''
for file in files:
    subdir = dir + '/' + file
    # print(subdir)
    subfiles = os.listdir(subdir)
    for subfile in subfiles:
        # print('----' + subdir + '/' + subfile)
        str = str + subdir + '/' + subfile + '\n'

# print(str)
saveFile = open('png_data.txt', mode='w')
saveFile.write(str)
