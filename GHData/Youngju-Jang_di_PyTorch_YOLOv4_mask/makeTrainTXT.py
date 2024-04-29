from glob import glob
from numpy import apply_over_axes
from sklearn.model_selection import train_test_split

img_list = glob('\dataset\images\\val\*.jpg')
len(img_list)

train_img_list, test_img_list = train_test_split(img_list, test_size=0.1, random_state=42)
print(len(train_img_list), len(test_img_list))
'''
with open('./data/train.txt', 'w') as f:
    f.write('\n'.join(train_img_list) + '\n')

with open('./data/test.txt', 'w') as f:
    f.write('\n'.join(test_img_list) + '\n')
'''
with open('./data/val.txt', 'w') as f:
    f.write('\n'.join(img_list) + '\n')

