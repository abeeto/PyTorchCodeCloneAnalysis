import numpy as np

fr = open('data.csv', 'r')
content = fr.readlines()
datas = [];labels=[]
for x in content[1:]:
    x = x.strip().split(' ')
    datas.append([float(i) for i in x[:-2]])
    if x[-1]=='危险': labels.append(0.)
    elif x[-1]=='可疑': labels.append(1.)
    else: labels.append(2.)
datas = np.array(datas).astype('float32')
labels=np.array(labels)
print(datas.shape)
print(labels.shape)