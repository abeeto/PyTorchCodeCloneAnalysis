import os
import random



in_dir = '/local-scratch/mma/DISN/ShapeNetRendering/03001627'
dir = '/local-scratch/mma/DISN/version1'
sdf_dir =  '/local-scratch/mma/DISN/ShapeNetOut64/03001627'

all_samples = [name for name in os.listdir(in_dir)
               if os.path.isdir(os.path.join(in_dir, name))]



path1 = []

for path, subdirs, files in os.walk(in_dir):
    #print(subdirs)
    for name in files:
        if name.endswith('.png'):
            path_split = path.split(os.sep)
            if os.path.isfile(os.path.join(sdf_dir, path_split[6], 'SDF.npy')):
                path1.append(path_split[6])


path2 = set(path1)
path_list = list(path2)



print(len(path1))
print(len(path2))
print(len(path_list))


seed = 1234
random.seed(seed)
random.shuffle(path_list)

r_val = 0.0002
r_test = 0.2

# Number of examples
n_total = len(path_list)

n_val = int(r_val * n_total)
n_test = int(r_test * n_total)
n_train = n_total - (n_val + n_test)

print(n_val, n_test, n_train)

# Select elements
train_set = path_list[:n_train]
val_set = path_list[n_train:n_train+n_val]
test_set = path_list[n_train+n_val:]


# Write to file
train = []
with open(os.path.join(dir, 'train1.txt'), 'w') as f:
    for l in range(len(train_set)):
        for i in range(24):
            name = '/' + str("{:02d}".format(i)) + '.png'
            train.append(str(train_set[l]) + name + '\n')
            f.write(str(train_set[l]) + '/rendering' + name + '\n')

with open('train.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('train.txt','w') as target:
    for _, line in data:
        target.write( line )


with open(os.path.join(dir, 'test1.txt'), 'w') as f:
    for l in range(len(test_set)):
        for i in range(24):
            name = '/' + str("{:02d}".format(i)) + '.png'
            f.write(str(test_set[l]) + '/rendering' + name + '\n')

with open('test.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('test.txt','w') as target:
    for _, line in data:
        target.write( line )


with open(os.path.join(dir, 'val1.txt'), 'w') as f:
    for l in range(len(val_set)):
        for i in range(24):
            name = '/' + str("{:02d}".format(i)) + '.png'
            f.write(str(val_set[l]) + '/rendering' + name + '\n')

with open('val.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('val.txt','w') as target:
    for _, line in data:
        target.write( line )
