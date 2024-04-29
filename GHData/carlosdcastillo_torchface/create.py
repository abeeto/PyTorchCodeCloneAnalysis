import os
import fnmatch

#Train and val need to have the same directories even if some are empty. 
#This will create the directories missing in val

#Copy this to the val directory and run it.

def recglob(directory,ext):
    l = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, ext):
            l.append(os.path.join(root, filename))
    return l

l = recglob('../train/','*.jpg')
l2 = []
for x in l:
    l2.append(x.split('/')[-2])

for x in set(l2):
    try:
        print x
        os.makedirs(x)
    except:
        pass
