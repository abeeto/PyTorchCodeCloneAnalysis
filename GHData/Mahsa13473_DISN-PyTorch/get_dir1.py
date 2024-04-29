import os


obj_preprocess_PATH = '/local-scratch/mma/DISN/03001627'

out_PATH = '/local-scratch/mma/DISN/ShapeNetOut64/03001627'


new_path = []

for path, subdirs, files in os.walk(obj_preprocess_PATH):
    #print(subdirs)
    for name in files:
        if name.endswith('model.obj'):
            path1 = os.path.normpath(path)
            path_split = path.split(os.sep)
            new_path.append(path_split[5])




#new_obj_path = []

f = open('bash_03001627_64.sh', "w")
c = 0

f.write('#!/bin/bash')
f.write("\n")

for i in range(len(new_path)):

    obj_PATH = os.path.join(obj_preprocess_PATH, new_path[i], 'model.obj')

    try:
        c = c+1
        print(c)
        os.makedirs(os.path.join(out_PATH, new_path[i]))

        output_obj_path = os.path.join(out_PATH, new_path[i], 'model.obj' )
        output_SDF_path = os.path.join(out_PATH, new_path[i], 'SDF.txt' )
        output_sampling_path = os.path.join(out_PATH, new_path[i], 'SDF.npy' )

        f.write('python generateObj.py {} {}'.format(obj_PATH, output_obj_path))
        f.write("\n")
        #f.write('../../utilities/computeDistanceField/computeDistanceField {} 64 64 64 -s 1 -m 1 -g 0.01 -o {}' .format(output_obj_path, output_SDF_path))
        #f.write("\n")
        #f.write('python sampling.py {} {}'.format(output_SDF_path, output_sampling_path))
        #f.write("\n")
        #f.write('rm -f {}'.format(output_SDF_path))
        #f.write("\n")
    except:
        print("HI")


print(c)

f.close()
