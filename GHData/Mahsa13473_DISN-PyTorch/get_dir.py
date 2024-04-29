import os


Rendering_PATH = '/local-scratch/mma/DISN/ShapeNetRendering'
obj_PATH = '/local-scratch/mma/DISN/ShapeNetCore.v2'

new_path = []

for path, subdirs, files in os.walk(Rendering_PATH):
    #print(subdirs)
    for name in files:
        if name.endswith('renderings.txt'):
            # print(path[:-10])
            path1 = os.path.normpath(path)
            path_split = path.split(os.sep)
            new_path.append(path_split[5:7])


print(len(new_path))

#new_obj_path = []

f = open('obj_path.txt', "w")
for i in range(len(new_path)):
    #new_obj_path1 = os.path.join(obj_PATH, new_path[i][0], new_path[i][1], 'models', 'model_normalized.obj')
    #new_obj_path.append(new_obj_path1)
    new_obj_path1 = os.path.join(new_path[i][0], new_path[i][1])
    f.write(new_obj_path1)
    f.write("\n")

f.close()
# print(new_obj_path)
