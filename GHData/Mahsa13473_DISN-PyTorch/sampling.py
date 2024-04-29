import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Sampling 2048 and save bounding box and SDF points')

parser.add_argument('in_file', type=str,
                    help='Location of SDF input file')

parser.add_argument('out_file', type=str,
                    help='Location of output file')

args = parser.parse_args()

sdfFile = open(args.in_file, 'r')
SDF_list = []
j = 1
for line in sdfFile:
    split = line.split()
    #if blank line, skip
    if not len(split):
      continue

    if j == 1:
        dimensionX = int(split[0])
    if j == 2:
        dimensionY = int(split[0])
    if j == 3:
        dimensionZ = int(split[0])

    if j == 4:
        bminx = float(split[0])
    if j == 5:
        bminy = float(split[0])
    if j == 6:
        bminz = float(split[0])

    if j == 7:
        bmaxx = float(split[0])
    if j == 8:
        bmaxy = float(split[0])
    if j == 9:
        bmaxz = float(split[0])


    if j>9:
        SDF_list.append(float(split[0]))

    j = j+1

sdfFile.close()

# print(dimensionX)
# print(dimensionY)
# print(dimensionZ)
#
# print(bminx, bminy, bminz)
# print(bmaxx, bmaxy, bmaxz)


gridsizeX = (bmaxx - bminx)/dimensionX
gridsizeY = (bmaxy - bminy)/dimensionY
gridsizeZ = (bmaxz - bminz)/dimensionZ

# print(gridsizeX)
# print(gridsizeY)
# print(gridsizeZ)


mu = 0.0
sigma = 0.1

#count, bins, ignored = plt.hist(SDF_list, 30 , alpha = 0.5, label = 'input', density = True) #density = True


SDF_value = np.asarray(SDF_list)
SDF_index = np.arange(SDF_value.size)

probability = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(SDF_value - mu)**2 / (2 * sigma**2))

sum = sum(probability)
probability = probability/sum

sample = np.random.choice(SDF_index, size=2048, replace=False, p=probability)



SDF = OrderedDict() #{}

for ii in range(len(sample)):

    index = sample[ii]

    k = index/((dimensionX+1)*(dimensionY+1))
    index = index % ((dimensionX+1)*(dimensionY+1))

    j = index/(dimensionX+1)
    index = index % (dimensionX+1)

    i = index

    x = i*gridsizeX + bminx
    y = j*gridsizeY + bminy
    z = k*gridsizeZ + bminz

    SDF[tuple([i, j, k])] = SDF_list[sample[ii]]


final = {'bmin':[bminx, bminy, bminz], 'bmax':[bmaxx, bmaxy, bmaxz], 'SDF': SDF}

print(len(SDF))

# Save
#np.save(args.out_file, final)
np.save(args.out_file, final)


# Load
# read_dict = np.load('M.npy')
# print(read_dict.item().get('SDF'))
