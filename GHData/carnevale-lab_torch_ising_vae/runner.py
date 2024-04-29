import os
import sys 

with open(sys.argv[1], "r") as f:
    os.system("export CUDA_VISIBLE_DEVICES=" + sys.argv[2])
    for line in f:
        os.system(line)
