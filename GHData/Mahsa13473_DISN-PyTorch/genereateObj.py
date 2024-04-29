import numpy as np
from decimal import Decimal
import math
import sys
import argparse
import os
import json

# TODO Change names and use for loop
#-------------------------------------------------------------------------------


def readObjfile(filepath):

    ObjFile= open(filepath, 'r')
    vertexList = []
    faceList = []

    for line in ObjFile:
        split = line.split()

        #if blank line, skip
        if not len(split):
            continue

        if split[0] == "v":
            vertexList.append(split[1:])

        elif split[0] == "f":
            count=1
            eachFaceList = []
            while count<=3:
                removeSlash = split[count].split('/')
                eachFaceList.append(int(removeSlash[0]))
                count+=1
            faceList.append(eachFaceList)

    ObjFile.close()
    return vertexList, faceList





#-------------------------------------------------------------------------------
#Writing Output obj file
def writeObjfile(vertexList, faceList, filepath):
    try:
        f = open(filepath, "w+")

        for i in range(len(vertexList)):
            f.write("v ")
            f.write(vertexList[i][0])
            f.write(" ")
            f.write(vertexList[i][1])
            f.write(" ")
            f.write(vertexList[i][2])
            f.write("\n")

        for i in range(len(faceList)):
            f.write("f ")
            f.write(str(faceList[i][0]))
            f.write(" ")
            f.write(str(faceList[i][1]))
            f.write(" ")
            f.write(str(faceList[i][2]))
            f.write("\n")

        f.close()

        print("Output Created!")
    except:
        print("ERROR")
        

parser = argparse.ArgumentParser(description='gen obj file')

parser.add_argument('in_file', type=str,
                    help='Location of obj input file')

parser.add_argument('out_file', type=str,
                    help='Location of obj output file')

args = parser.parse_args()

v, f = readObjfile(args.in_file)
writeObjfile(v, f, args.out_file)


