
import os
import sys
import math
import numpy as np
from PIL import Image
import random

class Dataset(object):

    nbInput = 0
    nbOutput = 0
    x = []
    y = []
    x_test = []
    y_test = []

    def __init__(self,nbInput, nbOutput):
        self.nbInput = nbInput
        self.nbOutput = nbOutput
        np.dtype('f')

    def add(self, x, y):
        self.x.append(x)
        self.y.append(y)

    def addToTest(self, x, y):
        self.x_test.append(x)
        self.y_test.append(y)

    ################################# PRINT ##################################
    def setPrintOption(self):
        lenMin = len(str(np.amax(self.x)))+1
        nbElem = math.sqrt(self.nbInput)
        np.set_printoptions(threshold=np.nan, linewidth=2+nbElem*lenMin, suppress=True)

    def printInput(self):
        self.setPrintOption()
        print np.asarray(self.x)

    def printOutput(self):
        self.setPrintOption()
        print np.asarray(self.y)

    def printValue(self):
        print "Input: " + str(self.nbInput)
        print "Output: " + str(self.nbOutput)
    ##########################################################################

    ################################# GETTER #################################
    def getInput(self):
        return (self.x)

    def getOutput(self):
        return (self.y)

    def getInputTest(self):
        return (self.x_test)

    def getOutputTest(self):
        return (self.y_test)
    ##########################################################################

    ################################# UTILS ##################################
    def write(self, str):
        sys.stdout.write(str)
        sys.stdout.flush()
    ##########################################################################

    def imageToArray(self, path, convert):
        img = Image.open(path).convert(convert)
        ar = np.array(img)
        return (np.reshape(ar, np.size(ar)).tolist())

    def selectedOutputToArray(self, selected):
		ar = np.array([0]*self.nbOutput)
		if (selected >= 0 and selected < self.nbOutput):
			ar[selected] = 1
		return ar

    def addFolderWithLabel(self, path, toTest=False):
        # path = "dataset/mnist_png/training/"
        if (toTest == True):
            print "---------- TESTING ----------"
        else:
            print "---------- TRAINING ----------"
        for label in os.listdir(path):
            if (label[0] is not "." ):
                self.write("Load folder: " + "\"" + label + "\" ")
                i = 0
                for filename in os.listdir(path + label):
                    if (filename[0] is not "." ):
                        if (i % 1000 == 0):
                            self.write(".")
                        i += 1
                        # self.add(self.imageToArray(path + label + "/" + filename, "L"), self.selectedOutputToArray(int(label)))
                        x = self.imageToArray(path + label + "/" + filename, "L")
                        y = self.selectedOutputToArray(int(label))
                        if (toTest == True):
                            self.addToTest(x, y)
                        else:
                            self.add(x, y)
                print ""
        print ""
