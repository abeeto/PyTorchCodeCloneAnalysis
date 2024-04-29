import csv
import os.path

import torch
import regression.linear_regression as lr
import regression.non_linear_regression as nlr

debug = False
print("** linear regression v1 **")
print('note: datafiles must be located in the data folder in root')


def loadData(filePath):
    csvFile = open(filePath)
    csvData = csv.reader(csvFile)

    dataList = []

    for row in csvData:
        dataList.append(row)
    csvFile.close()
    return dataList


def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def get_file():

    print("please choose a file to perform and visualize a linearization of")
    print("1. length_weight.csv")
    print("2. day_length_weight.csv")
    print("3. day_head_circumference.csv (non linear regression)")

    chose_file = False
    while not chose_file:
        selection = input("Selection:")
        if RepresentsInt(selection):
            selection = int(selection)

            if selection == 1:
                return "length_weight.csv"
            if selection == 2:
                return "day_length_weight.csv"
            if selection == 3:
                return "day_head_circumference.csv"


data = []
fileName = get_file()

fullPath = os.path.dirname(os.path.abspath(__file__))+'/data/' + fileName
if os.path.isfile(fullPath):
    print("loading " + fileName)
    data = loadData(fullPath)

if fileName == "day_head_circumference.csv":
    nlr.non_linear2d(data)
else:
    lr.performLinearRegression(data)
