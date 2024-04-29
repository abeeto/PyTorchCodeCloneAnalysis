import pyautogui as ui
import os
import time
from os import listdir
from os.path import isfile, join
from shutil import copyfile
from xml.etree import ElementTree as et
import cv2

def takeScreenShot():

    var = 0
    print("Enter Champion Name:")
    inp = input()
    time.sleep(10)
    try:
        os.mkdir("images\\" + inp)
    except:
        pass
    while True:
        ui.screenshot("images\\" + inp + '\\' + str(var) + '.png', region=(721, 279, 388, 440))
        print(var)
        var = var + 1
        time.sleep(.4)
def generateXML(inp):
    files = [f for f in listdir("images\\" + inp) if isfile(join("images\\"+inp, f))]
    print(files)
    for file in files:
        final = file.split('.')[0]
        string = str("<?xml version='1.0' encoding='UTF-8'?><annotation><folder>Aatrox</folder><filename>" + inp + final + '.png' + "</filename><path>C:\\Users\\sagef\\Documents\\Development\\python\\tensorLeague\\images\\" + inp + '\\' + inp + final + '.png' + "</path><source><database>Unknown</database></source><size><width>388</width><height>440</height><depth>3</depth></size><segmented>0</segmented><object><name>Aatrox</name><pose>Unspecified</pose><truncated>1</truncated><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>388</xmax><ymax>440</ymax></bndbox></object></annotation>")
        print(string)
        filename = "images\\" + inp + '\\' + inp + final +'.xml'
        f = open(filename, "w")
        f.write(string)
        f.close()
        print(file)


#takeScreenShot()
#generateXML("Akali")

#Code to remove champion name
"""
var = 0
print("Enter Character Name")
inp = input()
for f in listdir("images\\" + inp):
    if isfile(join("images\\" + inp + "\\" + f)) == True:
        if f.split('.')[1] == "png":
            print(f)
            os.rename("images\\"+ inp + "\\" + f, "images\\" + inp + "\\" + str(var) + ".png")
            var = var + 1
"""


#Code to rename incorrectly named files.
"""
for f in listdir("images\\Akali"):
    if isfile(join("images\\Akali\\" + f)) == True:
        if f.split('.')[1] == "png":
            print (f)
            os.rename("images\\Akali\\" + f, 'images\\Akali\\Akali' + f)
        else:
            print("Not png")
    else:
        print('Not file')
"""
#Convert conbinedDataset to BRG
#"""
for f in listdir("images\\combinedDataset"):
    if isfile(join("images\\combinedDataset\\" + f)) == True:
        if f.split('.')[1] == "png":
            print(f)
            img = cv2.imread("images\\combinedDataset\\" + f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite("images\\combinedDataset\\" + f.split('.')[0] + '.jpg', img)
            os.remove("images\\combinedDataset\\" + f)
#"""
