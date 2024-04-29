import json
import sys
import pandas as pd
import numpy as np
from itertools import combinations
from shapely.geometry import Polygon
# import matplotlib.pyplot as plt
import os
import cv2
import openpyxl
# from shapely.algorithms import polylabel


##---------------------------------
#default calibration for px top mm:
# 941px = 20mm
#1px = 0,021253985 mm
pxToMmCal = 0.021253985
###--------------------------------
# show the detection points drawn on the image during scripting
imgShow = False
# save the images with detection points drawn on the image in a separate folder
safeOutputImg = True

outputData = pd.DataFrame(
    columns=['FileName', 'NousAnnotationId', 'shapeId', 'BodyLengthPx', 'BodyLengthMm(1200dpi)', 'BodySurfacePx', 'Remark'])

# assign directory
if len(sys.argv) <= 2:
    print('Specify an input path as the first argument and output path as the second. \n\n' + \
          'On the input path The script expects a folder <images> containing image files \n' + \
          'and a folder <annotations> or <predictions> with NOUS json annotations of those images. \n\n' + \
          'On the output path the output file will be written and a folder with op images if selected.')
else:
    inputPath = sys.argv[1]
    print(inputPath)
    imgPath = inputPath + 'images/'
    jsonPath = inputPath + 'annotation/'
    #jsonPath = inputPath + 'prediction/'

    outputPath = sys.argv[2]
    print(outputPath)
    if not (os.path.exists(outputPath) and os.path.isdir(outputPath)):
        os.mkdir(outputPath)

    if safeOutputImg:
        imgOutputPath = outputPath + "images/"
        if not (os.path.exists(imgOutputPath) and os.path.isdir(imgOutputPath)):
            os.mkdir(imgOutputPath)

    # iterate over files in imgInputPath
    # os.walk yields tuple with rootpath, subdirs and filenames in path
    # giving file extension
    validExt = ('.png', '.jpg', '.tif', '.bmp')
    for root, subdirs, files in os.walk(jsonPath):
        for fn in files:
            # todo: Add magic to make the script safe, for now just assume the files are there, crash bigtime if not.
            jsonF = os.path.join(root, fn)

            # process all json files
            if fn.endswith('.json'):
                print("processing: " + jsonF)
                # Split the name on "_" to separate NOUS id and original filename.
                fnSplits = fn.split('_')
                maxI = len(fnSplits) - 1
                origImgF = ""
                # Rebuild original filename
                for i in range(0, maxI):
                    if i == 0:
                        origImgF = fnSplits[i]
                    else:
                        origImgF += "_" + fnSplits[i]
                print(origImgF)

                # parse the json annotations into a huge dict.
                # todo: add checks on validity and success.
                with open(jsonF) as f:
                    annot = json.load(f)

                # get the nous id and the image id for the image/annotation combination from the json
                nousId = annot.get('id')
                imgId = annot.get('image_id')

                # remove the extension from the name (Path.stem()) and set the Json file path \
                # ImgBaseName = imgPath + Path(jsonF).stem

                # Reconstruct the image filename and guess the extension.
                imgBase = origImgF + "_" + imgId
                imgBaseName = imgPath + imgBase
                print(imgBaseName)
                ext = '.jpg'
                imgName = imgBaseName + ext
                img = cv2.imread(imgName)
                # check if image read was successful and try .png if not, etc.
                if not np.any(img):
                    ext = '.png'
                    imgName = imgBaseName + ext
                    img = cv2.imread(imgName)
                elif not np.any(img):
                    ext = '.tif'
                    imgName = imgBaseName + ext
                    img = cv2.imread(imgName)
                elif not np.any(img):
                    ext = '.tiff'
                    imgName = imgBaseName + ext
                    img = cv2.imread(imgName)
                elif not np.any(img):
                    ext = '.bmp'
                    imgName = imgBaseName + ext
                    img = cv2.imread(imgName)
                elif not np.any(img):
                    print(" cannot find file fName")
                    break

                imgY = img.shape[0]
                imgX = img.shape[1]
                imgDim = [imgX, imgY]
                print(imgDim)
                # count the nr of segmentation annotations in the file
                nrAnnot = (len(annot['data']))

                if nrAnnot == 1:
                    # Get the NOUS annotation id, to separate muliple annotations later
                    annotId = annot.get('data')[0].get('id')
                    # get the polygon vertices from the json objext
                    vertices = pd.DataFrame(annot.get('data')[0].get('shapes')[0].get('geometry')['points'])

                    # Find the the vertices on the polygon that are the furthest apart
                    current_max = 0
                    v1 = [0, 0]
                    v2 = [0, 0]
                    for a, b in combinations(np.array(vertices), 2):
                        current_distance = np.linalg.norm(a - b)
                        if current_distance > current_max:
                            current_max = current_distance
                            v1 = a
                            v2 = b
                            # print(current_max)
                    print(v1)
                    print(v2)

                    # get the center of gravity
                    # todo probably change C0G/centroid to polylabel.
                    # --> CoG can be outside the polygon, polylabel should find the best CoG inside the polygon.
                    # --> might yield better results with big and strongly curved insects, such as Dragonflies.
                    # https://github.com/mapbox/polylabel

                    p = Polygon(np.array(vertices))
                    centroid = p.centroid
                    print(centroid)
                    vm = np.array(centroid)
                    # x, y = zip(v1, vm, v2)

                    # Calculate body length as the sum of the lengths of the lines from the edges to the CoG in pixels.
                    imgDim = np.array(imgDim)
                    v1 = v1 * imgDim
                    vm = vm * imgDim
                    v2 = v2 * imgDim
                    len1 = np.linalg.norm(v1 - vm)
                    len2 = np.linalg.norm(vm - v2)
                    bodyLenPx = len2 + len1
                    print('body length in pixels = ' + str(bodyLenPx))
                    bodyLenMm = bodyLenPx * pxToMmCal

                    # todo Implement calculation of the polygon surface area.

                    # generate output
                    FileName = imgBase + ext
                    outputData.loc[len(outputData.index)] = [FileName, nousId, annotId, bodyLenPx, bodyLenMm, np.nan, '']

                    # Draw the detection points and lines on the image.
                    v1 = v1.astype(int)
                    vm = vm.astype(int)
                    v2 = v2.astype(int)
                    newImg = cv2.circle(img, v1, radius=3, color=(0, 0, 255), thickness=-1)
                    newImg = cv2.circle(newImg, vm, radius=3, color=(0, 255, 0), thickness=-1)
                    newImg = cv2.circle(newImg, v2, radius=3, color=(255, 0, 0), thickness=-1)
                    newIMG = cv2.line(newImg, v1, vm, (255, 255, 255), thickness=1)
                    newIMG = cv2.line(newImg, vm, v2, (255, 255, 255), thickness=1)

                    if imgShow:
                        cv2.imshow('with points', newImg)
                        cv2.waitKey(1200)
                        cv2.destroyAllWindows()

                    if safeOutputImg:
                        newImgName = imgOutputPath + origImgF + '_' + imgId + '_' + annotId + ext
                        print('newImgname: ' + newImgName)
                        cv2.imwrite(newImgName, newImg)
                    # plt.scatter(x, y)
                    # plt.show()

                elif nrAnnot > 1:
                    # todo on multipel roi's: assume the biggest (by polygon vertices count) is the correct one.
                    # for now just make a remark in the output and do nothing :-(
                    print('more than 1 annotation')
                    FileName = imgBase + ext
                    outputData.loc[len(outputData.index)] = [FileName, nousId, np.nan, np.nan, np.nan, np.nan, \
                                                             'more than one annotation']

                else:
                    # todo, this might still crash, if empty annotations generate a json [data] object....
                    print('no annotations')
                    FileName = imgBase + ext
                    outputData.loc[len(outputData.index)] = [FileName, nousId, np.nan, np.nan, np.nan, np.nan, \
                                                             'no annotations']
    print(outputData)
    opFile = outputPath + 'insectBodyLengths.xlsx'
    outputData.to_excel(opFile, sheet_name='bodylengths')
