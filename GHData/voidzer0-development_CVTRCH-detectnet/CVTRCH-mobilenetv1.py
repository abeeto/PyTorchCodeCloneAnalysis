# Detectnet SSD300 metVGG16 basenet via Torch en OpenCV voor .pth modellen als toetsing voor TRT detectnet na ONNX conversie.

from urllib.request import proxy_bypass
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer
import cv2
import sys
import os

print("Started OpenCV Torch detectnet version 0.1")
print("The goal of this application is running standardized SSD models")

if len(sys.argv) < 5:
    print('Usage: python CVTRCH-detectnet.py.py  <model .pth path> <label.txt path> <images path> <output path>')
    sys.exit(0)

mdlPath = sys.argv[1]
labelstxtPath = sys.argv[2]
imagesFolder = sys.argv[3]
outputFolder = sys.argv[4]

os.chdir(outputFolder)

print("Outdir set: " + outputFolder)

classes = [name.strip() for name in open(labelstxtPath).readlines()]

detectionNetwork = create_mobilenetv1_ssd(len(classes), is_test=True)
detectionNetwork.load(mdlPath)

predictor = create_mobilenetv1_ssd_predictor(detectionNetwork, candidate_size=200) #predictor

#loop through images in folder and feed them to the predictor with openCV
queue = []

for file in os.listdir(imagesFolder):
    objectPath = os.path.join(imagesFolder, file)
    if os.path.isfile(objectPath):
        queue.append(objectPath)
        print("Enqueued image located at: " + objectPath)

#read paths in queue and run ssd300 algo
iFrame = 0
for iterPath in queue: 
    iFrame += 1
    print("Processing image: " + iterPath)
    oImage = cv2.imread(iterPath) #load
    processedImage = cv2.cvtColor(oImage, cv2.COLOR_BGR2RGB) #preprocessing

    #finished running dataupdate phase, this is where the bulk of the work gets done
    bboxes, labels, probability = predictor.predict(processedImage, 10, 0.4) # run prediction on processed image

    #postprocessing
    for iterDetection in range(bboxes.size(0)):
        box = bboxes[iterDetection, :]
        cv2.rectangle(oImage, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 4)
        label = f"{classes[labels[iterDetection]]} p[{probability[iterDetection]:.2f}]"
        cv2.putText(oImage, label, (int(box[0]), int(box[1]) - 20), cv2.FONT_HERSHEY_DUPLEX,1,(0, 0, 255),2) 

    outName = str(iFrame) + ".jpg"
    print("Write result: " + outName)
    cv2.imwrite(outName, oImage)
    print("Detected " + str(len(probability)) + " objects in " + iterPath)
