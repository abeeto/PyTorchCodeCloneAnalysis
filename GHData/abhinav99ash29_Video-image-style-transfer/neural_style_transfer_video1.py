from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import imutils
import time
import cv2


modelPaths = paths.list_files('models', validExts=(".t7",))
modelPaths = sorted(list(modelPaths))

models = list(zip(range(0, len(modelPaths)), (modelPaths)))

modelIter = itertools.cycle(models)
(modelID, modelPath) = next(modelIter)





print("Loading style transfer model...")
net = cv2.dnn.readNetFromTorch(modelPath)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print('Models loaded !')


print("Starting video stream...")
vs = VideoStream(src=0).start()
##vs = cv2.VideoCapture(0)
time.sleep(2.0)
print("{}. {}".format(modelID + 1, modelPath))

while True:
	
	frame = vs.read()

	frame = imutils.resize(frame, width=600)
	orig = frame.copy()
	(h, w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(frame, 1.0, (w, h),
		(103.939, 116.779, 123.680), swapRB=False, crop=False)
	net.setInput(blob)
	output = net.forward()

	output = output.reshape((3, output.shape[2], output.shape[3]))
	output[0] += 103.939
	output[1] += 116.779
	output[2] += 123.680
	output /= 255.0
	output = output.transpose(1, 2, 0)

	cv2.imshow("Input", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("a"):
		(modelID, modelPath) = next(modelIter)
		print("{}. {}".format(modelID + 1, modelPath))
		net = cv2.dnn.readNetFromTorch(modelPath)

	elif key == 27:
		break

vs.stop()
cv2.destroyAllWindows()
##vs.release()

