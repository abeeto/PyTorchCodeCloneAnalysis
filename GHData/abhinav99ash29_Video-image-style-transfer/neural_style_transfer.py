# python neural_style_transfer.py --image image_path --model models/model_path


import argparse
import imutils
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="neural style transfer model")
ap.add_argument("-i", "--image", required=True,
	help="input image to apply neural style transfer to")
args = vars(ap.parse_args())


print("[INFO] loading style transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])


image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
(h, w) = image.shape[:2]


blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
	(103.939, 116.779, 123.680), swapRB=False, crop=False)
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()


output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680
output /= 255.0
output = output.transpose(1, 2, 0)


print("[INFO] neural style transfer took {:.4f} seconds".format(
	end - start))


cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)
