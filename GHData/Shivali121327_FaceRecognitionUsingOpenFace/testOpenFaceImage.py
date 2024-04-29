import os,time
import sys
import dlib
import cv2
import numpy as np
import faceBlendCommon as fbc

facerecModelPath='../data/models/openface.nn4.small2.v1.t7'
recModel = cv2.dnn.readNetFromTorch(facerecModelPath)
recMean = [0,0,0]
recSize = (96, 96)
recScale = 1/255.0
recThreshold = 0.8


# Initialize face detector
faceDetector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "../data/models/shape_predictor_5_face_landmarks.dat"
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# load descriptors and index file generated during enrollment
index = np.load('index_openface.pkl')
faceDescriptorsEnrolled = np.load('descriptors_openface.npy')

# read image
if len(sys.argv) > 1:
  imagePath = sys.argv[1]
else:
  imagePath = '../data/images/faces/face_demo.jpg'
im = cv2.imread(imagePath)

# exit if unable to read frame from feed
if im is None:
  print('cannot read image')
  sys.exit(0)

t = time.time()
# detect faces in image
faces = faceDetector(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))

print("{} Face(s) found".format(len(faces)))
# Now process each face we found
for face in faces:

  # find coordinates of face rectangle
  x1 = face.left()
  y1 = face.top()
  x2 = face.right()
  y2 = face.bottom()

  alignedFace = fbc.alignFace(im, face, landmarkDetector, recSize)

  # Compute face descriptor using neural network defined in Dlib.
  # It is a 128D vector that describes the face in img identified by shape.
  blob = cv2.dnn.blobFromImage(alignedFace, recScale, recSize, recMean, False, False)
  recModel.setInput(blob)
  faceDescriptorQuery = recModel.forward()

  # Calculate Euclidean distances between face descriptor calculated on face dectected
  # in current frame with all the face descriptors we calculated while enrolling faces
  distances = np.linalg.norm(faceDescriptorsEnrolled - faceDescriptorQuery, axis=1)
  # Calculate minimum distance and index of this face
  argmin = np.argmin(distances)  # index
  minDistance = distances[argmin]  # minimum distance

  # If minimum distance if less than threshold
  # find the name of person from index
  # else the person in query image is unknown
  if minDistance <= recThreshold:
    label = index[argmin]
  else:
    label = 'unknown'

  print("time taken = {:.3f} seconds".format(time.time() - t))

  # Draw a rectangle for detected face
  cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255))

  # Draw circle for face recognition
  center = (int((x1 + x2)/2.0), int((y1 + y2)/2.0))
  radius = int((y2-y1)/2.0)
  color = (0, 255, 0)
  cv2.circle(im, center, radius, color, thickness=1, lineType=8, shift=0)

  # Write test on image specifying identified person and minimum distance
  org = (int(x1), int(y1))  # bottom left corner of text string
  font_face = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 0.8
  text_color = (255, 0, 0)
  printLabel = '{} {:0.4f}'.format(label, minDistance)
  cv2.putText(im, printLabel, org, font_face, font_scale, text_color, thickness=2)

# Show result
cv2.imshow('facerec', im)
cv2.imwrite('result-openface-{}.jpg'.format(label), im)
cv2.waitKey(0)
cv2.destroyAllWindows()
