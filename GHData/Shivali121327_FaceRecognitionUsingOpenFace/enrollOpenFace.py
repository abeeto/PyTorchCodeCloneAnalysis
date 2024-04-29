import os,sys
import dlib
import cv2
import numpy as np
import faceBlendCommon as fbc

try:
  import cPickle  # Python 2
except ImportError:
  import _pickle as cPickle  # Python 3

# Path to landmarks and face recognition model files
facerecModelPath='../data/models/openface.nn4.small2.v1.t7'
recModel = cv2.dnn.readNetFromTorch(facerecModelPath)
recMean = [0,0,0]
recSize = (96, 96)
recScale = 1/255.0

# Initialize face detector
faceDetector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "../data/models/shape_predictor_5_face_landmarks.dat"
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)


# Now let's prepare our training data
# data is organized assuming following structure
# faces folder has subfolders.
# each subfolder has images of a person
faceDatasetFolder = '../data/images/faces'

# read subfolders in folder "faces"
subfolders = []
for x in os.listdir(faceDatasetFolder):
  xpath = os.path.join(faceDatasetFolder, x)
  if os.path.isdir(xpath):
    subfolders.append(xpath)

# nameLabelMap is dict with keys as person's name
# and values as integer label assigned to this person
# labels contain integer labels for corresponding image in imagePaths
nameLabelMap = {}
labels = []
imagePaths = []
for i, subfolder in enumerate(subfolders):
  for x in os.listdir(subfolder):
    xpath = os.path.join(subfolder, x)
    if x.endswith('jpg'):
      imagePaths.append(xpath)
      labels.append(i)
      nameLabelMap[xpath] = subfolder.split('/')[-1]

# Process images one by one
# We will store face descriptors in an ndarray (faceDescriptors)
# and their corresponding labels in dictionary (index)
index = {}
i = 0
faceDescriptors = None
for imagePath in imagePaths:
  print("processing: {}".format(imagePath))
  # read image and convert it to RGB
  img = cv2.imread(imagePath)

  # detect faces in image
  faces = faceDetector(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

  print("{} Face(s) found".format(len(faces)))

  # Now process each face we found
  for k, face in enumerate(faces):

    alignedFace = fbc.alignFace(img, face, landmarkDetector, recSize)

    # Compute face descriptor using OpenFace network.
    # It is a 128D vector that describes the face.
    blob = cv2.dnn.blobFromImage(alignedFace, recScale, recSize, recMean, False, False)
    recModel.setInput(blob)
    faceDescriptor = recModel.forward()

    # Stack face descriptors (1x128) for each face in images, as rows
    if faceDescriptors is None:
      faceDescriptors = faceDescriptor
    else:
      faceDescriptors = np.concatenate((faceDescriptors, faceDescriptor), axis=0)

    # save the label for this face in index. We will use it later to identify
    # person name corresponding to face descriptors stored in NumPy Array
    index[i] = nameLabelMap[imagePath]
    i += 1

# Write descriors and index to disk
np.save('descriptors_openface.npy', faceDescriptors)
# index has image paths in same order as descriptors in faceDescriptors
with open('index_openface.pkl', 'wb') as f:
  cPickle.dump(index, f)
