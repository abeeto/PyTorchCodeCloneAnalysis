import os
import cv2
import imutils
import pickle
import face_recognition
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

from face_align import align
from arcface_net import ArcFaceNet
from imutils.video import WebcamVideoStream
from sklearn.decomposition import PCA


model = ArcFaceNet(num_classes=10)
# checkpoint = torch.load('arcface_pytorch.pt', map_location=torch.device('cpu'))
# checkpoint.eval()
# print(checkpoint.keys())
model.load_state_dict(torch.load('arcface_pytorch.pt', map_location=torch.device("cpu")))
# model.eval()
model.eval()

HEIGHT, WIDTH, CHANNELS = 128, 128, 3
FACE_FILE_PICKLE = 'faces.pickle'
LABEL_FILE_PICKLE = 'labels.pickle'
HAAR = 'haarcascade_frontalface_default.xml'
DATA_DIR = 'faces/'


faces = list()
labels = list()
known_encodings = list()

def lumination_neutralize(img):
	lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
	l, a, b = cv2.split(lab)

	clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
	l_clahe = clahe.apply(l)

	lab_clahe = cv2.merge((l_clahe, a, b))
	bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

	return bgr

def detect_face(img):
	haar = cv2.CascadeClassifier(HAAR)
	
	faces = haar.detectMultiScale(img, scaleFactor=1.05, minNeighbors=5)
	if(len(faces) == 0):
		return None, None
	else:
		(x,y,w,h) = faces[0]
		face = img[y:y+h, x:x+w]
		return (x,y,w,h), face

if(not os.path.exists(FACE_FILE_PICKLE) or not os.path.exists(LABEL_FILE_PICKLE)):
	for (dir, dirs, files) in os.walk(DATA_DIR):
		if(dir != DATA_DIR):
			for file in files:
				abs_img_path = dir + '/' + file
				print("[INFO] Reading " + abs_img_path)
				
				img = cv2.imread(abs_img_path)
				faces_, face_locations_ = align(img, width=WIDTH, height=HEIGHT, operation=None)
				face = faces_[0]
				faces.append(face)
				label = dir.split('/')[-1]
				
				labels.append(label)
				
	pickle.dump(faces, open(FACE_FILE_PICKLE, "wb"))
	pickle.dump(labels, open(LABEL_FILE_PICKLE, "wb"))
	
else:
	faces = pickle.load(open(FACE_FILE_PICKLE, "rb"))
	labels = pickle.load(open(LABEL_FILE_PICKLE, "rb"))
	
faces = np.array(faces)
labels = np.array(labels)
print(labels)

### IMPORTANTO : Pytorch preprocessing steps ###
faces = torch.Tensor(faces)
faces = faces.reshape(-1, CHANNELS, HEIGHT, WIDTH)
# labels = torch.Tensor(labels).type(torch.LongTensor)

pca = PCA(n_components=3)

outputs = model(faces).detach().numpy()
outputs /= np.linalg.norm(outputs, axis=1, keepdims=True)

# now save all the encodings
known_encodings = None
if(not os.path.exists("encodings.pickle")):
	pickle.dump(outputs, open("encodings.pickle", "wb"))
	known_encodings = outputs
else:
	known_encodings = pickle.load(open("encodings.pickle", "rb"))

emb = pca.fit_transform(outputs)
emb /= np.linalg.norm(emb, axis = 1, keepdims=True)

ax = plt.axes(projection='3d')

print("[INFO] Visualizing results ... ")
for label in np.unique(labels):
	cluster = emb[np.where(labels == label)]
	ax.scatter3D(cluster[:,0], cluster[:,1], cluster[:,2], label=label)
	
### After visualizing now comes the recognizing part ###
print("[INFO] Starting camera stream ... ")
vs = WebcamVideoStream(src=0).start()
detector = cv2.dnn.readNetFromCaffe("deploy.prototxt", "dnn_model.caffemodel")

PROCESS_FRAME = True
face_locations = list()
face_names = list()
while(True):
	frame = vs.read()
	(H, W) = frame.shape[:2]

	if(PROCESS_FRAME):
		face_locations = list()
		face_names = list()

		faces, face_locations = align(frame, width=WIDTH, height=HEIGHT, operation=None)

		for face in faces:
			cv2.imshow("Face", face)
			face = np.array([face])
			face = torch.Tensor(face).reshape(1, CHANNELS, HEIGHT, WIDTH)
			encoding = model(face).detach().numpy()[0]

			### Encoding normalization ###
			encoding = encoding / np.linalg.norm(encoding)

			matches = face_recognition.compare_faces(known_encodings, encoding, tolerance = 0.3)
			face_distances = face_recognition.face_distance(known_encodings, encoding)

			emb = pca.transform([encoding])
			emb /= np.linalg.norm(emb)
			ax.scatter3D(emb[:,0], emb[:,1], emb[:,2], color='brown', alpha=1.0)

			best_match = np.argmin(face_distances)
			print(face_distances)

			label = 'Unknown'
			if(matches[best_match]):
				label = labels[best_match]

			face_names.append(label)

	for (startX, startY, endX, endY), name in zip(face_locations, face_names):
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
		cv2.putText(frame, name, (startX, startY), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0,0,255))


	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1)

	if(key == ord("q")):
		break

vs.stop()
cv2.destroyAllWindows()
plt.legend()
plt.show()
