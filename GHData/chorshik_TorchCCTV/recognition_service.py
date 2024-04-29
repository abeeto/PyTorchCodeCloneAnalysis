import sys
from pathlib import Path
import datetime

sys.path.append(str(Path(__file__).parent))

from definitions import CONFIG_LOGGING_PATH

import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig(CONFIG_LOGGING_PATH)
logger = logging.getLogger('api')


import yaml
import cv2
import numpy as np
from lib.face_sdk.core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from lib.face_sdk.core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from lib.face_sdk.core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from lib.face_sdk.core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from lib.face_sdk.core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from lib.face_sdk.core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from lib.face_sdk.core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.full_load(f)

# common setting for all models, need not modify.
model_path = 'models'

# face detection model setting.
scene = 'non-mask'
model_category = 'face_detection'
model_name = model_conf[scene][model_category]

logger.info('Start to load the face detection model...')
try:
    faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
    model, cfg = faceDetModelLoader.load_model()
    faceDetModelHandler = FaceDetModelHandler(model, 'cpu', cfg)
except Exception as e:
    logger.error('Failed to load face detection Model.')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Success!')

# face landmark model setting.
model_category = 'face_alignment'
model_name = model_conf[scene][model_category]
logger.info('Start to load the face landmark model...')
try:
    faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
    model, cfg = faceAlignModelLoader.load_model()
    faceAlignModelHandler = FaceAlignModelHandler(model, 'cpu', cfg)
except Exception as e:
    logger.error('Failed to load face landmark model.')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Success!')

# face recognition model setting.
model_category = 'face_recognition'
model_name = model_conf[scene][model_category]
logger.info('Start to load the face recognition model...')
try:
    faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
    model, cfg = faceRecModelLoader.load_model()
    faceRecModelHandler = FaceRecModelHandler(model, 'cpu', cfg)
except Exception as e:
    logger.error('Failed to load face recognition model.')
    logger.error(e)
    sys.exit(-1)
else:
    logger.info('Success!')

import pickle
import os
import imutils.paths as paths
from datetime import datetime

imagePaths = list(paths.list_images('data/faces'))
known_face_feature = []
known_face_metadata = []
face_cropper = FaceRecImageCropper()


def create_database():
    for (i, imagePath) in enumerate(imagePaths):
        name = imagePath.split(os.path.sep)[-1].split('.')[0]

        image = cv2.imread(imagePath)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        dets = faceDetModelHandler.inference_on_image(image)
        face_nums = dets.shape[0]

        for i in range(face_nums):
            landmarks = faceAlignModelHandler.inference_on_image(image, dets[i])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cropped_image = face_cropper.crop_image_by_mat(image, landmarks_list)

            face_image = cv2.resize(cropped_image, (150, 150), fx=0.25, fy=0.25)

            feature = faceRecModelHandler.inference_on_image(cropped_image)

            known_face_feature.append(feature)
            known_face_metadata.append({
                "ident": name,
                "first_seen_this_interaction": datetime.now(),
                "last_seen": datetime.now(),
                "seen_count": 1,
                "seen_frames": 1,
                "face_image": face_image,
            })

    data = [np.array(known_face_feature), known_face_metadata]

    f = open("data/database/faces_jetson2.dat", "wb")
    f.write(pickle.dumps(data))
    f.close()


#create_database()


def load_known_faces():
    global known_face_feature, known_face_metadata

    try:
        with open("data/database/faces_jetson2.dat", "rb") as face_data_file:
            known_face_feature, known_face_metadata = pickle.load(face_data_file)
            print("Known faces.dat loaded from disk.")
    except FileNotFoundError as e:
        print("No previous face data found - starting with a blank known face list.")
        pass


video_capture = cv2.VideoCapture(0)

bbox = None
face_names = []
process_this_frame = True

load_known_faces()

while True:
    ret, frame = video_capture.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        dets = faceDetModelHandler.inference_on_image(small_frame)
        face_nums = dets.shape[0]
        bbox = dets
        feature_list = []
        for i in range(face_nums):
            landmarks = faceAlignModelHandler.inference_on_image(small_frame, dets[i])
            landmarks_list = []
            for (x, y) in landmarks.astype(np.int32):
                landmarks_list.extend((x, y))
            cropped_image = face_cropper.crop_image_by_mat(small_frame, landmarks_list)
            feature = faceRecModelHandler.inference_on_image(cropped_image)
            feature_list.append(feature)

        face_names = []
        for feature in feature_list:
            scores = np.dot(known_face_feature, feature.transpose())
            score = scores.max()
            name = "Unknown"
            if score > 0.5:
                metadata = known_face_metadata[scores.argmax()]
                name = metadata["ident"]
            face_names.append(name)
            logger.info('The similarity score of two faces: %f' % score)
    process_this_frame = not process_this_frame

    # Display the results
    for box, name in zip(bbox, face_names):
        box = list(map(int, box))
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top = box[1] * 4
        right = box[2] * 4
        bottom = box[3] * 4
        left = box[0] * 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
