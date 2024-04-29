import cv2
import torch
import os
# import dlib
from models import Autoencoder, toTensor, var_to_np
from image_augmentation import random_warp
import numpy as np

video_name = 'train/trump.mp4'
video_path = os.path.join(os.path.realpath('.'), video_name)
#/home/hanqing/deepfake/Faceswap-Deepfake-Pytorch/train/liu.mp4

##################################################
# Extract faces
##################################################
device = torch.device('cuda:0')
def extract_face(frame):
    image = cv2.imread(frame)
    image = imutils.resize(image, width=600)
    (h, w) = image.shape[:2]
    
    # construct a blob from the image
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 1.0, (300, 300),
        (104.0, 177.0, 123.0), swapRB=False, crop=False)
    
    # apply OpenCV's deep learning-based face detector to localize
    # faces in the input image
    detector.setInput(imageBlob)
    detections = detector.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]
    
        # filter out weak detections
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            position = {}
            position['left'] = startX
            position['top'] = startY
            position['right'] = endX
            position['bot'] = endY

            # extract the face ROI
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
    
            # ensure the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            return position, face

    # detector = dlib.get_frontal_face_detector()
    # img = frame
    # dets = detector(img, 1)
    # for idx, face in enumerate(dets):
    #     position = {}
    #     position['left'] = face.left()
    #     position['top'] = face.top()
    #     position['right'] = face.right()
    #     position['bot'] = face.bottom()
    #     croped_face = img[position['top']:position['bot'], position['left']:position['right']]

    #     return position, croped_face


def extract_faces(video_path):
    cap = cv2.VideoCapture(video_path)
    n = 0
    while (cap.isOpened() and n<1000):
        _, frame = cap.read()
        position, croped_face= extract_face(frame)
        #print(croped_face.shape)
        #exit(0)
        #cv2.imshow("croped_face",croped_face)
        #cv2.waitKey(2000)
        #cv2.destroyAllWindows()
        converted_face = convert_face(croped_face)
        converted_face = converted_face.squeeze(0)
        converted_face = var_to_np(converted_face)
        converted_face = converted_face.transpose(1,2,0)
        converted_face = np.clip(converted_face * 255, 0, 255).astype('uint8')
        cv2.imshow("converted_face", cv2.resize(converted_face, (256,256)))
        cv2.waitKey(2000)
        back_size = cv2.resize(converted_face, (croped_face.shape[0]-120, croped_face.shape[1]-120))
        #cv2.imshow("back_face", back_size)
        #cv2.waitKey(1000)
        #print(frame.shape)
        #print(back_size.shape)
        merged = merge(position, back_size, frame)
        #print(merged.shape)
        out.write(merged)
        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        n = n + 1
        print(n)

def convert_face(croped_face):
    resized_face = cv2.resize(croped_face, (256, 256))
    normalized_face = resized_face / 255.0
    #normalized_face = normalized_face.reshape(1, normalized_face.shape[0], normalized_face.shape[1], normalized_face.shape[2])
    warped_img, _ = random_warp(normalized_face)
    batch_warped_img = np.expand_dims(warped_img, axis=0)

    batch_warped_img = toTensor(batch_warped_img)
    batch_warped_img = batch_warped_img.to(device).float()
    #print(batch_warped_img.shape, batch_warped_img)
    model = Autoencoder().to(device)
    checkpoint = torch.load('./checkpoint/autoencoder.t7')
    model.load_state_dict(checkpoint['state'])

    converted_face = model(batch_warped_img,'B')
    #print(converted_face.size())
    return converted_face

def merge(postion, face, body):
    #im = cv2.imread("train/images/wood-texture.jpg")
    #obj = cv2.imread("train/images/iloveyouticket.jpg")
    #print(body.shape)
    mask = 255 * np.ones(face.shape, face.dtype)
    width, height, channels = body.shape
    center = (postion['left']+(postion['right']-postion['left'])//2, postion['top']+(postion['bot']-postion['top'])//2)
    normal_clone = cv2.seamlessClone(face, body, mask, center, cv2.NORMAL_CLONE)
    return normal_clone

# def InitwriteVideo():
#     fourcc = cv2.VideoWriter_fourcc()
#     frame_width = int(cap.get(3))
#     frame_height = int(cap.get(4))

if __name__ == "__main__":
    print(video_path)
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter("me4_out.avi", fourcc, 3, (1920, 1080))
    extract_faces(video_path)

    out.release()