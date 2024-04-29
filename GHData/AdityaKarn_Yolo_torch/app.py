from PIL import Image
import os , io , sys
from werkzeug.utils import secure_filename
from yolo import process
from datetime import datetime
from random import randint
import shutil
import requests
import cv2
import base64

uploads_dir = './instance/uploads'
output_dir = './instance/output'

image_path = 'image.jpg'

def yolo_processing(cv2_img):

    cv2.imwrite(image_path, cv2_img)
    
    now = datetime.now()
    filename = now.strftime("%Y%m%d_%H%M%S") + "_" + str(randint(000, 999))
    src_dir= image_path
    dst_dir= os.path.join(uploads_dir, (filename + '.jpg'))
    shutil.copy(src_dir,dst_dir)
    # file.save(os.path.join(uploads_dir, secure_filename(filename + '.jpg')))
    objects_count, objects_confidence = process(uploads_dir, output_dir, filename)
    
    response = {
        'objects_count': objects_count, 
        'objects_confidence': objects_confidence, 
        'filename': filename + '.jpg'
    }

    dst_dir = os.path.join(output_dir, (filename + '.jpg'))
    
    # print(response)

    cv2_img = cv2.imread(dst_dir)
    

    result = {}

    result['response'] = response
    result['img'] = cv2_img

    return result