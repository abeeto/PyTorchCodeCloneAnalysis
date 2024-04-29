import requests
import cv2
import time
import datetime as dt
import base64
from utils.utils import FileControl

path = FileControl()

ip = '211.55.94.113'
url = f"http://{ip}:5000/tracking"
location = 0
use_webCam = False
if use_webCam:
    file_name = 0
else:
    file_name = path.get_video_path('box.mp4')
video_capture = cv2.VideoCapture(file_name)

payload = {}
headers = {}
idx = 0
start_point = 0
now_time = dt.datetime.now()
while True:
    ret, frame = video_capture.read()
    if not ret:
        break
    if idx >= start_point:
        _, img_encoded = cv2.imencode('.jpg', frame)

        files = [
            ('file', (f'frame{idx}.jpg', base64.b64encode(img_encoded).decode('ascii'), 'image/jpeg'))
        ]
        # time = '0000_00_00_00_00_00_00'
        time_ = now_time + dt.timedelta(seconds=0.033*idx)
        time_ = time_.strftime("%Y_%m_%d_%H_%M_%S_%f")[:-4]
        url_full = url + f'?location={location}&time={time_}'

        check = True
        while check:
            try:
                start = time.time()
                print(url_full)
                response = requests.request("GET", url_full, headers=headers, data=payload, files=files)

                # print(response.text)
                print('request time : ', time.time() - start)
                check = False
                idx += 1
            except Exception as e:
                print(e)
                cv2.waitKey(1)
