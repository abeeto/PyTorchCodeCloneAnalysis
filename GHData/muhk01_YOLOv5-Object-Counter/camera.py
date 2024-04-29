import os
import cv2
from base_camera import BaseCamera
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
import paho.mqtt.client as mqtt
import json
from utils.datasets import *
from utils.utils import *
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
import time
from module.database import Database
from datetime import datetime

useIPCam = False

MQTT_TOPIC = "="
mylist = []
mycount = []
broker_url = "="
broker_port = 1883

username = ""
password = '' 

client = mqtt.Client()
client.username_pw_set(username)
client.connect(broker_url, broker_port)
client.loop_start()

db = Database()

class Camera(BaseCamera):
	def __init__(self):
		if os.environ.get('OPENCV_CAMERA_SOURCE'):
			Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
		super(Camera, self).__init__()

	def generateCentroid(rects):
		inputCentroids = np.zeros((len(rects), 2), dtype="int")
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			cX = int((startX + endX) / 2.0)
			cY = int((startY + endY) / 2.0)
			inputCentroids[i] = (cX, cY)
		return inputCentroids

	def frames():
		global useIPCam
		out, weights, imgsz = \
		'inference/output', 'weights/yolov5s.pt', 640
		if(useIPCam):
			source = 'stream.txt'
		else:
			source = 'traff.mp4'
		device = torch_utils.select_device()
		if os.path.exists(out):
			shutil.rmtree(out)  # delete output folder
		os.makedirs(out)  # make new output folder
		start = time.time()  
		elapsed = 0
		# Load model
		google_utils.attempt_download(weights)
		model = torch.load(weights, map_location=device)['model']
		
		model.to(device).eval()

		# Second-stage classifier
		classify = False
		if classify:
			modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
			modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
			modelc.to(device).eval()

		# Half precision
		half = False and device.type != 'cpu' 
		print('half = ' + str(half))

		if half:
			model.half()

		# Set Dataloader
		vid_path, vid_writer = None, None
		if(useIPCam):
			dataset = LoadStreams(source, img_size=imgsz)
		else:
			dataset = LoadImages(source, img_size=imgsz)
		
		names = model.names if hasattr(model, 'names') else model.modules.names
		colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

		# Run inference
		t0 = time.time()
		ct = CentroidTracker()
		listDet = ['person','bicycle','car','motorcycle','bus','truck']

		totalDownPerson = 0
		totalDownBicycle = 0
		totalDownCar = 0
		totalDownMotor = 0
		totalDownBus = 0
		totalDownTruck = 0

		totalUpPerson = 0
		totalUpBicycle = 0
		totalUpCar = 0
		totalUpMotor = 0
		totalUpBus = 0
		totalUpTruck = 0
		pub = False
		trackableObjects = {}
		img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
		_ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
		for path, img, im0s, vid_cap in dataset:
			elapsed = time.time() - start
			img = torch.from_numpy(img).to(device)
			img = img.half() if half else img.float()  # uint8 to fp16/32
			img /= 255.0  # 0 - 255 to 0.0 - 1.0
			if img.ndimension() == 3:
				img = img.unsqueeze(0)

			
			# Inference
			t1 = torch_utils.time_synchronized()
			pred = model(img, augment=False)[0]
			
			# Apply NMS
			pred = non_max_suppression(pred, 0.4, 0.5,
							   fast=True, classes=None, agnostic=False)
			t2 = torch_utils.time_synchronized()
		   
			# Apply Classifier
			if classify:
				pred = apply_classifier(pred, modelc, img, im0s)

			rects = []
			labelObj = []
			yObj = []
			arrCentroid = []
			for i, det in enumerate(pred):  # detections per image
				
				if(useIPCam):
					p, s, im0 = path[i], '%g: ' % i, im0s[i].copy() #if rtsp/camera
				else:
					p, s, im0 = path, '', im0s	
				height, width, channels = im0.shape
				cv2.line(im0, (0, int(height/1.5)), (int(width), int(height/1.5)), (0, 0, 0), thickness=1)
				save_path = str(Path(out) / Path(p).name)
				s += '%gx%g ' % img.shape[2:]  # print string
				gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
				if det is not None and len(det):
					# Rescale boxes from img_size to im0 size
					det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

					for c in det[:, -1].unique():
						n = (det[:, -1] == c).sum()  # detections per class
						s += '%g %s, ' % (n, names[int(c)])  # add to string                        
					for *xyxy, conf, cls in det:
						label = '%s %.2f' % (names[int(cls)], conf)
						x = xyxy
						tl = None or round(0.002 * (im0.shape[0] + im0.shape[1]) / 2) + 1  # line/font thickness
						c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

						label1 = label.split(' ')
						if label1[0] in listDet:
							box = (int(x[0]), int(x[1]), int(x[2]), int(x[3]))
							rects.append(box)
							labelObj.append(label1[0])
							cv2.rectangle(im0, c1 , c2, (0,0,0), thickness=tl, lineType=cv2.LINE_AA)
							tf = max(tl - 1, 1)  
							t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
							c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
							cv2.rectangle(im0, c1, c2, (0,100,0), -1, cv2.LINE_AA)
							cv2.putText(im0, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

				detCentroid = Camera.generateCentroid(rects)
				objects = ct.update(rects)  
			
				for (objectID, centroid) in objects.items():
					arrCentroid.append(centroid[1])
				for (objectID, centroid) in objects.items():
					#print(idxDict)
					to = trackableObjects.get(objectID, None)
					if to is None:
						to = TrackableObject(objectID, centroid)
					else:           
						y = [c[1] for c in to.centroids]
						direction = centroid[1] - np.mean(y)
						to.centroids.append(centroid)
						if not to.counted: #arah up
							
							if direction < 0 and centroid[1] < height / 1.5 and centroid[1] > height / 1.7: ##up truble when at distant car counted twice because bbox reappear
								idx = detCentroid.tolist().index(centroid.tolist())
								if(labelObj[idx] == 'person'):
									totalUpPerson += 1
									to.counted = True
								elif(labelObj[idx] == 'bicycle'):
									totalUpBicycle += 1
									to.counted = True
								elif(labelObj[idx] == 'car'):
									totalUpCar += 1
									to.counted = True
								elif(labelObj[idx] == 'motorbike'):
									totalUpMotor += 1
									to.counted = True
								elif(labelObj[idx] == 'bus'):
									totalUpBus += 1
									to.counted = True
								elif(labelObj[idx] == 'truck'):
									totalUpTruck += 1
									to.counted = True
							
							elif direction > 0 and centroid[1] > height / 1.5:  #arah down
								idx = detCentroid.tolist().index(centroid.tolist())
								if(labelObj[idx] == 'person'):
									totalDownPerson += 1
									to.counted = True
								elif(labelObj[idx] == 'bicycle'):
									totalDownBicycle += 1
									to.counted = True
								elif(labelObj[idx] == 'car'):
									totalDownCar += 1
									to.counted = True
								elif(labelObj[idx] == 'motorbike'):
									totalDownMotor += 1
									to.counted = True
								elif(labelObj[idx] == 'bus'):
									totalDownBus += 1
									to.counted = True
								elif(labelObj[idx] == 'truck'):
									totalDownTruck += 1
									to.counted = True

					trackableObjects[objectID] = to

				cv2.putText(im0, 'Down Person : ' + str(totalDownPerson), (int(width * 0.7) , int(height * 0.05)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Down bicycle : ' + str(totalDownBicycle), (int(width * 0.7) , int(height * 0.1)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Down car : ' + str(totalDownCar), (int(width * 0.7) , int(height * 0.15)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Down motorbike : ' + str(totalDownMotor), (int(width * 0.7) , int(height * 0.2)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Down bus : ' + str(totalDownBus), (int(width * 0.7) , int(height * 0.25)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Down truck : ' + str(totalDownTruck), (int(width * 0.7) , int(height * 0.3)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)

				cv2.putText(im0, 'Up Person : ' + str(totalUpPerson), (int(width * 0.02) , int(height * 0.05)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Up bicycle : ' + str(totalUpBicycle), (int(width * 0.02) , int(height * 0.1)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Up car : ' + str(totalUpCar), (int(width * 0.02) , int(height * 0.15)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Up motorbike : ' + str(totalUpMotor), (int(width * 0.02) , int(height * 0.2)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Up bus : ' + str(totalUpBus), (int(width * 0.02) , int(height * 0.25)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				cv2.putText(im0, 'Up truck : ' + str(totalUpTruck), (int(width * 0.02) , int(height * 0.3)),
						cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 100), 2)
				#print(elapsed)
				if(elapsed > 60):
					ObjListku = ['Person','Bicycle','Car','Motorbike','Bus','Truck']
					objCountUp = []
					objCountDown = []
					objCountDown.append(totalDownPerson)
					objCountDown.append(totalDownBicycle)
					objCountDown.append(totalDownCar)
					objCountDown.append(totalDownMotor)
					objCountDown.append(totalDownBus)
					objCountDown.append(totalDownTruck)

					objCountUp.append(totalUpPerson)
					objCountUp.append(totalUpBicycle)
					objCountUp.append(totalUpCar)
					objCountUp.append(totalUpMotor)
					objCountUp.append(totalUpBus)
					objCountUp.append(totalUpTruck)

					date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

					totalDownPerson = 0
					totalDownBicycle = 0
					totalDownCar = 0
					totalDownMotor = 0
					totalDownBus = 0
					totalDownTruck = 0

					totalUpPerson = 0
					totalUpBicycle = 0
					totalUpCar = 0
					totalUpMotor = 0
					totalUpBus = 0
					totalUpTruck = 0

					elapsed = 0
					start = time.time()
					#db.insert(date,ObjListku,objCountUp,objCountDown) #insert ke module database
					date = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
					data_set = {"Timestamp" : str(date), "dPerson": totalDownPerson, "dBicycle": totalDownBicycle, "dCar": totalDownCar, "dBus" : totalDownBus, "dTruck" : totalDownTruck, "uPerson": totalUpPerson, "uBicycle": totalUpBicycle, "uCar": totalUpCar, "uBus" : totalUpBus, "uTruck" : totalUpTruck}
					MQTT_MSG = json.dumps(data_set)
					client.publish(MQTT_TOPIC, MQTT_MSG)
				#time.sleep(00.1)
				if pub == False:
					proc = subprocess.Popen('ffmpeg -re -f mjpeg -i http://0.0.0.0:5000/video_feed -f lavfi -i anullsrc -c:v libx264 -g 60 -c:a aac -ar 44100 -ac 2 -f flv rtmp://your-rtmp-server', shell=True)
					pub = True
			yield cv2.imencode('.jpg', cv2.resize(im0,(800,600)))[1].tobytes()
