import torchvision
import cv2
import torch
import argparse
import time
import detect_utils
from PIL import Image

#memilih device gpu atau cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#mengatur inputan di terminal
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800,help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# download or load the model from disk
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,min_size=args['min_size'])
# model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True,min_size=args['min_size'])
model = model.eval().to(device)     # load the model onto the computation device

# memanggil video menggunkan openCV
cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
frame_width = int(cap.get(3))                                                   #mendapatkan lebar frame
frame_height = int(cap.get(4))                                                  #mendapatkan tinggi frame
save_name = f"{args['input'].split('/')[-1].split('.')[0]}_{args['min_size']}"  #saving input
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))                              # define codec and create VideoWriter object 

frame_count = 0                     # menghitung jumlah frame
total_fps = 0                       # mendapatkan final FPS
          
while(cap.isOpened()):              # membaca vidoe sampai berakhir
    ret, frame = cap.read()         # capture tiap frame pada video
    if ret == True:
        start_time = time.time()    #start time
        with torch.no_grad():
            boxes, classes, labels = detect_utils.predict(frame, model, device, 0.8) # get predictions for the current frame
            image = detect_utils.draw_boxes(boxes, classes, labels, frame)           # draw boxes and show current frame on screen
            counter = detect_utils.Counter(classes, nama_kelas = 'car')# counter class

        end_time = time.time()                                                       # get the end time
        fps = 1 / (end_time - start_time)                                            # get the end time
        total_fps += fps                                                             # get the end time       
        frame_count += 1                                                             # increment frame count
        wait_time = max(1, int(fps/4))
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break


cap.release()                                           # release VideoCapture()
cv2.destroyAllWindows()                                 # close all frames and video windows
avg_fps = total_fps / frame_count                       # calculate and print the average FPS
print(f"Average FPS: {avg_fps:.3f}")

