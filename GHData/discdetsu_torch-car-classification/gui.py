from logging import root
import tkinter
from turtle import width
import cv2
import PIL.Image, PIL.ImageTk
import time
from tkinter import *
from tkinter.ttk import *
from pathlib import Path
import tkinter
from PIL import ImageTk, Image
import os
from tkinter import ttk
from tkinter import filedialog

from video import *

class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.window.geometry("1200x900")
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack(padx=5, pady=10)

        self.label = Label(self.window, font= ("Ubuntu", 20), text='Model name')
        self.label.pack(pady=10)

        # Create text label to display predicted classes and confident scores
        self.labels = [Label(self.window, font= ("Ubuntu", 20), text='') for i in range(5)]
        for label in self.labels:
            label.pack(padx=5, pady=10, side=tkinter.RIGHT)

        # Button that lets the user take a snapshot
        
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame),)
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            pred_class, cf = self.predict(model, classes, frame)
           
            
            for index, pred in enumerate(zip(pred_class, cf)):
                
                text = '{} Conf: {:.4f}'.format(pred[0], pred[1])
                self.labels[index].config(text=text)


        self.window.after(self.delay, self.update)
    
    def predict(self, model, classes, frame):
        
        pil_image = Image.fromarray(frame)
        transform = transforms.Compose([transforms.Resize((244,244)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])])
        pil_tfd = transform(pil_image)
        array_im_tfd = np.array(pil_tfd)
        img_tensor = torch.from_numpy(array_im_tfd).type(torch.FloatTensor)
        img_add_dim = img_tensor.unsqueeze_(0)
            
        with torch.no_grad():
            output = model.forward(img_add_dim)

        topk = 5
        predicted_top = output.topk(topk)[1]
        predicted = np.array(predicted_top)[0]
        predicted = [classes[i] for i in predicted]

        probs_top = output.topk(topk)[0]
        cfscs = probs_top/probs_top.sum()
        cfscs = list(cfscs.detach().cpu().numpy().flat)
        
        return predicted, cfscs 
 
class MyVideoCapture:
     def __init__(self, video_source):
         # Open the video source
         
         self.vid = cv2.VideoCapture(video_source)
         
         if not self.vid.isOpened():
             raise ValueError("Unable to open video source", video_source)
 
         # Get video source width and height
         self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
         self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
         
 
     def get_frame(self):
         if self.vid.isOpened():
             ret, frame = self.vid.read()
             if ret:
                 # Return a boolean success flag and the current frame converted to BGR
                 return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
             else:
                 return (ret, None)
         else:
             return (ret, None)
 
     # Release the video source when the object is destroyed
     def __del__(self):
         if self.vid.isOpened():
             self.vid.release()

 
 # Create a window and pass it to the Application object


if __name__ == "__main__":
    
    model = load_checkpoint('model.pth').cpu()
    model.eval()

    data_dir = "car_data"
    test_dir = data_dir + '/test'
    
    classes, c_to_idx = find_classes(data_dir + '/train')

    video_file = 'NISSAN 240SX.mp4'
    
    App(tkinter.Tk(), "Tkinter and OpenCV", video_file)
