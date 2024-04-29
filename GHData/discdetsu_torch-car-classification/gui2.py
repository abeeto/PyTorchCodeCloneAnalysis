import tkinter
import cv2
from PIL import Image, ImageTk
import time
from csv import writer

from video import *
from color_detector import get_color_name
from lpr2_api import get_license_plate
from record import create_record_window

class App:
    def __init__(self, window, window_title, coor, video_source=0):
        self.temp = 0
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        
        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        self.resized_res = (int(self.vid.width/3), int(self.vid.height/3))
        self.coor = coor

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(self.window, width=self.resized_res[0], height=self.resized_res[1])
        self.canvas.grid(row=0, column=0, rowspan=6)

        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(self.window, text="Snapshot", width=20, command=self.snapshot)
        self.btn_snapshot.grid(row=8, column=0, pady=10)

        self.car_model_name = tkinter.Label(self.window, font= ("Ubuntu", 20), text='Model name')
        self.car_model_name.grid(row=7, column=0)
       
        self.license_plate_col = tkinter.Label(self.window, font=("Ubuntu", 20), text="License Plate")
        self.license_plate_col.grid(row=0, column=1, padx=10)

        self.car_model_col = tkinter.Label(self.window, font=("Ubuntu", 20), text="Car Model")
        self.car_model_col.grid(row=0, column=2, padx=10)
        
        self.time_col = tkinter.Label(self.window, font=("Ubuntu", 20), text="Time")
        self.time_col.grid(row=0, column=3, padx=10)

        self.predicted_label = [[None for _ in range(3)] for _ in range(5)]

        self.table_queue = []

        for r in range(1, 6):
            for c in range(1, 4):
                self.predicted_label[r-1][c-1] = tkinter.Label(self.window, font=("Ubuntu", 14), text="")
                self.predicted_label[r-1][c-1].grid(row=r, column=c, padx=10)
       
        self.btn_all_recording_time=tkinter.Button(self.window, text="All Recording Time", width=20, command=self.open_record)
        self.btn_all_recording_time.grid(row=8, column=1, pady=10, columnspan=3)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.window.mainloop()

    def snapshot(self):
        
        color_point = (int(((self.coor['p2'][0] - self.coor['p1'][0]) / 2) + self.coor['p1'][0]),
            int(((self.coor['p2'][1] - self.coor['p1'][1]) / 2) + self.coor['p1'][1]) - 100)

        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            current_time = time.strftime("%d-%m-%Y-%H-%M-%S")
            image_path = "screenshot/frame-" + current_time + ".jpg"

             # save image for api requests
            cropped_frame = frame[self.coor['p1'][1]:self.coor['p2'][1], self.coor['p1'][0]:self.coor['p2'][0]]    
            cv2.imwrite(image_path, cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2BGR))
            
            data = {
                "file_name": image_path,
                "lp": get_license_plate(image_path),
                "car": self.predict(model, classes, cropped_frame),
                "color": get_color_name(color_point, frame),
                "time_entry": current_time
            }

            with open('record.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(data.values())
                f_object.close()

            if len(self.table_queue) < 5:
                self.table_queue.append(data)
                
            else:
                self.table_queue.pop(0)
                self.table_queue.append(data)

            # Table labels
            for i in range(len(self.table_queue)):
                self.predicted_label[i][0].config(text=self.table_queue[i]['lp'])
                self.predicted_label[i][1].config(text=self.table_queue[i]['car'] + '|' + self.table_queue[i]['color'])
                self.predicted_label[i][2].config(text=self.table_queue[i]['time_entry'])
          

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            # Coordiante for detecting car color
            # cv2.putText(frame, 'X', color_point, 2, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.rectangle(frame, self.coor['p1'], self.coor['p2'], (255, 0, 0), 5)
            
            image = Image.fromarray(frame).resize(self.resized_res)
            self.photo = ImageTk.PhotoImage(image=image)
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
            
            # Realtime label
            cropped_frame = frame[self.coor['p1'][1]:self.coor['p2'][1], self.coor['p1'][0]:self.coor['p2'][0]] 
            pred_class = self.predict(model, classes, cropped_frame)
            self.car_model_name.config(text=pred_class)



        self.window.after(self.delay, self.update)

    def open_record(self):
        create_record_window(self.window)
        
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

        topk = 1
        predicted_top = output.topk(topk)[1]
        predicted = np.array(predicted_top)[0]
        predicted = [classes[i] for i in predicted]
        
        return predicted[0]

class MyVideoCapture:
    def __init__(self, video_source=0):
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

if __name__ == "__main__":
    
    model = load_checkpoint('model.pth').cpu()
    model.eval()

    data_dir = "car_data"
    test_dir = data_dir + '/test'
    
    classes, c_to_idx = find_classes(data_dir + '/train')

    video_file = 'test_video.mp4'
    bbox_coordinate = {
                        'p1': (979, 69),
                        'p2':(1832, 931)
                    }
  
    App(tkinter.Tk(), "Snap", bbox_coordinate, video_file)


# Todos
# detect on cropped frame