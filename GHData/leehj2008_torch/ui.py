import Tkinter
from PIL import ImageTk,Image
from Tkinter import Label,Canvas,Button
from Image import readCam

import time
import cv2
global captured
global cnt
cnt=0
captured=False
def capture():
    global captured
    captured=True

root=Tkinter.Tk()
cap=cv2.VideoCapture(0)

canvas=Canvas(root,width=800,height=600)
canvas.grid(row=0,column=0)
button=Button(root,text="Capture",command=capture)
button.grid(row=1,column=0)

ret=False
while not ret:
    ret,img=cap.read()
while True:    
    ret,img=cap.read()
    img=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    photo=ImageTk.PhotoImage(image=img)
    canvas.create_image(0,0,anchor='nw',image=photo)
    root.update_idletasks()
    root.update()
    global captured
    if captured:
        print 'capture'
        img.save('/Users/lihaijiang/Documents/PYWork/torchai/img/img_%d.jpg'%cnt)
        cnt=cnt+1
        captured=False
