from PIL import Image
from PIL import ImageTk
import cv2,time

def readCam():
    img = None
    cap=cv2.VideoCapture(0)
    for i in xrange(10):
        if cap.isOpened():
            break
        time.sleep(1)
        print i
    for i in xrange(10):
        ret,im=cap.read()
        if ret:
            img=im
            break
        time.sleep(1.8)
    #ret,img=cap.read()
    
    #ret,img=cap.read()
    #time.sleep(5)
    print ret,img
    print 'showing'
    p=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    return p

if __name__=="__main__":
    p=readCam()
    p.show("ABC")
    #cv2.imshow('Image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #time.sleep(10)
    print 'showd'