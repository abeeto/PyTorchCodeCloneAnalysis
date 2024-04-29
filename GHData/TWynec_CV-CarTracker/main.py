import cv2
import numpy as np
import torch
import time
from Car import *


if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

    car1 = Car(0, 0)

    """
    Coordinates of the demo gif in lat/lon

    landmarks =  [(52.03677, 6.60201), # lamp top right pt 1
        (52.03660, 6.60362), # entrance bottom of screen pt 3
        (52.036461, 6.60330), # entrance left pt 2
        (52.036788, 6.60341)] #entrance right pt 4
    
        # between pt 1 and 2 is 317 feet
        # between pt 2 and 3 is 76 feet
        # between 3 and 4 is 75 feet
        # beween 4 and 1 is 229 feet
    """

    cap = cv2.VideoCapture("cctv.gif")

    while cap.isOpened():
        ret, img = cap.read()

        height, width, _ = img.shape

        results = model(img)

        # Draw a circle at each landmark
        cv2.circle(img, (520, 133), 1, (0, 255, 0), 2)
        cv2.circle(img, (69, 271), 1, (0, 255, 0), 2)
        cv2.circle(img, (164, 178), 1, (0, 255, 0), 2)

        labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        coords = results.xyxy[0]

        """
        Locations in pixels
        lamp to right = 566, 133
        entrance bott = 69, 271
        entrance far left = 164, 178
        """

        selecty = [item[1] for item in coords]
        selectx = [item[0] for item in coords]
        x = str(selectx)
        y = str(selecty)

        # Display the speed of the first car detected
        car1.locate(x, y)
        if(car1.getSpeed() != 0):
            cv2.putText(img, car1.getSpeed() + " MPH", (164, 178), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)


        cv2.imshow("Img", img)
        #cv2.imshow("Image", np.squeeze(results.render()))
        cv2.waitKey(200) # Should be roughly 5 fps
        

    cap.release()
    cv2.destroyAllWindows()

    