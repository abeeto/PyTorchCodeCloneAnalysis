import cv2


if __name__ == '__main__':
    # Create a VideoCapture object and read from input file
    # If the input is the camera, pass 0 instead of the video file name
    cap = cv2.VideoCapture('./data/test.mp4')

    if (cap.isOpened() is False):
        print("Error opening video stream or file")
    count = 0

    # Read until video is completed
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret is True:

            # Display the resulting frame
            # cv2.imshow('Frame',frame)
            if(count % 1000 == 0):
                print("Processing Image {}".format(count))
            cv2.imwrite("./data/images_test/frame%d.jpg" % count, frame)

        # Break the loop
        else:
            break
        count += 1

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
