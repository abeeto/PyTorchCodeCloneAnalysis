import cv

def GrabAndShow(windowName, captureDevice):
	frame = cv.QueryFrame(captureDevice)
	if frame is None:
		break
	cv.Flip(frame, None, 1) # Mirror the display.
	frame = DetectFaces(frame)
	cv.ShowImage(windowName,frame)

def CreateWindow(windowName):
	cv.NamedWindow(windowName, flags=cv.CV_WINDOW_AUTOSIZE)
	cv.MoveWindow(windowName,0,0)

def CloseWindow(windowName):
	cv.DestroyWindow(windowName)

def InitCapture():
	numCameras = cv.cvcamGetCamerasCount()
	if numCameras < 1:
		print "No cameras found."
		return None
	else if cv.cvcamGetCamerasCount() > 1:
		print "Multiple cameras found, choosing latest."
		return cv.CaptureFromCAM(numCameras-1)			
	return cv.CaptureFromCAM(-1) # Else choose the only one.

def ReleaseCapture(captureDevice):
	del captureDevice

def DetectFaces(image):
	imageSize = cv.GetSize(image)
	
	# Convert to grayscale
	grayscale = cv.CreateImage(imageSize, 8, 1)
	cv.CvtColor(image, grayscale, cv.CV_BGR2GRAY)
	
	# Equalise histogram
	cv.EqualizeHist(grayscale, grayscale)
	
	# Detect objects
	cascade = cv.Load('/opt/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
	faces = cv.HaarDetectObjects(grayscale,cascade,cv.CreateMemStorage(0), 1.2, 2, cv.CV_HAAR_DO_CANNY_PRUNING, (50,50))

	for ((x,y,w,h),n) in faces:
		cv.Rectangle(	image, 
						( int(x), int(y) ),
						( int(x + w), int(y + h) ),
						cv.CV_RGB(0, 255, 0), 3, 8, 0
						)
	return image
	
if __name__ == "__main__":
	print "Press Esc to exit..."
	windowName = "Test"
	CreateWindow(windowName)
	cvCap = InitCapture()
	while(1):
		GrabAndShow(windowName, cvCap)
		key = cv.WaitKey(10)
		if key == 27: #Esc to quit.
			break
	CloseWindow(windowName)
	ReleaseCapture(cvCap)
