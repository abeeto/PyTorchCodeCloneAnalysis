import cv2

#face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')
cascade_path = "opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path) # detectorを作成

cap = cv2.VideoCapture(0)
cap.set(3,340)
cap.set(4,200)

cnt = 0

while True:
    success,img = cap.read()
    pathImage = "Resources/Images/"+str(cnt)+".jpg"
    pathImgTrim = "Resources/ImgTrim/"+str(cnt)+".jpg"

    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgResults = img.copy()

    #カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(10,10))

    color = (82,188,222)

    

    #検出した場合
    if len(facerect) > 0:
        for rect in facerect:
            cv2.rectangle(imgResults,tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
            x = rect[0]
            y = rect[2]
            xw,yh = tuple(rect[0:2]+rect[2:4])
            imgTrim = img[y:yh, x:xw]

            #保存
            cv2.imwrite(pathImgTrim,imgTrim)
            cv2.imwrite(pathImage,img)

            #表示
            cv2.imshow("Results",imgResults)
            cv2.imshow("ImageTrim",imgTrim)


    cv2.imshow("Original",img)

    if cnt >= 99:
        cnt = 0
    else:
        cnt += 1

    if cv2.waitKey(100) & 0xFF == ord("q"):
        break
    

