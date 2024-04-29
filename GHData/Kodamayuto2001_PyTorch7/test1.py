import cv2

cascadePath = 'haarcascades/haarcascade_frontalface_default.xml'

# Haar-like特徴分類器の読み込み
cascade = cv2.CascadeClassifier(cascadePath)

cap = cv2.VideoCapture(0)

#print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320))
#>True

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)

#print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080))
#>True

cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

while True:
    #動画ファイルの読み込み
    success,img = cap.read()
    #グレースケール変換
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #顔検知(グレースケールで)z
    faces = cascade.detectMultiScale(imgGray)
    for (x,y,w,h) in faces:
        #左上(原点)=(x,y)
        #左上が(0,0)で、右下が(1920,1080)となっている
        #顔を囲む
        #rectangle(img,leftTop,rightBottom,color)
        color = (255,0,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color)
        imgTrim = img[y:y+h,x:x+w]

        cv2.imshow("Results",img)
        #cv2.imshow("ImageTrim",imgTrim)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
