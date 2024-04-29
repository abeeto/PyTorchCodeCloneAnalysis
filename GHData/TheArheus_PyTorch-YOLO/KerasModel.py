import json
import base64
from hashlib import sha256
import cv2

gg = cv2.imread("C:\\Users\\1\\Pictures\\mslf.jpg")
gg = cv2.resize(gg, (240, 320))
cv2.imwrite("C:\\Users\\1\\Pictures\\mslf.jpg", gg)

data = {}
with open("C:\\Users\\1\\Pictures\\mslf.jpg", "rb") as f:
    image = f.read()

data["sha256"] = sha256(image).hexdigest()
data["img"] = base64.encodebytes(image).decode("utf-8")

print(json.dumps(data))