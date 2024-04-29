import cv2
from Model.AdaIN import AdaIN
import  torchvision.transforms as transforms
from PIL import Image
import torch 


def check_cams():
    cams_test = 10
    for i in range(0, cams_test):
        cap = cv2.VideoCapture(i)
        test, frame = cap.read()
        print("i : "+str(i)+" /// result: "+str(test))
    
    
#check_cams()

cam = cv2.VideoCapture(0)

cv2.namedWindow("raw")
cv2.namedWindow("adain")
cv2.namedWindow("style")

img_counter = 0

frame_transform = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.ToTensor()])
            
adain = AdaIN()
adain.decoder.eval()
adain.encoder.eval()
adain.load_save()
style_name = "2.jpg"
style_image = Image.open(style_name).convert('RGB')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
style_image = frame_transform(style_image).unsqueeze(0).detach().to(device)

######### TRACKBAR
alpha = 1

def on_trackbar(val):
    global alpha 
    alpha = val / alpha_slider_max
    print(alpha)
    
alpha_slider_max = 100
trackbar_name = 'Alpha x %d' % alpha_slider_max
cv2.createTrackbar(trackbar_name, "adain" , 0, alpha_slider_max, on_trackbar)
############ 

style_frame = style_image.squeeze(0).permute(1,2,0).cpu().detach().numpy()
cv2.imshow("style", style_frame)

while True:
    ret, frame = cam.read()
    tframe = frame_transform(Image.fromarray(frame)).unsqueeze(0).detach()

    adain_frame = adain.forward(style_image=style_image, content_image=tframe.to(device), alpha=alpha)[0]
    adain_frame = adain_frame.squeeze(0).permute(1,2,0).cpu().detach().numpy()

    cv2.imshow("raw", frame)
    cv2.imshow("adain", adain_frame)

    
    if not ret:
        break
        
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC Pressed
        print("Escape hit, closing..")
        break
    elif k%256 == 32:
        # SPACE Pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

