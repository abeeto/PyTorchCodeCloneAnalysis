
from PIL import Image
import os.path
import glob
def convertjpg(jpgfile,outdir,width=416,height=416):
    img=Image.open(jpgfile)
    try:
        new_img=img.resize((width,height),Image.BILINEAR)
        new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    except Exception as e:
        print(e)

for jpgfile in glob.glob("D:/Honsia/Pro-pycharm/PyTorch-YOLOv3/data/custom/images/*.jpg"):

    convertjpg(jpgfile,"D:/Honsia/Pro-pycharm/PyTorch-YOLOv3/data/custom/images")
