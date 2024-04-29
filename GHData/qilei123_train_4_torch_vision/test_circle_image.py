
import os
from PIL import Image

#圆形头像
def circle(img_path):
    path_name = os.path.dirname(img_path)
    cir_file_name = 'cir_img.png'
    cir_path = path_name + '/' + cir_file_name
    ima = Image.open(img_path).convert("RGBA")
    size = ima.size
    print(size)
    # 因为是要圆形，所以需要正方形的图片
    r2 = min(size[0], size[1])
    if size[0] != size[1]:
        ima = ima.resize((r2, r2), Image.ANTIALIAS)
    # 最后生成圆的半径
    r3 = int(r2/2)
    imb = Image.new('RGBA', (r3*2, r3*2),(255,255,255,0))
    pima = ima.load() # 像素的访问对象
    pimb = imb.load()
    r = float(r2/2) #圆心横坐标
 
    for i in range(r2):
        for j in range(r2):
            lx = abs(i-r) #到圆心距离的横坐标
            ly = abs(j-r)#到圆心距离的纵坐标
            l = (pow(lx,2) + pow(ly,2))** 0.5 # 三角函数 半径
            if l < r3:
                pimb[i-(r-r3),j-(r-r3)] = pima[i,j]
 
    imb.save(cir_path)

circle("/media/cql/DATA0/DOWNLOAD/1163156166.jpg")
