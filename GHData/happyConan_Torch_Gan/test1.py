# coding=gbk
from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt

def ImageToMatrix(filename):
    # ¶ÁÈ¡Í¼Æ¬
    im = Image.open(filename)
    # ÏÔÊ¾Í¼Æ¬
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
#     new_im = Image.fromarray(new_data)
#     # ÏÔÊ¾Í¼Æ¬
#     new_im.show()
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im



filename = 'test.jpg'
data = ImageToMatrix(filename)
print(data)
new_im = MatrixToImage(data)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
plt.show()
new_im.show()
new_im.save('lena_1.bmp')