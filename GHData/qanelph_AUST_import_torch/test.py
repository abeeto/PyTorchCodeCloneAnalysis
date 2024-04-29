from PIL import Image
import os
for fname in os.listdir("./out0/"):
    img = Image.open("./out0/{}".format(fname))
    area = (800, 334, 900, 409)
    cropped_img = img.crop(area)
    # print(cropped_img)
    # cropped_img.show()

    basewidth = 640

    wpercent = (basewidth/float(cropped_img.size[0]))
    hsize = int((float(cropped_img.size[1])*float(wpercent)))
    cropped_img = cropped_img.resize((basewidth, hsize), Image.ANTIALIAS)
    cropped_img.save("./croped_image0/{}".format(fname))