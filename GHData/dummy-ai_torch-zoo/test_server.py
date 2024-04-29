import os
import moxel
from moxel.space import Image

model = moxel.Model('jimfan/squeezenet:latest', where='localhost')
for img in os.listdir('demo_imgs'):
    img = os.path.join('demo_imgs', img)
    Image.from_file(img)
    output = model.predict(img=Image.from_file(img))
    print('query "{}" -> {}'.format( img, output ))

