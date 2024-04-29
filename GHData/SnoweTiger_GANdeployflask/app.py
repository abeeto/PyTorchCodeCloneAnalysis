from flask import Flask, render_template, request #, url_for, send_file, request

import torch

import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from math import ceil

import gan_model as gan

def tensor_to_byteimg(tensor):
    obj = io.BytesIO()
    plt.imsave(obj, tensor, format='jpeg')
    obj.seek(0)

    data = obj.read()
    data = base64.b64encode(data)
    data = data.decode()
    return data

PATH_G = f'ZAElr21_L400C32_400_G.pth'
STATS = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
generator = gan.Generator()
generator.load_state_dict(torch.load(PATH_G, map_location=torch.device(device)))
generator.eval()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', images=[])


@app.route('/gen', methods=['GET'])
def generate():
    n_img = int(request.args.get("nimg"))
    # print('n_img=',n_img)
    tensor = torch.randn(n_img, 400)

    latent_img = tensor * STATS[1][0] + STATS[0][0]
    latent_img = tensor_to_byteimg(latent_img.numpy())

    with torch.no_grad():
        img_tensor = generator.pred_image(tensor, STATS)
        img_tensor.detach().to('cpu')
    img_tensor = img_tensor.permute(0, 2, 3, 1)

    images = []
    n_img = img_tensor.shape[0]
    n_col = int(n_img ** 0.5)
    n_row = ceil(n_img/n_col)

    for k in range(n_img):
      data = tensor_to_byteimg(img_tensor[k].numpy())
      images.append(data)

    # print(n_img, n_col, n_row)

    return render_template('index.html', images=images,
                            n_row=n_row, n_col=n_col, n_img=n_img,
                            latent=latent_img)


if __name__ == '__main__':
    app.run()

# trash

# def generate_image():
#     plt.cla()   # to clear it because it may keep it between sessions
#     y = [random.randint(-10, 10) for _ in range(10)]
#     x = list(range(10))
#     plt.plot(x, y)
#     obj = io.BytesIO()              # file in memory to save image without using disk
#     plt.savefig(obj, format='png')  # save in file (BytesIO)
#     obj.seek(0)                     # move to beginning of file (BytesIO) to read it
#     return obj

# @app.route('/send')
# def example2():
#     img1 = generate_image()
#     data1 = img1.read()              # get data from file (BytesIO)
#     data1 = base64.b64encode(data1)  # convert to base64 as bytes
#     data1 = data1.decode()           # convert bytes to string
#     img1 = '<img src="data:image/png;base64,{}">'.format(data1)
#     html = img1
#     return html

# @app.route('/image', methods=['GET', 'POST'])
# def image():
#
#     X = torch.rand(3, 128, 128)
#     send = [X.size(), X.min(), X.max(), X.mean()]
#     buffer = 'temp2.jpg'
#     save_image(X, f'static\{buffer}')
#     img = generate_image()
#     return render_template('index.html', text=send, pic=img)
