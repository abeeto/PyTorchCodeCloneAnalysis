from flask import Flask, jsonify, request, send_file # initialize our Flask application
import json
from flask_cors import CORS
import base64
from torchvision import transforms
import dnnlib
import torch
import PIL.Image
from io import BytesIO
from datetime import datetime
import legacy

device = torch.device('cuda')
with dnnlib.util.open_url("snapshot.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

app = Flask(__name__)
CORS(app)

@app.route("/test", methods=['GET', 'POST', 'PUT'])
def test():
    return "OK"

@app.route("/query", methods=['GET', 'POST', 'PUT'])
def getjsondata():

    if request.method=='POST':
        # print("received POST")

        data = request.get_json()

        #print(format(data['z']))
        jzf = [float(i) for i in data['z']]
        jzft = torch.FloatTensor(jzf)
        jzftr = jzft.reshape([1, 512])

        z = jzftr.cuda()
        c = None                   # class labels (not used in this example)
        trunc = data['truncation']
        img = G(z, c, trunc)

        #img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)

        # turn into PIL image
        pil_img = transforms.ToPILImage()(img[0]).convert("RGB")
        #pil_img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')

        # SAVING...
        #fn = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        #pil_img.save('saved_images/' + fn + '.jpg')

        response = serve_pil_image64(pil_img)
        response.headers.add('Access-Control-Allow-Origin', '*')
        # response.headers.add('Content-Transfer-Encoding', 'base64')
        return response


    return 'OK'

def serve_pil_image64(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_str = base64.b64encode(img_io.getvalue()).decode("utf-8")
    return jsonify({'status': True, 'image': img_str})


if __name__ == '__main__':
    app.run(host='localhost', port=9000, debug=True)
