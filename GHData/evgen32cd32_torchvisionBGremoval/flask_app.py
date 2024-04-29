import os
from flask import Flask, request
import requests
import telepot
import urllib3
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import gc

num_classes = 2
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
model.load_state_dict(torch.load('./maskrcnn_resnet50_fpn_20',map_location=torch.device('cpu')))
model = model.eval()

token = '******'

secret = '*******'
bot = telepot.Bot(token)
bot.setWebhook("https://bgremovaltelegram.herokuapp.com/{}".format(secret), max_connections=1)

app = Flask(__name__)


@app.route('/{}'.format(secret), methods=["POST"])
def telegram_webhook():
    return "OK"

@app.after_request
def after_request(response):
    update = request.get_json()
    @response.call_on_close
    def bgremoval():
        if "message" in update:
            chat_id = update["message"]["chat"]["id"]
            if "photo" in update["message"]:
                file_path = bot.getFile(update["message"]["photo"][-1]['file_id'])['file_path']
                content = requests.get('https://api.telegram.org/file/bot' + token + '/' + file_path).content
                image = Image.open(BytesIO(content))
                img = transforms.ToTensor()(image)
                image.close()
                bot.sendMessage(chat_id, 'start computing')
                with torch.no_grad():
                    output = model(torch.unsqueeze(img,dim=0))
                bot.sendMessage(chat_id, 'end computing')
                scores = output[0]['scores']
                if len(scores) == 0:
                    bot.sendMessage(chat_id, 'nothing detected(')
                    return
                mask = (output[0]['masks'][output[0]['scores'].argmax()] >= 0.5).float()[0]
                oimg = transforms.ToPILImage()(img[:] * mask)
                b = BytesIO()
                oimg.save(b,'JPEG')
                oimg.close()
                b.seek(0)
                bot.sendPhoto(chat_id,b)
                gc.collect()
    return response
    
if __name__ == '__main__':
    app.run(debug=False,port=os.getenv('PORT',5000))

