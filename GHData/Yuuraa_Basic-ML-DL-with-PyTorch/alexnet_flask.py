import json
import torch
import requests
import torchvision.models as models
import torch.nn.functional as F

from PIL import Image
from io import BytesIO
from flask import Flask
from flask import request
from flask_restful import Api
from flask_restful import Resource
from torchvision import transforms


class Predict(Resource):
    def __init__(self):
        self.model = models.alexnet(pretrained=True)

        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.preprocess = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=imagenet_mean, std=imagenet_std)]
            )

        with open("imagenet_class_index.json", 'r') as f:   
            class_idx = json.load(f)
        self.labels = [class_idx[str(k)][1] for k in range(len(class_idx))]

    def post(self):
        payload = request.get_json()
        url = payload['url']

        img = requests.get(url).content
        img_pil = Image.open(BytesIO(img))
        img_tensor = self.preprocess(img_pil)
        batch = torch.unsqueeze(img_tensor, 0)

        self.model.eval()
        out = self.model(batch)

        _, index = torch.max(out, 1)
        pct = F.softmax(out, dim=1)[0] * 100
        return (self.labels[index[0]], pct[index[0]].item()),


app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.run(host='0.0.0.0')
