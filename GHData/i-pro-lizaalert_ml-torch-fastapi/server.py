import uvicorn
import cv2
import torch
import torchvision
from PIL import Image
import numpy as np

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File

app = FastAPI(title='Демо нейронной модели')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


class MultilabelClassifier(torch.nn.Module):
    def __init__(
        self,
        n_avia=2,
        n_auto=2,
        n_bpla=2,
        n_diver=2,
        n_cynologist=2,
        n_horses=2,
        n_hugs=2,
        n_sherp=2,
        n_timeOfTheDay=4,
        n_timeOfTheYear=5,
        n_place=3,
    ):
        super().__init__()
        self.resnet = torchvision.models.resnet34(pretrained=True)
        self.model_wo_fc = torch.nn.Sequential(
            *(list(self.resnet.children())[:-1]))

        self.avia = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_avia),
            # torch.nn.Softmax()
        )
        self.auto = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_auto),
            # torch.nn.Softmax()
        )

        self.bpla = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_bpla),
            # torch.nn.Softmax()
        )
        self.diver = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_diver),
            # torch.nn.Softmax()
        )
        self.cynologist = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_cynologist),
            # torch.nn.Softmax()
        )
        self.horses = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_horses),
            # torch.nn.Softmax()
        )
        self.hugs = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_hugs),
            # torch.nn.Softmax()
        )
        self.sherp = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_sherp),
            # torch.nn.Softmax()
        )
        self.timeOfTheDay = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_timeOfTheDay),
            # torch.nn.Softmax()
        )
        self.timeOfTheYear = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_timeOfTheYear),
            # torch.nn.Softmax()
        )
        self.place = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2),
            torch.nn.Linear(in_features=512, out_features=n_place),
            # torch.nn.Softmax()
        )

    def forward(self, x):
        x = self.model_wo_fc(x)
        x = torch.flatten(x, 1)

        return {
            'avia': self.avia(x),
            'auto': self.auto(x),
            'bpla': self.bpla(x),
            'diver': self.diver(x),
            'cynologist': self.cynologist(x),
            'horses': self.horses(x),
            'hugs': self.hugs(x),
            'sherp': self.sherp(x),
            'timeday': self.timeOfTheDay(x),
            'timeyear': self.timeOfTheYear(x),
            'place': self.place(x),
        }


def predToRes(pred):
    res = {}
    for k, v in pred.items():
        res[k] = v.argmax().item()
    return res


model = torch.load(
    'model_last_gpu__lastlast.pth',
    map_location=torch.device('cpu')
)



def decodeImage(pilBytes):
    imBytes = pilBytes
    imArr = np.frombuffer(imBytes, dtype=np.uint8)
    img = Image.fromarray(cv2.imdecode(imArr, flags=cv2.IMREAD_COLOR))
    return img


@app.post("/prediction",description='Присваивает тэги к картинке')
async def predict(
    File: UploadFile = File(..., description='Файл для теггирования'),
):
    # They come already 256 x 256, only need is to turn to Tensor.
    image = decodeImage(File.file.read())
    photo = torchvision.transforms.ToTensor()(image)
    # if torch.cuda.is_available():
    # photo.cuda()
    photo = photo.unsqueeze(0)
    pred = model(photo)
    return predToRes(pred)


# ngrok_tunnel = ngrok.connect(8001)
# print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
uvicorn.run(app, port=8001)
