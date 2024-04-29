from fastapi import FastAPI
import uvicorn
from commands import HyperParam
from train_utils import fit
from model_utils import CNN
from utils import get_device, save_results
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import io
from starlette.responses import StreamingResponse
import cv2
import base64


app = FastAPI()


@app.get('/')
def main():
    return {"status": "OK"}


@app.post('/train')
def train(hp: HyperParam):
    # transform to normalize the data
    compose = []
    for key, value in hp.transform.items():
        if type(value) == bool:
            if value is True:
                compose.append(eval("transforms.{}()".format(key)))
        elif key == "Normalize":
            compose.append(eval(f"transforms.{key}(hp.transform['{key}']['mean'], hp.transform['{key}']['std'])"))
        else:
            compose.append(eval(f"transforms.{key}(hp.transform['{key}'])"))

    transform = transforms.Compose(compose)
    # Download and load the training data
    dataset = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
    train_dl = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
    # Download and load the test data
    val_dataset = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)
    val_dl = DataLoader(val_dataset, batch_size=hp.batch_size, shuffle=True)

    model = CNN(in_channels=1, hidden_size=hp.initial_hidden_size, out_classes=len(hp.label_names)).to(get_device())
    criterion = eval("torch.nn.{}()".format(hp.criterion['name']))
    optimizer = eval("torch.optim.{}(model.parameters(), lr=hp.optimizer['lr'])".format(hp.optimizer['name']))
    res = fit(hp.n_epoch, model, train_dl, dataset, val_dl, val_dataset,
              loss_func=criterion, device=get_device(), optimizer=optimizer)

    torch.save(model.state_dict(), f'./data/{hp.model_name}.pth')
    save_results(res)       # save results as png in data dir
    

    # with open("./data/results.png", "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read())

    # return {"History": res, "Image": encoded_string}
    img = cv2.imread("./data/results.png")
    res, im_png = cv2.imencode(".png", img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0")
