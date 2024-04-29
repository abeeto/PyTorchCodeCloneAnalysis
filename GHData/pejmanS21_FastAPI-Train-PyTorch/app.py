from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from commands import HyperParam
from train_utils import fit
from model_utils import CNN
from utils import get_device, save_results
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch
import os
import io
from starlette.responses import StreamingResponse
import cv2
import base64


app = FastAPI()
templates = Jinja2Templates(directory="templates/")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


if not os.path.isdir("static"):
    os.mkdir("static")

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_form():
    return "hello world"


@app.get("/form")
def form_post(request: Request):
    return templates.TemplateResponse("form.html", context={"request": request})


@app.post("/form")
def form_post(
    request: Request,
    model_name: str = Form(...),
    dataset_name: str = Form(...),
    Optimizer: str = Form(...),
    learning_rate: float = Form(...),
    batch_size: int = Form(...),
    epoch: int = Form(...),
    initial_hidden_size: int = Form(...),
):
    out_ch = 1
    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        out_ch = 3
    # transform to normalize the data
    hp = HyperParam()
    compose = []
    for key, value in eval(f"hp.transform_{out_ch}D").items():
        if type(value) == bool:
            if value is True:
                compose.append(eval("transforms.{}()".format(key)))
        elif key == "Normalize":
            compose.append(
                eval(
                    f"transforms.{key}(hp.transform_{out_ch}D['{key}']['mean'], hp.transform_{out_ch}D['{key}']['std'])"
                )
            )
        else:
            compose.append(eval(f"transforms.{key}(hp.transform_{out_ch}D['{key}'])"))

    transform = transforms.Compose(compose)
    # Download and load the training data
    dataset = eval(
        f"datasets.{dataset_name}('./data', download=True, train=True, transform=transform)"
    )
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # Download and load the test data
    val_dataset = eval(
        f"datasets.{dataset_name}('./data', download=True, train=False, transform=transform)"
    )
    val_dl = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    model = CNN(
        in_channels=dataset[0][0].shape[0],
        hidden_size=initial_hidden_size,
        out_classes=len(dataset.classes),
    ).to(get_device())
    criterion = eval("torch.nn.{}()".format(hp.criterion["name"]))
    optimizer = eval(
        "torch.optim.{}(model.parameters(), lr={})".format(Optimizer, learning_rate)
    )
    res = fit(
        epoch,
        model,
        train_dl,
        dataset,
        val_dl,
        val_dataset,
        loss_func=criterion,
        device=get_device(),
        optimizer=optimizer,
    )

    torch.save(model.state_dict(), f"./data/{model_name}.pth")
    save_results(res)  # save results as png in data dir

    return RedirectResponse(url="/results")


@app.post("/results")
def results(request: Request):
    return templates.TemplateResponse("result.html", context={"request": request})
