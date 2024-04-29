import io
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from skimage.util import img_as_ubyte

from model import load_predictor

app = FastAPI()
version = "0.0.1"
model = None

MODEL_STORE = "model_store"


@app.on_event("startup")
async def startup_event():
    if not Path(MODEL_STORE).is_dir():
        os.makedirs(MODEL_STORE)
    global model
    model = load_predictor("model_store/wrist-detection/1.0/detection-model-created-on-2020-07-24.pth")
    logger.info(f"model loaded")


@app.get("/")
async def root():
    return {"message": f"Running version: {version}"}


def read_image(npz):
    buf = io.BytesIO(npz)
    npzfile = np.load(buf)
    image = npzfile["arr_0"]
    return image


@app.post("/predictions")
async def predictions(file: UploadFile = File(...)):
    logger.debug(f"Running prediction for filename: {file.filename}")
    image = read_image(await file.read())
    outputs = model(img_as_ubyte(image))
    out = outputs["instances"].to("cpu")
    bboxes = out.pred_boxes.tensor.numpy()
    classes = out.pred_classes.numpy()
    return {"bboxes": bboxes.tolist(), "classes": classes.tolist()}
