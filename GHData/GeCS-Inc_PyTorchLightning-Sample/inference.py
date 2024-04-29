import model
import train
import os
import numpy as np
from PIL import Image
from glob import glob
from pytorch_lightning import seed_everything

env = train.env
transform = train.transform

IMAGE_DIR = "./dataset/dog/"
PRETRAINED = "lightning_logs/version_10/checkpoints/epoch=1.ckpt"

if __name__ == "__main__":
    seed_everything(42)

    model = model.FineTuningModel.load_from_checkpoint(PRETRAINED)
    model.eval()
    img_paths = glob(os.path.join(IMAGE_DIR, "*.png")) + \
        glob(os.path.join(IMAGE_DIR, "*.jpg")) + \
        glob(os.path.join(IMAGE_DIR, "*.jpeg"))

    for path in img_paths:
        img = Image.open(path)
        img = transform(img)
        img = img.unsqueeze(0)

        output = model(img).squeeze(0).detach().numpy()
        idx = np.argmax(output)
        print(f"{os.path.basename(path)}: {idx} [{output[idx]}]")
