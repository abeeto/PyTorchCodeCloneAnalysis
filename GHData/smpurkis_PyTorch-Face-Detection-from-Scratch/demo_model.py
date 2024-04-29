import os

import cv2
import torch

from datasets.utils import convert_bbx_to_xyxy

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def define_model(model_path: str):
    model = torch.jit.load(model_path)
    return model


@torch.no_grad()
def extract_face(frame, model):
    image = cv2.resize(frame, (480, 480))
    tensor = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
    tensor = torch.stack([tensor, tensor], dim=0)
    bbxs = model(tensor, predict=torch.tensor(1))
    for b in bbxs:
        if len(b) == 5:
            b = b[1:]
        if b[2] <= 15 or b[3] <= 15:
            width = 1
        else:
            width = 3
        bbx = [int(p.numpy()) for p in convert_bbx_to_xyxy(b)]
        image = cv2.rectangle(
            image,
            pt1=(bbx[0], bbx[1]),
            pt2=(bbx[2], bbx[3]),
            thickness=width,
            color=(0, 0, 200),
        )
    return image


def run_camera(model):
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        process_frame = extract_face(frame, model)
        cv2.imshow("Input", process_frame)

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    model = define_model(
        "./saved_models/official/PoolResnet/medium_model_10x10_480.pth"
    )
    run_camera(model)
