import os
from network import ReCoNet
import torch

ALL_MODEL_OBJS = [
        {
            "model_name": "style25_train_20200722213134.pt",
            "original": ""
        },
        {
            "model_name": "style25_train_20200725012422.pt",
            "original": ""
        },
        {
            "model_name": "style25_train_edtaonisl_final_20200725184845.pt",
            "original": ""
        },
        {
            "model_name": "style25_train_udnie_final_20200725154137.pt",
            "original": ""
        }
    ]
def get_all_models(device):
    print(device)
    print("Getting all models...")
    all_models = []
    original_images = []
    for model_obj in ALL_MODEL_OBJS:
        original_images.append(model_obj["original"])
        model_name = model_obj["model_name"]
        model = ReCoNet()
        model.load_state_dict(
            torch.load(os.path.join("trained_models", model_name), map_location="cuda:0")
        )
        model=model.to("cuda:0")
        all_models.append(model)
    return all_models, original_images