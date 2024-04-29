import torch
import time

import glob
import os
import numpy as np
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import MAX_WAV_VALUE
from models import Generator


def benchmark_fn(fn, ip, count=100):
    # Warm start before actual benchmark, TorchScript dynamically makes JIT optimizations.
    [fn(*ip) for x in range(10)]

    start_time = time.time()
    for i in range(count):
        fn(*ip)
    end_time = time.time()

    return (end_time - start_time) / count


def get_model(path, device):
    config_file = os.path.join(os.path.split(path)[0], 'config.json')
    with open(config_file) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    def load_checkpoint(filepath, device):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(path, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    return generator


if __name__ == '__main__':
    # Change parameters below according to your setting.
    MODEL_PATH = "models/UNIVERSAL_V1/g_02500000"
    DEVICE = torch.device("cuda")
    SCRIPT_MODEL_SAVE_PATH = "hifigan_torchscript.pt"

    model = get_model(MODEL_PATH, DEVICE)

    script_model = torch.jit.script(model)

    # This is done to ensure save/load functionality is working correctly.
    torch.jit.save(script_model, SCRIPT_MODEL_SAVE_PATH)
    script_model = torch.jit.load(SCRIPT_MODEL_SAVE_PATH)

    random_ip = torch.rand([1, 80, 128], device=DEVICE)

    raw_model_average_secs_per_inference = benchmark_fn(model, [random_ip])
    script_model_average_secs_per_inference = benchmark_fn(script_model, [random_ip])

    print("Unscripted Model Average Inference Time:", raw_model_average_secs_per_inference)
    print("Scripted Model Average Inference Time:", script_model_average_secs_per_inference)

    print("TorchScript Performance Ratio: {:.2f}".format(raw_model_average_secs_per_inference/script_model_average_secs_per_inference))

