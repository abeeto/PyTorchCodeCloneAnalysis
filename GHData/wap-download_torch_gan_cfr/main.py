import json
import os
from generator.data_generator import run as generator_run
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == "__main__":
    with open('configs/run.json', 'r') as f:
        run_dict = json.load(f)
    generator_run(run_dict["generator"])
