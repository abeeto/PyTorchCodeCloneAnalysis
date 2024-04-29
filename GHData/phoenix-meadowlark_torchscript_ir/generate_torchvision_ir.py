import argparse
import os

import utils
import torch

VISION_GITHUB = "pytorch/vision"
ALL_MODELS = torch.hub.list(VISION_GITHUB)


def parse_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--models",
      default="all",
      help=f"Either 'all' or a comma delimited list of {ALL_MODELS}.")
  parser.add_argument("--save_models",
                      action="store_true",
                      help="Whether or not to save ")
  return parser.parse_args()


def main():
  args = parse_arguments()
  models = ALL_MODELS if args.models == "all" else args.models.split(",")

  for i, model_name in enumerate(models):
    print(f"Compiling {i + 1:2d}/{len(models)}: {model_name}")

    model = torch.hub.load(VISION_GITHUB, model_name, verbose=False)
    jit_model = torch.jit.script(model)
    model_path = os.path.join("tsir", "torchvision", model_name)

    utils.save_tsir(model, jit_model, model_path, args.save_models)


if __name__ == "__main__":
  main()
