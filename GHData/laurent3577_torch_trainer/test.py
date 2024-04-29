import argparse
from torch_trainer.models import load_from_path
from train import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Test Classification Model")

    parser.add_argument(
        "--save-path", help="Save path to model", required=True, type=str
    )
    parser.add_argument(
        "--dataset-path", required=None, type=str)
    args = parser.parse_args()

    return args


def main():
    print("Testing...")
    args = parse_args()
    model, config = load_from_path(args.save_path, return_config=True)
    if args.dataset_path is not None:
        config.defrost()
        config.DATASET.ROOT = args.dataset_path
        config.freeze()
    evaluate(model, config)


if __name__ == "__main__":
    main()
