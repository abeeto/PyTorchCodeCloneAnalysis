#
from argparse import ArgumentParser

#
from main_train import main_train
from main_eval import main_eval
from utils import logger, load_config_file

#
if __name__ == "__main__":

    """
    Training example:
    > dir="results/swin_small_patch4_window7_224"
    > python3 main.py --mode="train" --config="configs/swin_transformer.yaml" --save-dir=$dir
    """

    """
    Evaulating example:
    > dir="results/swin_small_patch4_window7_224"
    > python3 main.py --mode="eval" --save-dir=$dir
    """

    # 1. Use argument parser to deal with terminal inputs
    parser = ArgumentParser(description="torch-Classification")
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument(
        "--mode",
        "-m",
        default="train",
        help="The running mode,  whether to train or to evaluate",
    )
    parser.add_argument(
        "--config",
        "-cf",
        default="configs/swin_transformer.yaml",
        help="The configuartion file containing hyper-parameters and other settings",
    )
    parser.add_argument(
        "--save-dir",
        "-sd",
        default="model_v0",
        help="The output directory to store the results",
    )
    args = parser.parse_args()

    # 2. Parse the config file ({args.config}.yaml) into opts (dict)
    opts = load_config_file(args)
    logger.info(f"Configurations loaded from {args.config}")
    logger.info(f"Configurations:\n{dir(opts)}")

    # 3. Start training/evaluating
    mode = getattr(opts, "mode", None)
    assert mode is not None, "[ERROR] Attribute $mode should not ne None"
    if mode == "train":
        assert args.config is not None, "Need to specify the training config file."
        main_train(opts)
    elif mode == "eval":
        main_eval(opts)
    else:
        raise RuntimeError(f"Unknown running mode: {mode}.")
