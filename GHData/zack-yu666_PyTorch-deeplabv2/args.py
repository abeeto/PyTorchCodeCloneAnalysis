from argparse import ArgumentParser


def get_arguments():
    """Defines command-line arguments, and parses them.

    """
    parser = ArgumentParser()

    # Execution mode
    parser.add_argument(
        "--mode",
        "-m",
        choices=['train', 'test', 'full'],
        default='train',
        help=("train: performs training and validation; test: tests the model "
              "found in \"--save_dir\" with name \"--name\" on \"--dataset\"; "
              "full: combines train and test modes. Default: train"))
    parser.add_argument(
        "--resume",
        action='store_true',
        help=("The model found in \"--checkpoint_dir/--name/\" and filename "
              "\"--name.h5\" is loaded."))

    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=2.5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.5")
    parser.add_argument(
         "--power", 
         type=float,
         default= 0.9,
         help="Decay parameter to compute the learning rate.")
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")
    parser.add_argument(
            "--momentum", 
            type=float, 
            default=0.9,
            help="Momentum component of the optimiser.")
    parser.add_argument(
            "--learning_rate_D", 
            type=float, 
            default=1e-4,
            help="Base learning rate for discriminator.")
    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['cityscapes'],
        default='cityscapes',
        help="Dataset to use. Default: camvid")
    parser.add_argument(
            "--save_pred_every",
            type=int, 
            default=2000,
            help="How many images to save.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/home/huachen_yu/cityscapes",
        help="Path to the root directory of the selected dataset.")
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="The image width. Default: 480")
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100000,
        help="The max iterations for train. Default: 25000")
    parser.add_argument(
        "--weighing",
        choices=['deeplabv2', 'none'],
        default='deeplabv2',
        help="The class weighing technique to apply to the dataset. ")
    parser.add_argument(
        "--with-unlabeled",
        dest='ignore_unlabeled',
        action='store_false',
        help="The unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print-step",
        action='store_true',
        help="Print loss every step")
    parser.add_argument(
        "--imshow-batch",
        action='store_true',
        help=("Displays batch images when loading the dataset and making "
              "predictions."))
    parser.add_argument(
        "--device",
        default='cuda',
        help="Device on which the network will be trained. Default: cuda")

    # Storage settings
    
    parser.add_argument(
        "--name",
        type=str,
        default='deeplabv2',
        help="Name given to the model when saving. Default: ENet")

    parser.add_argument(
        "--save-dir",
        type=str,
        default='save',
        help="The directory where models are saved. Default: save")
    
    # test settings
    parser.add_argument(
        "--test_dir",
        type=str,
        default="test-img",
        help="Path to the directory of the test dataset.")
    parser.add_argument(
        "--test_out",
        type=str,
        default="outputs",
        help="Path to the directory of the testing outputs labels maps.")

    return parser.parse_args()
