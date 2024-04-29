from argparse import ArgumentParser

parser = ArgumentParser()

def MyArgumentParser():
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
        "--batch_size",
        "-b",
        type=int,
        default=4,
        help="The batch size. Default: 10")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs. Default: 300")
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=5e-4,
        help="The learning rate. Default: 5e-4")
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.1,
        help="The learning rate decay factor. Default: 0.5")
    parser.add_argument(
        "--lr_decay_epochs",
        type=int,
        default=100,
        help="The number of epochs before adjusting the learning rate. "
        "Default: 100")
    parser.add_argument(
        "--weight_decay",
        "-wd",
        type=float,
        default=2e-4,
        help="L2 regularization factor. Default: 2e-4")

    # Dataset
    parser.add_argument(
        "--dataset",
        choices=['camvid', 'cityscapes', "aeroscapes"],
        default='camvid',
        help="Dataset to use. Default: camvid")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=r"E:\anqc\datasets\CamVid480x360\\",
        help="Path to the root directory of the selected dataset. "
        "Default: data/CamVid")
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="The image height. Default: 360")
    parser.add_argument(
        "--width",
        type=int,
        default=480,
        help="The image width. Default: 480")
    parser.add_argument(
        "--weights",
        default=[ 6.100,  4.313, 33.974,  3.670,  12.006,  8.074,
                32.983, 29.677, 15.549, 38.390, 39.944,  18.395],
        help="The class weighing technique to apply to the dataset.default is Paszke ")
    parser.add_argument(
        "--with_unlabeled",
        dest='ignore_unlabeled',
        default=None,
        action='store_false',
        help="The unlabeled class is not ignored.")

    parser.add_argument(
        "--num_classes",
        default=12,
        action='store_false',
        help="The unlabeled class is not ignored.")

    # Settings
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. Default: 4")
    parser.add_argument(
        "--print_step",
        action='store_true',
        default=40,
        help="Print loss every step")
    parser.add_argument(
        "--imshow_batch",
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
        default='ENet',
        help="Name given to the model when saving. Default: ENet")
    parser.add_argument(
        "--save_dir",
        type=str,
        default='./model',
        help="The directory where models are saved. Default: save")
    parser.add_argument(
        '--mean',
        type=list,
        default=[ 95.83291771,  99.40274314, 101.67830424],
        help="the mean value of dataset, default:camvid"
    )
    parser.add_argument(
        '--std',
        type=list,
        default= [66.55860349, 70.37032419, 70.06857855],
        help="the stf value of dataset, default:camvid"
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default= './images/',
        help="the stf value of dataset, default:camvid"
    )

    return parser.parse_args()
