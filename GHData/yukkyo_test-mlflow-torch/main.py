import argparse
import torch
import torch.optim as optim
import mlflow
import mlflow.pytorch
from cnn_finetune import make_model

from dataset import get_dataloaders
from utils import train


"""
Reference:
    https://github.com/mlflow/mlflow/blob/master/examples/pytorch/mnist_tensorboard_artifact.py
"""


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--num-workers', default=8, type=int, help='batch size')
    parser.add_argument('--multi-gpu', action='store_true', help='use multi gpu')
    parser.add_argument('--init-lr', default=1e-3, type=float, help='init lr')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--epochs', default=50, type=int, help='epochs')
    parser.add_argument('--use-pretrain', action='store_true', help='use pretrain')
    parser.add_argument('--model', default='se_resnext50_32x4d', type=str, help='batch size')
    return parser


def main():
    args = make_parse().parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_dataloaders(batch_size=args.batch_size,
                                                num_workers=args.num_workers)

    # num_classes and input_size is params of CIFAR-10
    print(f'model: {args.model}')
    model = make_model(args.model, num_classes=10,
                       pretrained=args.use_pretrain, input_size=(32, 32))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr, momentum=args.momentum)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[int(args.epochs*0.8), int(args.epochs*0.9)],
        gamma=0.1
    )

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    results = train(
        epochs=args.epochs,
        model=model,
        train_loader=train_loader,
        valid_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )  # train_loss, val_loss, val_acc

    # mlflow
    with mlflow.start_run() as run:
        # Log args into mlflow
        for key, value in vars(args).items():
            mlflow.log_param(key, value)

        # Log results into mlflow
        for key, value in results.items():
            mlflow.log_metric(key, value)

        # Log other info
        mlflow.log_param('loss_type', 'CrossEntropy')

        # Log model
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
