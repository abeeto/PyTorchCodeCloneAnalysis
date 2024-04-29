import torch
import argparse
from tqdm import tqdm
from torch.cuda.amp import autocast
from utils.data.dataloader import create_dataloader
from utils.misc import load_config, build_model
from utils.metrics import ConfusionMatrix


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cfg', type=str, required=True,
                        help="config file")
    parser.add_argument('--csv', type=str, required=True,
                        help="dataset CSV file")
    parser.add_argument('--pth', type=str, required=True,
                        help="checkpoint")
    parser.add_argument('--workers', type=int, default=4,
                        help="number of dataloader workers")
    parser.add_argument('--no_amp', action='store_true',
                        help="disable automatic mix precision")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = load_config(args.cfg)

    model = build_model(cfg)
    model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.pth)['model_state_dict'])

    dataloader = create_dataloader(args.csv,
                                   batch_size=cfg.batch_size,
                                   image_size=cfg.input_size,
                                   num_workers=args.workers)

    metric = ConfusionMatrix(cfg.num_classes)
    metric.reset()
    pbar = tqdm(dataloader, bar_format="{l_bar}{bar:30}{r_bar}")
    with torch.no_grad():
        for (images, annos) in pbar:
            images = images.to(device)
            annos = annos.to(device)

            with autocast(enabled=(not args.no_amp)):
                logits = model(images)

            preds = torch.argmax(logits, axis=1)
            metric.update(preds, annos)

    mIoU = metric.IoUs.mean()
    accuracy = metric.accuracy
    print("mIoU: %.3f, accuracy: %.3f" % (mIoU, accuracy))


if __name__ == '__main__':
    main()
