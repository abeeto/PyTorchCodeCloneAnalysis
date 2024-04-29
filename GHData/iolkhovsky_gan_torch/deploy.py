import argparse
import cv2
import torch
from os.path import isfile, join

from dataset.utils import decode_img
from gan.generator import ImageGenerator
from utils import force_create_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="generator.pt",
                        help="Abs path to generator's model")
    parser.add_argument("--feat_cnt", type=int, default=100,
                        help="Length of the feature vector")
    parser.add_argument("--out_dir", type=str, default="generated",
                        help="Abs path to the output image")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: cpu/cuda")
    parser.add_argument("--cnt", type=int, default=1,
                        help="Count of images being generated")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen_model = ImageGenerator(feature_vector=args.feat_cnt)
    gen_model = gen_model.to(args.device)
    if isfile(args.model):
        gen_model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
    else:
        print(f"{args.model} doesnt exist. Using default weights.")
    gen_model.eval()
    force_create_dir(args.out_dir)
    with torch.no_grad():
        for i in range(args.cnt):
            random_descriptor = torch.randn(1, args.feat_cnt)
            fake_img = gen_model(random_descriptor)[0]
            fake_img = decode_img(fake_img.numpy())
            cv2.imwrite(join(args.out_dir, f"{i}.jpg"), fake_img)
