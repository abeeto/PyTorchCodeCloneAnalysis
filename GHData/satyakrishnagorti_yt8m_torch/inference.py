import os
import pdb
import glob
import torch
import argparse

import metric
from validate import Validation
from models import graph
from dataloaders.tfrecord_dataset import TFRecordFrameDataSet

torch.manual_seed(0)
# disable tensorflow deprecation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser(description='Inference')
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--vocab-size', type=int, default=3862)
parser.add_argument('--model-path', type=str, required=True, help="path to load model")
parser.add_argument('--inference-data-pattern', type=str, default='/data/yt8m/v3/frame/validate*.tfrecord')
parser.add_argument('--model', type=str, required=True, help="model to use for training")
parser.add_argument('--segment-labels', type=bool, default=True)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--num-features', type=list, default=[1024, 128])
parser.add_argument('--features', type=list, default=['rgb', 'audio'])
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--top-n', type=int, default=100_000)
parser.add_argument('--validate-file', type=str, default='/data/yt8m/v3valid.csv')
parser.add_argument('--output-file', type=str, default='data/inference_output.csv')
parser.add_argument('--generate-blend-file', type=bool, default=False)


class Inference:
    def __init__(self, data_loader, model, device, args):
        self.val_obj = Validation(data_loader, model, device, label_cache="/data/yt8m/label_cache")
        self.args = args

    def run_inference(self):
        print("running inference")
        with torch.no_grad():
            self.val_obj.validation_helper(self.args.generate_blend_file, self.args.top_n, self.args.output_file)


def get_inference_data_loader(args):
    if args.validate_file:
        lines = open(args.validate_file, 'r')
        val_files = [l.strip() for l in lines]
        files = glob.glob(args.inference_data_pattern)
        files = [f for f in files if f.split('/')[-1] in val_files]
    else:
        print('Full inference')
        files = glob.glob(args.inference_data_pattern)
    record_dataset = TFRecordFrameDataSet(files, segment_labels=args.segment_labels)
    data_loader = torch.utils.data.DataLoader(record_dataset,
                                              batch_size=args.batch_size,
                                              collate_fn=record_dataset.get_collate_fn(),
                                              num_workers=args.num_workers)
    return data_loader


def load_torch_model(model: graph.GraphModel, device, args):
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    model = model.double()
    return model


def main():
    args = parser.parse_args()
    print(args)
    device = torch.device(args.device)
    model = graph.GraphModel(vocab_size=args.vocab_size)
    model = load_torch_model(model, device, args)
    data_loader = get_inference_data_loader(args)
    infer_obj = Inference(data_loader, model, device, args)
    infer_obj.run_inference()


if __name__ == '__main__':
    main()
