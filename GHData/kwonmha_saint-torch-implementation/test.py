
import argparse
import time

import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from data import KT1Data, SaintDataset
from utils import get_model

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--model", type=str, default="saint",
                    choices=["saint", "vsaint", "dkt", "vrnn"])
model_args, _ = parser.parse_known_args()
model = get_model(model_args.model)
parser = model.get_parser(parser)
parser.add_argument("--model_path", type=str,
                    # required=True,
                    default="/ext2/mhkwon/knowledge-tracing/saved-models/Ednet/saint-KT1-train-all_d128_l4_dr0.1_lr0.001_b128_adam.pt"
                    )
parser.add_argument("--data_path", type=str, default="/ext_hdd/mhkwon/knowledge-tracing/data/Ednet/KT1-test-all.csv")
parser.add_argument("--min_items", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--norm_first", action="store_true", default=False)
parser.add_argument("--gpu", type=str, default="2")
args = parser.parse_args()

device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")

test_data = KT1Data(args.data_path, True)
test_dataset = SaintDataset(test_data.data, max_seq=args.seq_len, min_items=args.min_items)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
print("loaded data")

args.total_ex = 18143
args.total_cat = 7 # (1 ~ 7)
args.total_in = 2 # (O, X)

model = model(args).to(device)
state_dict = torch.load(args.model_path)
model.load_state_dict(state_dict)
model.to(device)
print("loaded model")
model.eval()

all_labels = []
all_outs = []
with torch.no_grad():
    start_time = time.time()
    for i, data in enumerate(test_loader):
        inputs, labels = data

        outputs = model(inputs["qids"].to(device), inputs["part_ids"].to(device), inputs["correct"].to(device))
        mask = (inputs["qids"] != 0).to(device)

        outputs = torch.masked_select(outputs, mask)
        labels = torch.masked_select(labels.to(device), mask)

        # calc auc
        all_labels.extend(labels.view(-1).data.cpu().numpy())
        all_outs.extend(outputs.view(-1).data.cpu().numpy())

        if i % 500 == 499:
            print("[%5d] %.2f" % (i + 1, time.time() - start_time))

    auc = roc_auc_score(all_labels, all_outs)

    print("AUC - {:.4f} ".format(auc))
