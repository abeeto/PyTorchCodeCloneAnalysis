import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('test_list', type=str)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
args = parser.parse_args()

score_npz_files = [np.load(x) for x in args.score_files]

if args.score_weights is None:
    score_weights = [1] * len(score_npz_files)
else:
    score_weights = args.score_weights
    if len(score_weights) != len(score_npz_files):
        raise ValueError("Only {} weight specifed for a total of {} score files"
                         .format(len(score_weights), len(score_npz_files)))

f_val = open(args.test_list, "r")
val_list = f_val.readlines()
match_count = 0

line_id = 0
for line in val_list:
    line_info = line.split(" ")
    input_video_label = int(line_info[2])
    pred_index = np.argmax((score_weights[0] * score_npz_files[0][line_id] + score_weights[1] * score_npz_files[1][line_id]) / float(score_weights[0] + score_weights[1]))

    if pred_index == input_video_label:
            match_count += 1
    line_id += 1

print("Accuracy is %4.4f" % (float(match_count)/len(val_list)))