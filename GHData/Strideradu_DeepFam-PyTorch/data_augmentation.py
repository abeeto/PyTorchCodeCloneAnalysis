import pandas as pd
import numpy as np
import random
import os

# augment data by slice only partial of the sequences
df = pd.read_csv("C:/Research/20181018_partial_seq/test.txt", sep='\t', header = None)
save_path = "C:/Research/20181018_partial_seq/"
percentage = np.linspace(0.1, 0.9, 9)
for perc in percentage:
    filename = "cv0_test_perc{}.txt".format(perc)
    with open(os.path.join(save_path, filename), 'w') as fout:
        for index, row in df.iterrows():
            label = row[0]
            seq = row[1].strip('_')
            seq_len = len(seq)
            for i in range(10):
                output_len = int(seq_len * perc)
                ran_range = seq_len - output_len
                seq_start = random.randrange(ran_range)
                seq_out = seq[seq_start:seq_start + output_len]

                if len(seq_out) > 1000:
                    seq_out = seq_out[:1000]

                elif len(seq_out) < 1000:
                    seq_out = seq_out + '_' * (1000 - len(seq_out))
                id = "test_{}_{}".format(index, i)
                print("\t".join([str(label), seq_out, id]), file=fout)