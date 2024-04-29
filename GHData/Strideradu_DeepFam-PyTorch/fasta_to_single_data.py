import argparse
import sys
from Bio import SeqIO
import numpy as np
import random

from utils import GPCR_subfamily, GPCR_family, GPCR_label

parser = argparse.ArgumentParser()
parser.add_argument("input", help="path of the folder", type=str)
parser.add_argument("output", help="path of output file", type=str)
parser.add_argument("label", help="name of class", type=str)
parser.add_argument("--length", help="length of seq", type=int, default=1000)
parser.add_argument("--debug", help="whether debug", default=False, action="store_true")

try:
    args = parser.parse_args()


except:
    parser.print_help()
    sys.exit(1)

records = SeqIO.parse(args.input, format='fasta')
with open(args.output, 'w') as fout:
    for record in records:

        seq = str(record.seq)
        if not args.debug:
            if len(seq) > args.length:
                seq = seq[:args.length]

            elif len(seq) < args.length:
                seq = seq + '_' * (args.length - len(seq))

            label = GPCR_label[args.label]
            id = record.id
            print("\t".join([str(label), seq, id]), file=fout)

        else:
            # we only keep partial sequences for prediction
            seq_len = len(seq)
            # generate sequences with 50% to 95% length, each with 3 copy with diffent start and end position
            percentage = np.linspace(0.2, 1, 17)
            for perc in percentage:
                for i in range(5):
                    if perc != 1:
                        output_len = int(seq_len * perc)
                        ran_range = seq_len - output_len
                        seq_start = random.randrange(ran_range)
                        seq_out = seq[seq_start:seq_start + output_len]

                        if len(seq_out) > args.length:
                            seq_out = seq_out[:args.length]

                        elif len(seq_out) < args.length:
                            seq_out = seq_out + '_' * (args.length - len(seq_out))

                        label = GPCR_label[args.label]
                        id = record.id + "perc{}_{}".format(perc, i)
                        print("\t".join([str(label), seq_out, id]), file=fout)
