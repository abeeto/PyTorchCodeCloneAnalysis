import argparse
import random
import numpy as np

DIR="/home/qingyu/data/kevin/"
#parse arguments
parser = argparse.ArgumentParser(description="Experiemts for Coreference Resolution (by qyyin)\n")

parser.add_argument("-embedding_dir",default = DIR+"features/mention_data/word_vectors.npy", type=str, help="specify dir for embedding file")
parser.add_argument("-DIR",default = DIR, type=str, help="Home direction")
parser.add_argument("-language",default = "en", type=str, help="language")
parser.add_argument("-gpu",default = 3, type=int, help="GPU number")
parser.add_argument("-reduced",default = 0, type=int, help="GPU number")
parser.add_argument("-random_seed",default = 12345, type=int, help="Random Seed")

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
