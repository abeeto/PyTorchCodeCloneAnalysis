#!/usr/bin/env python

# Load config, tokenizer and model and save them on disk.

import sys

from argparse import ArgumentParser

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel
)

def argparser():
    ap = ArgumentParser()
    ap.add_argument('model')
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)

    config.save_pretrained(args.model)
    tokenizer.save_pretrained(args.model)
    model.save_pretrained(args.model)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
