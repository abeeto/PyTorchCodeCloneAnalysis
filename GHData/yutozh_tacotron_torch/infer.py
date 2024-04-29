import os
import glob 
import time
import argparse
import numpy as np
from tqdm import tqdm
import hydra
# from utils import audio
# from utils.utils import *
from srcs.synthesizer.taco_synthesizer import *

class Synthesizer():
    """Main entrance for synthesizer"""

    def __init__(self, hparams, args):
        self.hparams = hparams
        self.args = args
        self.synthesizer = eval(hparams.synthesizer_type)(hparams, args)

    def __call__(self):
        labels = [fp for fp in glob.glob(
            os.path.join(self.args.label_dir, '*'))]
        print(labels)
        for i, label_filename in enumerate(tqdm(labels)):
            start = time.time()

            generated_acoustic, acoustic_filename = self.synthesizer(label_filename)
            if generated_acoustic is None:
                print("Ignore {}".format(os.path.basename(label_filename)))
                continue
            end = time.time()
            spent = end - start
            n_frame = generated_acoustic.shape[0]
            audio_lens = n_frame * self.hparams.hop_size / self.hparams.sample_rate
            print("Label: {}, generated wav length: {}, synthesis time: {}, RTF: {}".format(
                os.path.basename(label_filename), audio_lens, spent, spent / audio_lens))

@hydra.main(config_path='conf/', config_name='infer')
def main(config):
    if config.alignment_dir is not None:
        os.makedirs(config.alignment_dir, exist_ok=True)
    print(config.hparams)
    print(config.args)

    synthesizer = Synthesizer(config.hparams, config.args)
    synthesizer()

if __name__ == '__main__':
    main()