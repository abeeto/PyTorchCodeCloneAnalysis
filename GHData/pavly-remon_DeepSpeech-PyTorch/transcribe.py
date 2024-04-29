"""
This file is to print arabic text after decoding speech 

"""
import argparse
import warnings

from opts import add_decoder_args, add_inference_args
from utils import load_model

warnings.simplefilter('ignore')

from decoder import GreedyDecoder

import torch

from data.data_loader import SpectrogramParser
from model import DeepSpeech
import os.path
import json

"""ASR EECE_19 edit"""
###################### PRRAG edit for printing arabic output################################
char_map_str = """
A ء
B آ
C أ
D ؤ
E إ
F ئ
G ا
H ب
I ة
J ت
K ث
L ج
M ح
N خ
O د
P ذ
Q ر
R ز
S س
T ش
U ص
V ض
W ط
X ظ
Y ع
Z غ
a ف
b ق
c ك
d ل
e م
f ن
g ه
h و
i ى
j ي
k ً
l ٌ
m ٍ
n َ
o ُ
p ِ
q ّ
r ْ
s ـ
"""
out_sen = []
char_seq = []
char_map = {}
for line in char_map_str.strip().split('\n'):
    en, ar = line.split()
    char_map[en] = ar
###################################################################################################

def decode_results(model, decoded_output, decoded_offsets):
    results = {
        "output": [],
        "_meta": {
            "acoustic_model": {
                "name": os.path.basename(args.model_path)
            },
            "language_model": {
                "name": os.path.basename(args.lm_path) if args.lm_path else None,
            },
            "decoder": {
                "lm": args.lm_path is not None,
                "alpha": args.alpha if args.lm_path is not None else None,
                "beta": args.beta if args.lm_path is not None else None,
                "type": args.decoder,
            }
        }
    }

    for b in range(len(decoded_output)):
        for pi in range(min(args.top_paths, len(decoded_output[b]))):
            result = {'transcription': decoded_output[b][pi]}
            if args.offsets:
                result['offsets'] = decoded_offsets[b][pi].tolist()
            results['output'].append(result)

    ## PRRAG edit for printing arabic output#################
    for sen in result['transcription']:
        for i in range(len(sen)):
            if sen[i] == ' ' or sen[i] == '*' or sen[i] == '\n':
                char_seq.append(sen[i])
            else:
                char_seq.append(char_map[sen[i]])
        out_sen.append(''.join(char_seq))
        char_seq.clear()

    with open(args.out_trans_path,'w') as ofile:
        for _ in out_sen:
            ofile.write(_)
    #############################################################
    return results


def transcribe(audio_path, parser, model, decoder, device):
    spect = parser.parse_audio(audio_path).contiguous()
    spect = spect.view(1, 1, spect.size(0), spect.size(1))
    spect = spect.to(device)
    input_sizes = torch.IntTensor([spect.size(3)]).int()
    out, output_sizes = model(spect, input_sizes)
    decoded_output, decoded_offsets = decoder.decode(out, output_sizes)
        	
    return decoded_output, decoded_offsets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser = add_inference_args(parser)
    parser.add_argument('--audio-path', default='audio.wav',
                        help='Audio file to predict on')
    parser.add_argument('--offsets', dest='offsets', action='store_true', help='Returns time offset information')
    parser = add_decoder_args(parser)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.cuda)
    parser.add_argument('--out_trans_path' , default=None , help='choose path of output transcript') # we added this argument

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    else:
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))

    parser = SpectrogramParser(model.audio_conf, normalize=True)

    decoded_output, decoded_offsets = transcribe(args.audio_path, parser, model, decoder, device)
    print(json.dumps(decode_results(model, decoded_output, decoded_offsets)))
