import argparse
import json
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from retriever import Retriever
from utils.hifigan import HiFiGANWrapper
from utils.libritts import DumpedLibriTTS
from speechset.utils.melstft import MelSTFT


parser = argparse.ArgumentParser()
parser.add_argument('--ret-config')
parser.add_argument('--ret-ckpt')
parser.add_argument('--hifi-config')
parser.add_argument('--hifi-ckpt')
parser.add_argument('--data-dir')
parser.add_argument('--audio1')
parser.add_argument('--audio2')
args = parser.parse_args()


with open(args.ret_config) as f:
    config = Config.load(json.load(f))

ckpt = torch.load(args.ret_ckpt, map_location='cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

retriever = Retriever(config.model)
retriever.load(ckpt)
retriever.to(device)
retriever.eval()

hifigan = HiFiGANWrapper(args.hifi_config, args.hifi_ckpt, device)

if args.data_dir is not None:
    libritts = DumpedLibriTTS(args.data_dir)
    testset = libritts.split(config.train.split)

    START = 400
    sid, text, mel, textlen, mellen = testset[START:START + 2]
else:
    def mel_fn(path: str, stft: MelSTFT = MelSTFT(config.data)) -> np.ndarray:
        wav, _ = librosa.load(path)
        return stft(wav)
    # pack
    mel = [mel_fn(path) for path in [args.audio1, args.audio2]]
    mellen = np.array([m.shape[0] for m in mel], dtype=np.int64)
    mel = np.stack([np.pad(m, [[0, mellen.max() - m.shape[0]], [0, 0]]) for m in mel])

# wrap
mel, mellen = torch.tensor(mel, device=device), torch.tensor(mellen, device=device)
# clip
MAX_MELLEN = 240
mel, mellen = mel[:, :MAX_MELLEN], torch.clamp_max(mellen, MAX_MELLEN)

with torch.no_grad():
    synth, aux = retriever(mel, mellen)
    # flip style
    style = aux['style'][[1, 0]]

    mixed, _ = retriever(mel, mellen, refstyle=style)
    # vocoding
    out = hifigan.forward(torch.cat([synth, mixed], dim=0))
    out = out.cpu().numpy()

    # [B, G, V, T]
    indices = torch.zeros_like(aux['logits']).scatter(
        2, aux['logits'].argmax(dim=2, keepdim=True), 1.).cpu().numpy()

os.makedirs('./synth', exist_ok=True)
for i, (wav, mlen) in enumerate(zip(out, mellen.repeat(2))):
    # unwrap
    librosa.output.write_wav(
        f'./synth/{i}.wav', wav[:mlen.item() * config.data.hop], sr=config.data.sr)

cmap = np.array(plt.get_cmap('viridis').colors)
# B x ([G, V, T], [T x R, mel])
for i, (idx, mel) in enumerate(zip(indices, synth.cpu().numpy())):
    rest = mel.shape[0] % config.model.reduction
    if rest > 0:
        mel = np.pad(mel, [[0, config.model.reduction - rest], [0, 0]], constant_values=np.log(config.data.eps))
    # [T, mel]
    mel = mel.reshape(-1, config.model.reduction, config.data.mel).mean(axis=1)
    # [T, mel]
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-7)
    # [mel, T]
    mel = np.flip(mel.transpose(1, 0), axis=0)
    # G x [V, T]
    for j, imap in enumerate(idx):
        # [mel + V, T]
        img = np.concatenate([mel, imap], axis=0)
        # [mel + V, T, 3]
        plt.imsave(f'./synth/{i}_{j}.png', cmap[(img * 255).astype(np.uint8)])
