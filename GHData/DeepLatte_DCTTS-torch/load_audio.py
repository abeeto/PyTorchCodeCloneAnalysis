import numpy as np
import librosa
import librosa.display
import os
import re
import codecs
import unicodedata
import matplotlib.pyplot as plt
from tqdm import tqdm
from params import param
from torch.utils.data.dataset import Dataset
from gAtt import guideAttentionNT

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(param.vocab)}
    idx2char = {idx: char for idx, char in enumerate(param.vocab)}
    return char2idx, idx2char

def loadTrnscpt(fpath):
    char2idx, idx2char = load_vocab()

    wavfList, scriptList, lenList = [], [], []
    lines = codecs.open(fpath, 'r', 'utf-8').readlines()
    
    for line in lines:
        wavName, _, script = line.strip().split('|')
        wavPath = os.path.join(param.filePath, 'wavs', wavName+'.wav')
        wavfList.append(wavPath)
        processedTxt, lengthTxt = textProcess(script, char2idx)
        scriptList.append(processedTxt)
        lenList.append(lengthTxt)
        
    return wavfList, scriptList, lenList

def audio_load(audPathList, txtlenList):
    '''
    return
    melList : (N, T/r, n_mels)
    magList : (N, T, 1+n_fft/2)
    '''
    # melList, magList = [], []
    for idx, audioDir in tqdm(enumerate(audPathList)):
        mel, mag = load_spectrogram(audioDir)

        t = len(mel)
        num_paddings = param.r - (t % param.r) if t % param.r != 0 else 0
        mel = np.pad(mel, [[0, num_paddings], [0, 0]], mode="constant")
        mag = np.pad(mag, [[0, num_paddings], [0, 0]], mode="constant")

        # T -> T/r, Reduction
        mel = mel[::param.r, :]
        gMat = guideAttentionNT(0.2, txtlenList[idx], len(mel))
        np.save(os.path.join(param.filePath, "mel_dir", "mel_{}.npy".format(idx)), mel)
        np.save(os.path.join(param.filePath, "mag_dir", "mag_{}.npy".format(idx)), mag)
        np.save(os.path.join(param.filePath, "gMat_dir", "gMat_{}.npy".format(idx)), gMat)


def load_spectrogram(audioDir):
    # load an audio file
    y, sr = librosa.load(audioDir, sr=param.sr)

    # trimmed audio
    y, _  = librosa.effects.trim(y)
    # print('ok')
    # plt.figure(figsize=(12, 5))
    # plt.plot(range(len(y)),y)
    # plt.show()

    # preemphasis
    y = np.append(y[0], y[1:] - param.preemphasis * y[:-1])
    # plt.figure(figsize=(12, 5))
    # plt.plot(y)
    # plt.title('Pre-emphasis')
    # plt.show()

    # STFT
    y = librosa.stft(y=y,
                     n_fft=param.n_fft,
                     hop_length=param.hopSize,
                     win_length=param.winSize)
    mag = np.abs(y) # (1+n_fft//2, T)
    # plt.figure(figsize=(12,6))
    # librosa.display.specshow(mag, sr=param.sr, x_axis='time', y_axis='mel')
    # plt.title('magnitude')
    # plt.colorbar(format='%+0.1f')
    # plt.savefig('./pic/mag_stft.png')
    # plt.show()
    
    # mel bank
    mel_bank = librosa.filters.mel(param.sr, param.n_fft, param.n_mels)  # (n_mels, 1+n_fft//2)
    mel = np.dot(mel_bank, mag) # (n_mels, T)
    # plt.figure(figsize=(12,6))
    # librosa.display.specshow(mel, sr=param.sr, x_axis='time', y_axis='mel')
    # plt.title('mel')
    # plt.colorbar(format='%+0.1f')
    # plt.savefig('./pic/mel_stft.png')
    # plt.show()

    # to decibel
    mel = 20 * np.log10(np.maximum(1e-5, mel))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    # plt.figure(figsize=(12,6))
    # librosa.display.specshow(mel, sr=param.sr, x_axis='time', y_axis='mel')
    # plt.title('mel power')
    # plt.colorbar(format='%+0.1f')
    # plt.savefig('./pic/mel_db.png')
    # plt.show()

    # plt.figure(figsize=(12,6))
    # librosa.display.specshow(mag, sr=param.sr, x_axis='time', y_axis='mel')
    # plt.title('mag power')
    # plt.colorbar(format='%+0.1f')
    # plt.savefig('./pic/mag_db.png')
    # plt.show()

    # normalize
    mel = np.clip((mel - param.ref_db + param.max_db) / param.max_db, 1e-8, 1)
    mag = np.clip((mag - param.ref_db + param.max_db) / param.max_db, 1e-8, 1)
    # plt.figure(figsize=(12,6))
    # librosa.display.specshow(mel, sr=param.sr, x_axis='time', y_axis='mel')
    # plt.colorbar(format='%+0.1f')
    # plt.savefig('./pic/normalized_mel.png')
    # plt.title('mel power')
    # plt.show()
    # plt.figure(figsize=(12,6))
    # librosa.display.specshow(mag, sr=param.sr, x_axis='time', y_axis='mel')
    # plt.title('mag power')
    # plt.colorbar(format='%+0.1f')
    # plt.savefig('./pic/normalized_mag.png')
    # plt.show()

    mel = mel.T.astype(np.float32) # (T, n_mels)
    mag = mag.T.astype(np.float32) # (T, 1+n_fft//2)

    return mel, mag


def textProcess(text, char2idx):
    # text processing.
    # text normalization

    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                    if unicodedata.category(char) != 'Mn')
    text = text.lower()
    
    # remove characters not in vocab using regular experession.
    text = re.sub('[^{}]'.format(param.vocab), ' ', text)
    text = re.sub('[ ]+', ' ', text)
    
    text += "E"  # E: EOS
    text = [char2idx[char] for char in text]
    lengthText = len(text)
    # text = np.array(text, np.int32).tostring() # string , binary

    return text, lengthText

if __name__ == '__main__':
    '''
    Save mel, mag, gMat list from the directory of audio

    scriptList : scripts from metadata.csv
    lenList : the list includes lengths of the script
    '''
    transcript = os.path.join(param.filePath, param.transcriptName)

    # if not os.path.exists(os.path.join(param.filePath, "script_dir")):
    #     os.mkdir(os.path.join(param.filePath, "script_dir"))
    #     print("make script_dir folder")
    if not os.path.exists(os.path.join(param.filePath, "mel_dir")):
        os.mkdir(os.path.join(param.filePath, "mel_dir"))
        print("make mel_dir folder")
    if not os.path.exists(os.path.join(param.filePath, "mag_dir")):
        os.mkdir(os.path.join(param.filePath, "mag_dir"))
        print("make mag_dir folder")
    if not os.path.exists(os.path.join(param.filePath, "gMat_dir")):
        os.mkdir(os.path.join(param.filePath, "gMat_dir"))
        print("make gMat_dir folder")

    wavfList, _, txtlenList = loadTrnscpt(transcript)

    print("saving speech data...")
    audio_load(wavfList, txtlenList)
    print("All pre-process have been finished.")