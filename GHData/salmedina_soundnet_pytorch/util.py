import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path

local_config = {
            'batch_size': 64, 
            'load_size': 22050*10,
            'phase': 'extract'
            }


def load_from_list(name_list, config=local_config):
    assert len(name_list) == config['batch_size'], \
            "The length of name_list({})[{}] is not the same as batch_size[{}]".format(
                    name_list[0], len(name_list), config['batch_size'])
    audios = np.zeros([config['batch_size'], config['load_size'], 1, 1])
    for idx, audio_path in enumerate(name_list):
        sound_sample, _ = load_audio(audio_path)
        audios[idx] = preprocess(sound_sample, config)
        
    return audios


# NOTE: Load an audio as the same format in soundnet (MP3)
# 1. Keep original sample rate (which conflicts their own paper described as 22050)
# 2. Audio must be in mono
# 3. Value range must be [-256, 256]
def load_audio(audio_path, sample_rate=22050, mono=True):
    # By default, librosa will resample the signal to 22050Hz(sr=None). And range in (-1., 1.)
    sound_sample, sr = librosa.load(audio_path, sr=sample_rate, mono=mono)
    
    assert sample_rate == sr

    return sound_sample, sr


def load_from_txt(txt_path, config=local_config):
    txt_list = []
    with open(txt_path, 'r') as txt_file:
        txt_list = [line.strip() for line in txt_file.readlines() if line.strip() != '']

    audios = []
    audio_paths = []
    for idx, audio_path in enumerate(txt_list):
        if idx % 20 == 0:
            print('Processing: {}'.format(idx))
        sound_sample, _ = load_audio(audio_path)
        audios.append(preprocess(sound_sample, config))
        audio_paths.append(audio_path)
        
    return audios, audio_paths


def gen_audio_from_txt(txt_path, config=local_config):
    '''Audio loader generator'''
    txt_list = []
    with open(txt_path, 'r') as txt_file:
        txt_list = [line.strip() for line in txt_file.readlines() if line.strip() != '']

    for audio_path in tqdm(txt_list):
        sound_sample, _ = load_audio(audio_path)
        yield preprocess(sound_sample, config), audio_path


def gen_audio_from_dir(dir, file_ext='.wav', config=local_config):
    '''Audio loader from dir generator'''
    txt_list = []
    
    audio_path_list = Path(dir).glob(f'*{file_ext}')

    for audio_path in tqdm(audio_path_list):
        sound_sample, _ = load_audio(audio_path)
        yield preprocess(sound_sample, config), audio_path 


def preprocess(raw_audio, config=local_config):
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Make minimum length available
    length = config['load_size']
    if length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(length/raw_audio.shape[0] + 1))

    # Make equal training length
    if config['phase'] != 'extract':
        raw_audio = raw_audio[:length]

    assert len(raw_audio.shape) == 1, "Audio is not mono"
    assert np.max(raw_audio) <= 256, "Audio max value beyond 256"
    assert np.min(raw_audio) >= -256, "Audio min value beyond -256"

    # Shape for network is 1 x DIM x 1 x 1
    raw_audio = np.reshape(raw_audio, [1, 1, -1, 1])

    return raw_audio.copy()
