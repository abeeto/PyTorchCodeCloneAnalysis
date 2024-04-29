import argparse
from lib2to3.pgen2.token import AT
from typing import List, Tuple
import csv
import pickle
from collections import OrderedDict
from pathlib import Path

import gym
import numpy as np
import pandas as pd
# import streaming_image_env
from gym.envs.registration import registry as env_registry

# from metrics import nss, kldiv, auc_shuffled

# import run_baselines
# from utils import maybe_tqdm, assert_equal
# from visualization import upscale_smap

# Prevent PyCharm from removing this import
# assert hasattr(streaming_image_env, 'StreamingImageEnv')


class AtariHead:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.meta = self._read_meta()

    def _read_meta(self):
        meta = pd.read_csv(self.dataset_dir / 'meta_data.csv').dropna(axis=0)
        meta['TrialNumber'] = meta['TrialNumber'].astype(int)
        meta['NumberOfFrames'] = meta['NumberOfFrames'].astype(int)

        data = {
            int(path.name.split('_')[0]): path.name[:-len('.tar.bz2')]
            for path in self.dataset_dir.iterdir()
            if path.name.endswith('.tar.bz2')
        }
        meta['frames'] = meta['TrialNumber'].map(lambda tn: data[tn] + '.tar.bz2')
        meta['gazes'] = meta['TrialNumber'].map(lambda tn: data[tn] + '.txt')

        meta['Game'] = meta['Game'].str.replace('Mspacman', 'MsPacman')

        return meta

    def read_gazes(self, trial_number: int):
        run_gazes = self.get_run(trial_number).gazes
        cols = ['frame_id', 'episode_id', 'score', 'duration', 'unclipped_reward', 'action', 'gaze_positions']

        records = []
        with (self.dataset_dir / run_gazes).open() as fp:
            csv_reader = csv.reader(fp)
            next(csv_reader)  # header
            for row in csv_reader:
                row = [(None if item == 'null' else item) for item in row]
                values = row[:len(cols) - 1]
                gazes = row[len(cols) - 1:]
                if gazes == [None]:
                    gazes = []
                else:
                    gazes = [float(v) for v in gazes]
                    gazes = list(zip(gazes[::2], gazes[1::2]))
                values.append(gazes)
                d = OrderedDict(zip(cols, values))
                for field in ['duration', 'unclipped_reward', 'action']:
                    if d[field] is not None:
                        d[field] = int(d[field])
                records.append(d)

        df_gazes = pd.DataFrame.from_records(records)

        df_gazes['gaze_num'] = df_gazes['gaze_positions'].map(len)

        df_gazes['run_name'] = df_gazes['frame_id'].map(lambda s: s[:s.rindex('_')])
        df_gazes['frame_idx'] = df_gazes['frame_id'].map(lambda s: int(s[s.rindex('_') + 1:]))

        return df_gazes

    def get_run(self, trial_number: int):
        matches = self.meta[self.meta['TrialNumber'] == trial_number]
        assert len(matches) == 1
        return matches.iloc[0]

    def get_env_name(self, trial_number):
        # but yeah we need to change this hehe 
        # env_name = f'StreamingImageEnvNoFrameskip{trial_number}-v0'
        env_name = 'Breakout-v0'


        if env_name not in env_registry.env_specs:
            run = self.get_run(trial_number)
            gym.envs.register(
                id=env_name,
                entry_point='streaming_image_env:StreamingImageEnv',
                kwargs={'tar_path': self.dataset_dir / run['frames'], 'base_env_name': run['Game'] + 'NoFrameskip-v4'}
            )

        return env_name

    def game_trials(self, game: str):
        return self.meta[self.meta['Game'] == game]['TrialNumber']


if __name__ == '__main__':
    dataset = AtariHead(Path('breakout'))

    # now testing each individual function within it ...
    gazes = dataset.read_gazes(trial_number=132)
    # print('gazes are: ', gazes.head())
    # print('gazes summary: ', gazes.describe())
    # print('gazes shape: ', gazes.shape)
    # print('gazes[0]: ', gazes.iloc[0])

    run = dataset.get_run(trial_number=132)
    # print('run is: ', run)

    env_name = dataset.get_env_name(trial_number=132)
    # print('env_name is :', env_name)

    trials = dataset.game_trials('Breakout')
    print('trials are', trials)

    ##### now we want to do something with those gaze positions
    



    ##### we want to do something with RL loss functions ...