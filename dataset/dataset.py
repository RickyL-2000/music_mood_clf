from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pickle
import pandas as pd

from multiprocessing import Manager
import os

from utils import *

class EmotifyDataset(Dataset):
    def __init__(self, config):
        # super(EmotifyDataset, self).__init__()
        self.config = config
        self.data_path = f"{base_dir}/{self.config['path']}"
        self.files = self.get_files()

        self.allow_cache = self.config["allow_cache"]
        if self.allow_cache:
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.files))]

        self.labels = self.get_soft_labels()

    def get_files(self):
        files = {}
        for song_idx in range(self.config['start_idx'], self.config['end_idx']+1):
            genre = self.config['genres'][int((song_idx-1) / 100)]
            files[song_idx] = f"{self.data_path}/{genre}_mel/{(song_idx-1)%100+1}_mel.npy"
        return files

    def get_soft_labels(self):
        if os.path.exists(f"{self.data_path}/data_soft.csv"):
            df = pd.read_csv(f"{self.data_path}/data_soft.csv")
        else:
            df = pd.read_csv(f"{self.data_path}/data.csv")
            df.drop(columns=df.columns[[11, 12, 13, 14, 15, 16]], axis=1, inplace=True)
            df = df.groupby(by=["track id"], as_index=False).agg('mean')
            df.to_csv(f"{self.data_path}/data_soft.csv", index=False)
        soft_labels = {}
        for song_idx in range(self.config['start_idx'], self.config['end_idx']+1):
            soft_labels[song_idx] = np.array(df.iloc[song_idx-1])[1:]
        return soft_labels

    def __getitem__(self, song_idx):
        item = {
            'mel': torch.tensor(self.mel_load_fn(self.files[song_idx])),
            'label': torch.tensor(self.labels[song_idx])
        }

        if self.allow_cache:
            self.caches[song_idx] = item

        return item

    def __len__(self):
        return len(self.files)

    @staticmethod
    def mel_load_fn(path):
        with open(path, 'rb') as f:
            mel = np.load(f, allow_pickle=True)
        return mel

class EmotifyCollator(object):
    def __init__(self):
        pass

    def __call__(self, batch_dic):
        mel_max_length = np.max([batch_dic[i]['mel'].shape[1] for i in range(len(batch_dic))])
        mel_batch = []
        label_batch = []
        for dic in batch_dic:
            mel = dic['mel']
            # zero-padding: https://blog.csdn.net/sinat_36618660/article/details/100122745
            mel_batch.append(F.pad(mel, pad=(0, mel_max_length-mel.shape[1], 0, 0), mode='constant', value=0.0).numpy())
            label_batch.append(dic['label'].numpy())
        # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting
        # the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        ret = {
            'mel': torch.tensor(np.array(mel_batch)),
            'label': torch.tensor(np.array(label_batch))
        }
        return ret
