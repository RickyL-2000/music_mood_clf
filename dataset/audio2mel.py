# 这个脚本用于将vocal的音频转化为mel-feats
import librosa
import numpy as np
from tqdm import tqdm
import pickle
import yaml

import os
import multiprocessing as mp

import sys
# os.chdir("..")
# sys.path.append("..")
# print(sys.path)
from utils.utils import *
from utils.signal import load_audio

# def get_files_list(base_dir, config, dataset="emotifymusic"):
#     if dataset=="emotifymusic":

def audio2mel(y, fs, to_db=True, channel=80, n_fft=512, hop_length=128, win_length=512, fmax=8000):
    if len(y) <= 0:
        print('audio sequence should be longer than 0 elements')
        return None
    mel = librosa.feature.melspectrogram(y=y, sr=fs, n_fft=n_fft, hop_length=hop_length, win_length=win_length, fmax=fmax)
    if to_db:
        mel = librosa.power_to_db(mel, ref=np.max)
    mel = mel[:channel, :]
    return mel

def main_slave(song_idx_begin, song_idx_end, task_idx, config, resume=False):
    if resume and os.path.exists(f'{base_dir}/checkpoints/audio2mel_checkpoints/task_{task_idx}_checkpoints.pkl'):
        with open(f'{base_dir}/checkpoints/audio2mel_checkpoints/task_{task_idx}_checkpoints.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
    else:
        mkdir(f'{base_dir}/checkpoints/audio2mel_checkpoints')
        checkpoint = set()
    codec = config['codec']
    genres = config['genres']
    data_path = f"{base_dir}/{config['path']}"
    for song_idx in tqdm(range(song_idx_begin, song_idx_end), ncols=80, desc=f'audio2mel task {task_idx} processing'):
        genre = genres[int((song_idx-1) / 100)]
        y, fs = load_audio(f"{data_path}/{genre}/{(song_idx-1)%100+1}.{codec}")
        mel = audio2mel(y, fs,
                        channel=config['channel'],
                        n_fft=config['n_fft'],
                        hop_length=config['hop_length'],
                        win_length=config['win_length'],
                        fmax=config['fmax']
                        )
        if mel is not None:
            mkdir(f"{data_path}/{genre}_mel")
            with open(f'{data_path}/{genre}_mel/{(song_idx-1)%100+1}_mel.npy', 'wb') as f:
                np.save(f, mel)
            if resume:
                checkpoint.add(song_idx)

        if int((song_idx - song_idx_begin) * 100 / (song_idx_end - song_idx_begin)) % 10 == 0:
            with open(f'{base_dir}/checkpoints/audio2mel_checkpoints/task_{task_idx}_checkpoints.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)

def main_master(num_tasks=32, dataset="emotifymusic", resume=False):
    with open(f'{base_dir}/config/audio2mel_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = config[dataset]
    song_idx_begin = config['start_idx']
    song_idx_end = config['end_idx']

    num_cores = int(mp.cpu_count())
    pool = mp.Pool(num_cores)
    task_size = (song_idx_end + 1 - song_idx_begin) // num_tasks
    # final_set_ver = load_from_cache('final_set_ver')
    results = [pool.apply_async(main_slave, args=(
        song_idx_begin + task_idx*task_size, min(song_idx_end+1, song_idx_begin + (task_idx+1)*task_size),
        task_idx, config, resume))
               for task_idx in range(num_tasks)]
    pool.close()
    pool.join()

    print('Everything Done.')

# %%
if __name__ == "__main__":
    # you may run this file in IDE (like pycharm)
    # main_master(num_tasks=32, dataset="emotifymusic", resume=False)
    dataset = "emotifymusic"
    with open(f'{base_dir}/config/audio2mel_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = config[dataset]
    main_slave(1, 401, 0, config, resume=False)
