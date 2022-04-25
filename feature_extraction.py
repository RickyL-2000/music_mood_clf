import yaml
from tqdm import tqdm

from utils import *
from dataset.audio2mel import audio2mel

def process_4Q():
    with open(f'{base_dir}/config/audio2mel_config.yaml') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config = config["4Q"]
    codec = "mp3"
    data_path = f"{base_dir}/{config['path']}"
    for quad in config["category"]:
        for f_name in tqdm(os.listdir(f"{data_path}/{quad}"), ncols=80, desc=f"audio2mel task {quad}"):
            y, fs = load_audio(f"{data_path}/{quad}/{f_name}")
            mel = audio2mel(y, fs,
                            channel=config['channel'],
                            n_fft=config['n_fft'],
                            hop_length=config['hop_length'],
                            win_length=config['win_length'],
                            fmax=config['fmax']
                            )
            if mel is not None:
                mkdir(f"{data_path}/{quad}_mel")
                with open(f'{data_path}/{quad}_mel/{f_name[:-4]}.npy', 'wb') as f:
                    np.save(f, mel)

if __name__ == "__main__":
    process_4Q()
