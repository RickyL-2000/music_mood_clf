import torch
import numpy as np
import time

def inference(model, mel, channel=80):
    """

    :param model: the trained model
    :param mel: mel feat, np.ndarray, shape(C, N) where C >= channel(80)
    :return:
    """
    mel = mel[:channel, :]
    mel = torch.tensor(mel).to(torch.float32).unsqueeze(0)

    model.eval()
    y_ = model(mel)
    y_ = torch.argmax(y_, -1).numpy()

    return y_

if __name__ == "__main__":
    model = torch.load("checkpoints/checkpoint_1_21/model-9600steps.pt")
    mel_path = "4Q/MER_audio_taffc_dataset/Q3_mel/MT0000004637.npy"
    with open(mel_path, 'rb') as f:
        mel = np.load(f, allow_pickle=True)

    t_start = time.time()
    y_ = inference(model, mel)
    t_end = time.time()
    print(y_)
    print(f"time: {t_end - t_start}s")

# %%
mel_path = "4Q/MER_audio_taffc_dataset/Q3_mel/MT0000004637.npy"
with open(mel_path, 'rb') as f:
    mel = np.load(f, allow_pickle=True)
