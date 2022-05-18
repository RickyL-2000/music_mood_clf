"""
Save the entire model with desired parameters
"""
import torch
from torch import package

import os
import argparse
import yaml

from models.mood_recog import MoodRecog

def save_model(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    if config['trainer']['checkpoint_path'] == "checkpoints":
        config['trainer']['checkpoint_path'] += f"/checkpoint_{config['version']}"

    # load checkpoint
    model = MoodRecog(config=config['model']).to(torch.device("cpu"))
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict['model'])

    # save model
    # default save path: the directory of state dict checkpoint
    # 这样是不行的！这样保存模型仍然会依赖源代码
    # torch.save(model, f"{os.path.dirname(checkpoint_path)}/model-{os.path.basename(checkpoint_path)[11: -4]}.pt")

    with package.PackageExporter(f"{os.path.dirname(checkpoint_path)}/model-{os.path.basename(checkpoint_path)[11: -4]}.pt") as exp:
        exp.extern("numpy.**")
        exp.intern("layers.**")
        exp.intern("models.**")
        exp.save_pickle("emotion_recog", "model.pt", model)

def main():
    parser = argparse.ArgumentParser(description="Train Mood Recognition")
    parser.add_argument("--config", '-c', type=str, default="config/mood_clf_config.yaml",
                        help="yaml format configuration file")
    parser.add_argument("--checkpoint", '-cp', default="", type=str, nargs="?",
                        help="checkpoint file path to load. (default=\"\")")
    args = parser.parse_args()
    save_model(args.config, args.checkpoint)

if __name__ == "__main__":
    # main()
    config = "checkpoints/checkpoint_1_21/config.yaml"
    checkpoint = "checkpoints/checkpoint_1_21/checkpoint-9600steps.pkl"
    save_model(config, checkpoint)
