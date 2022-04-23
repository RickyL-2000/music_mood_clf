import torch
import torch.distributed
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.utils.data
from tensorboardX import SummaryWriter

import os
import yaml
import logging
import argparse
import sys
from collections import defaultdict

from tqdm import tqdm

from utils import *
from dataset.dataset import *
from models.mood_recog import MoodRecog

class Trainer(object):
    def __init__(self,
                 steps,
                 epochs,
                 data_loader,
                 sampler,
                 model,
                 criterion,
                 optimizer,
                 scheduler,
                 config,
                 logger,
                 device=torch.device("cpu"),
                 version="0_1"
                 ):
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.loss_fn = criterion[config['loss_fn']]
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.version = version
        # self.checkpoint_path = f"{config['checkpoint_path']}/checkpoint_{version}"

        self.logger = logger
        self.writer = SummaryWriter(f"{self.config['checkpoint_path']}/tensorboard")
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.total_eval_score = defaultdict(float)

        self.tqdm = None
        self.finish_train = False

    def run(self):
        self.tqdm = tqdm(initial=self.steps,
                         total=self.config["train_max_steps"],
                         desc="[train]")
        while True:
            self._train_epoch()

            if self.finish_train:
                break
        self.tqdm.close()
        self.logger.info("Finish training")

    def _train_epoch(self):
        for batch_idx, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # check interval
            self._check_log_interval()
            self._check_save_interval()
            self._check_eval_interval()

            if self.finish_train:
                return

        self.epochs += 1
        self.logger.info(f"Epochs: {self.epochs} | Steps: {self.steps}")

        if self.config["distributed"]:
            self.sampler["train"].set_epoch(self.epochs)

    def _train_step(self, batch):
        self.optimizer.zero_grad()

        mel = batch["mel"].to(torch.float32).to(self.device)
        y = batch["label"].to(torch.float32).to(self.device)

        y_ = self.model(mel)

        loss = self.loss_fn(y_, y)
        loss.backward()
        self.optimizer.step()

        self.total_train_loss["train/onset_loss"] += loss.item()

        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _eval_epoch(self):
        self.logger.info(f"(Steps: {self.steps}) Start Evaluation.")
        self.model.eval()

        batch_idx = 1
        for batch_idx, batch in enumerate(tqdm(self.data_loader["eval"], desc="[eval]"), 1):
            self._eval_step(batch)

        self.logger.info(f"(Steps: {self.steps}) Finished Evaluation")

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= batch_idx
            self.logger.info(f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.")

        # average score
        for key in self.total_eval_score.keys():
            self.total_eval_score[key] /= batch_idx
            self.logger.info(f"{key} = {self.total_eval_score[key]:.4f}.")

        # record
        self._write_to_tensorboard(self.total_eval_loss)
        self._write_to_tensorboard(self.total_eval_score)

        # reset
        self.total_eval_loss = defaultdict(float)
        self.total_eval_score = defaultdict(float)

        # restore mode
        self.model.train()

    def _eval_step(self, batch):
        mel = batch["mel"].to(torch.float32).to(self.device)
        y = batch["label"].to(torch.float32).to(self.device)

        y_ = self.model(mel)

        loss = self.loss_fn(y_, y)

        self.total_eval_loss["eval/onset_loss"] += loss.item()

        # NOTE: map y_ to [0, 1]
        if self.config["loss_fn"] == "BCEWithLogits":
            y_ = F.sigmoid(y_)

        # compute score
        self.total_eval_loss["eval/score"] += self.criterion(y, y_)

    def _check_log_interval(self):
        if self.steps % self.config['log_interval_steps'] == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config['log_interval_steps']
                self.logger.info(f'(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.')
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)  # loss重设为0

    def _check_save_interval(self):
        if self.steps % self.config['save_interval_steps'] == 0:
            self.save_checkpoint(self.config['checkpoint_path'], f"checkpoint-{self.steps}steps.pkl")
            self.logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config['eval_interval_steps'] == 0:
            self._eval_epoch()

    def _check_train_finish(self):
        if self.steps >= self.config['train_max_steps']:
            self.finish_train = True

    def _write_to_tensorboard(self, item):
        for key, value in item.items():
            self.writer.add_scalar(key, value, self.steps)

    def save_checkpoint(self, checkpoint_path, f_name):
        state_dict = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "steps": self.steps,
            "epochs": self.epochs
        }
        if self.config['distributed']:
            state_dict['model'] = self.model.module.state_dict()
        else:
            state_dict['model'] = self.model.state_dict()
        mkdir(checkpoint_path)
        torch.save(state_dict, f"{checkpoint_path}/{f_name}")

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if self.config["distributed"]:
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        if not load_only_params:
            self.steps = state_dict['steps']
            self.epochs = state_dict['epochs']
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])

def main():
    parser = argparse.ArgumentParser(description="Train Mood Recognition")
    parser.add_argument("--config", '-c', type=str, default="config/mood_clf_config.yaml",
                        help="yaml format configuration file")
    parser.add_argument("--resume", '-r', default="", type=str, nargs="?",
                        help="checkpoint file path to resume training. (default=\"\")")
    parser.add_argument("--verbose", '-v', type=int, default=1,
                        help="logging level. higher is more logging. (default=1)")
    parser.add_argument("--rank", "--local_rank", default=0, type=int,
                        help="rank for distributed training. no need to explicitly specify.")
    parser.add_argument("--dataset_name", '-d', type=str, default="emotifymusic",
                        help="Dataset name")
    args = parser.parse_args()

    # ----------------- load config ----------------- #
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    # update checkpoint path
    config['trainer']['checkpoint_path'] += f"/checkpoint_{config['version']}"
    mkdir(f"{config['trainer']['checkpoint_path']}")
    # save config
    with open(f"{config['trainer']['checkpoint_path']}/config.yaml", 'w') as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)

    # ----------------- set logger ----------------- #
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(filename=f"{config['trainer']['checkpoint_path']}/train.log")
    streamhandler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    filehandler.setFormatter(formatter)
    streamhandler.setFormatter(formatter)
    if args.verbose > 1:
        filehandler.setLevel(logging.DEBUG)
        streamhandler.setLevel(logging.DEBUG)
    elif args.verbose > 0:
        filehandler.setLevel(logging.INFO)
        streamhandler.setLevel(logging.INFO)
    else:
        filehandler.setLevel(logging.WARN)
        streamhandler.setLevel(logging.WARN)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    if args.verbose <= 0:
        logger.warning("Skip DEBUG/INFO messages")

    # ----------------- resume ----------------- #
    if len(args.resume) == 0:
        logger.info("\nStart Training Pipeline")
    else:
        logger.info("\n\nResume Training Pipeline")
    logger.info("Config:")
    for key, value in config.items():
        logger.info(f"{key} = {value}")

    # ----------------- device ----------------- #
    # FIXME: distributed!
    args.distributed = False
    torch.manual_seed(config['dataset']['seed'])
    if not torch.cuda.is_available() or not config['using_gpu']:
        device = torch.device("cpu")
        logger.info("Using CPU")
    else:
        logger.info(f"Using GPU {config['rank']}")
        device = torch.device("cuda")
        torch.cuda.manual_seed(config['dataset']['seed'])
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(config['rank'])
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if "WORLD_SIZE" in os.environ:
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.distributed = args.world_size > 1
        if args.distributed:
            torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # --------------------------------------- dataset ---------------------------------------- #
    dataset = EmotifyDataset(config=config['dataset'][config['dataset_name']])

    train_size = int(len(dataset) * config['dataset'][config['dataset_name']]['train_eval_split'])
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset,
                                                                [train_size, eval_size],
                                                                generator=torch.Generator().manual_seed(
                                                                    config['dataset']['seed']))
    logger.info(f"The number of training files = {train_size}")
    logger.info(f"The number of evaluation files = {eval_size}")

    dataset = {'train': train_dataset, 'eval': eval_dataset}
    collator = EmotifyCollator()

    # --------------------------------------- criterion ---------------------------------------- #
    criterion = {
        'MSE': torch.nn.MSELoss(),
        'BCE': torch.nn.BCELoss()
    }

    # --------------------------------------- define models ---------------------------------------- #
    model = MoodRecog(config=config['model']).to(device)

    # --------------------------------------- optimizer ---------------------------------------- #
    optimizer = {
        'Adam': torch.optim.Adam(model.parameters(),
                                 **config['trainer']['optimizer_params'][config['trainer']['optimizer']])
    }

    # --------------------------------------- scheduler ---------------------------------------- #
    scheduler = {
        'StepLR': torch.optim.lr_scheduler.StepLR(optimizer=optimizer[config['trainer']['optimizer']],
                                                  **config['trainer']['scheduler_params'])
    }

    # --------------------------------------- distributed ---------------------------------------- #
    sampler = {'train': None, 'eval': None}
    if args.distributed:
        sampler['train'] = DistributedSampler(
            dataset=dataset['train'],
            num_replicas=args.world_size,
            rank=config['rank'],
            shuffle=True
        )
        sampler['eval'] = DistributedSampler(
            dataset=dataset['eval'],
            num_replicas=args.world_size,
            rank=config['rank'],
            shuffle=False
        )

    if args.distributed:
        try:
            from apex.parallel import DistributedDataParallel
        except ImportError:
            raise ImportError("apex is not installed. please check https://github.com/NVIDIA/apex.")
        model = DistributedDataParallel(model)

    # print model
    logger.info(model)
    simple_table([
        ('Checkpoints Path', config['trainer']['checkpoint_path']),
        ('Config File', args.config)
    ])

    # --------------------------------------- dataLoader ---------------------------------------- #
    data_loader = {
        'train': DataLoader(
            dataset=dataset['train'],
            shuffle=False if args.distributed else True,
            collate_fn=collator,
            batch_size=config['dataset'][config['dataset_name']]['batch_size'],
            num_workers=config['dataset'][config['dataset_name']]['num_workers'],
            sampler=sampler['train'],
            pin_memory=config['dataset'][config['dataset_name']]['pin_memory']
        ),
        'eval': DataLoader(
            dataset=dataset['eval'],
            shuffle=False if args.distributed else True,
            collate_fn=collator,
            batch_size=config['dataset'][config['dataset_name']]['batch_size'],
            num_workers=config['dataset'][config['dataset_name']]['num_workers'],
            sampler=sampler['eval'],
            pin_memory=config['dataset'][config['dataset_name']]['pin_memory']
        )
    }

    # --------------------------------------- trainer ---------------------------------------- #
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config['trainer'],
        logger=logger,
        device=device,
        version=config['version']
    )

    if len(args.resume) != 0:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Successfully resumed from {args.resume}.")

    # --------------------------------------- training ---------------------------------------- #
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(f"{config['trainer']['checkpoint_path']}", f"checkpoint-{trainer.steps}steps.pkl")
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")

if __name__ == "__main__":
    main()