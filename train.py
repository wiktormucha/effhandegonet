from utils.trainer import TrainerH2O, TrainerFreiHAND, Trainer_FPHA
from datasets.FreiHAND import get_FreiHAND_dataloaders
from utils.general_utils import count_parameters
from utils.loses import IoULoss, EffHandEgoNetLoss
import torch
import torch.optim as optim
import sys
import random
import numpy as np
from spock_dataclasses import *
from datasets.h2o import get_H2O_dataloader
import torch.nn as nn
import wandb
import yaml
from datasets.FPHA import get_FPHAB_dataset
from constants import ALBUMENTATION_TRAIN
from utils.loses import FPHALoss

MAX_NUM_THREADS = 16


def freeze_seeds(seed_num=42):

    torch.set_num_threads(MAX_NUM_THREADS)
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)


def main() -> None:
    # Set up fixed shuffeling for hyperparameter tuning
    freeze_seeds()

    # Build config
    cfg = SpockBuilder(Data, TrainingConfig,
                       desc='Quick start example').generate()

    if cfg.TrainingConfig.dataset == 'freihand':

        config = {
            "data_dir": cfg.Data.path,
            "batch_size": cfg.Data.batch_size,
            "device": cfg.TrainingConfig.device,
            "num_workers": cfg.TrainingConfig.num_workers,
        }

        dataloader = get_FreiHAND_dataloaders(config)
        criterion = IoULoss()
        albumentations_train = ALBUMENTATION_TRAIN

    elif cfg.TrainingConfig.dataset == 'h2o':
        dataloader = get_H2O_dataloader(cfg)
        albumentations_train = dataloader['albumentation_train']
        criterion = EffHandEgoNetLoss()
    elif cfg.TrainingConfig.dataset == 'fphab':
        dataloader = get_FPHAB_dataset(cfg)
        albumentations_train = dataloader['albumentation_train']
        criterion = FPHALoss()
    else:
        raise ValueError('Dataset not supported')

    # Load config yaml to wandb

    wandbcfg_pth = sys.argv[2]
    # opening a file
    with open(wandbcfg_pth, 'r') as stream:
        try:
            # Converts yaml document to python object
            wandbcfg = yaml.safe_load(stream)
        # Program to convert yaml file to dictionary
        except yaml.YAMLError as e:
            print(e)

    wandbcfg['albu'] = str(albumentations_train)
    if cfg.TrainingConfig.debug:
        logger = None
    else:
        logger = wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.TrainingConfig.experiment_name,
            config=wandbcfg)

    model = getattr(models, cfg.TrainingConfig.model_type)()
    model = model.to(cfg.TrainingConfig.device)
    model.freeze_backbone()
    model.freeze_hand_pose()
    # If loading weights from checkpoint
    if cfg.TrainingConfig.load_model:
        model.load_state_dict(torch.load(
            cfg.TrainingConfig.load_model_path, map_location=torch.device(cfg.TrainingConfig.device)))
        print("Model's checkpoint loaded")

    print('Number of parameters to learn:', count_parameters(model))

    optimizer = optim.SGD(model.parameters(
    ), lr=cfg.TrainingConfig.lr, weight_decay=cfg.TrainingConfig.weight_decay, momentum=cfg.TrainingConfig.momentum)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[40, 100], gamma=0.5, last_epoch=- 1, verbose=True)

    if cfg.TrainingConfig.dataset == 'freihand':
        trainer = TrainerFreiHAND(model, criterion, optimizer,
                                  cfg.TrainingConfig, wandb_logger=logger, grad_clip=cfg.TrainingConfig.grad_clipping, scheduler=scheduler)

    elif cfg.TrainingConfig.dataset == 'fphab':
        trainer = Trainer_FPHA(model, criterion, optimizer,
                               cfg.TrainingConfig, wandb_logger=logger, grad_clip=cfg.TrainingConfig.grad_clipping, scheduler=scheduler)
    elif cfg.TrainingConfig.dataset == 'h2o':
        trainer = TrainerH2O(model, criterion, optimizer,
                             cfg.TrainingConfig, wandb_logger=logger, grad_clip=cfg.TrainingConfig.grad_clipping, scheduler=scheduler)
    else:
        raise ValueError('Dataset not supported')

    print(f'Starting training on device: {cfg.TrainingConfig.device}')
    model = trainer.train(
        train_dataloader=dataloader['train'], val_dataloader=dataloader['val'])

    # trainer.predict(dataloader['val'])

    if cfg.TrainingConfig.debug:
        wandb.finish()


if __name__ == '__main__':
    main()
