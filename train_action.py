from models import models
from utils.trainer import TrainerH2OAction, TrainerFPHAAction
from datasets.h2o import get_H2O_actions_dataloader
import torch
import torch.nn as nn
from spock_dataclasses import *
import wandb
import sys
from utils.general_utils import freeze_seeds, count_parameters, define_optimizer, load_cfg, make_model_action, get_wandb_cfg
from datasets.FPHA import get_FPHAB_action_dataset


def main() -> None:
    """
    Main training loop
    """

    # Build config
    config = SpockBuilder(OptimizerConfig, ModelConfig, TrainingConfigAction, DataConfig,
                          desc='Quick start example').generate()

    freeze_seeds(seed_num=config.TrainingConfigAction.seed_num)

    a = load_cfg(config.DataConfig.albumentation_pth)

    # Load config yaml to wandb
    wandbcfg_pth = sys.argv[2]
    # opening a file
    wandbcfg = get_wandb_cfg(wandbcfg_pth)

    wandbcfg['applied_aug'] = a.albumentations

    if config.TrainingConfigAction.debug == False:
        logger = wandb.init(
            # set the wandb project where this run will be logged
            project=config.TrainingConfigAction.wandb_name,
            config=wandbcfg)
    else:
        logger = None

    if config.DataConfig.dataset == 'h2o':
        dataloader = get_H2O_actions_dataloader(
            config, albumentations=a.albumentations)
    elif config.DataConfig.dataset == 'fpha':
        dataloader = get_FPHAB_action_dataset(
            config, albumentations=a.albumentations)
    else:
        raise ValueError('Dataset not supported: ', config.DataConfig.dataset)

    # Create model
    model = make_model_action(model_cfg=config.ModelConfig,
                              device=config.TrainingConfigAction.device, dataset=config.DataConfig.dataset)

    print('Number of parameters to learn:', count_parameters(model))

    # If loading weights from checkpoin
    if config.ModelConfig.load_checkpoint:
        model.load_state_dict(torch.load(
            config.ModelConfig.checkpoint_path, map_location=torch.device(config.TrainingConfigAction.device)))
        print("Model's checkpoint loaded")

    criterion = nn.CrossEntropyLoss()
    optimizer = define_optimizer(model, config.OptimizerConfig)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.TrainingConfigAction.scheduler_steps, gamma=0.5, last_epoch=- 1, verbose=True)

    if config.DataConfig.dataset == 'h2o':
        trainer = TrainerH2OAction(model, criterion, optimizer,
                                   config.TrainingConfigAction, model_config=config.ModelConfig, wandb_logger=logger, scheduler=scheduler)

        if config.TrainingConfigAction.run_type == 'train':
            print(
                f'Starting training on device: {config.TrainingConfigAction.device}')
            model = trainer.train(dataloader['train'], dataloader['val'])

        elif config.TrainingConfigAction.run_type == 'test':
            model = trainer.test_model(dataloader['val'])
            model = trainer.test_h2o(dataloader['test'])
        else:
            raise ValueError('Run type not supported')

    elif config.DataConfig.dataset == 'fpha':

        trainer = TrainerFPHAAction(model, criterion, optimizer,
                                    config.TrainingConfigAction, model_config=config.ModelConfig, wandb_logger=logger, scheduler=scheduler)
        if config.TrainingConfigAction.run_type == 'train':
            print(
                f'Starting training on device: {config.TrainingConfigAction.device}')
            model = trainer.train(dataloader['train'], dataloader['val'])

        elif config.TrainingConfigAction.run_type == 'test':
            model = trainer.test_model(dataloader['val'])
        else:
            raise ValueError('Run type not supported')

    if logger:
        wandb.finish()


if __name__ == '__main__':
    main()
