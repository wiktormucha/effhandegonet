from typing import List
from spock import spock
from spock import SpockBuilder
from models import models
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple
import torch.optim as optim
import albumentations as A


class Model(Enum):
    resnet = 'NewModel'
    EffHandNet = 'EffHandNet'
    EffnetSimpleRegression = 'CustomEgocentricRegression'
    EffnetWaterfall = 'EfficientWaterfallEgo'
    CustomHeatmapsModel = 'CustomHeatmapsModel'
    SwinV2Egocentric = 'SwinV2Egocentric'
    MobileNetV3Egocentric = 'MobileNetV3Egocentric'
    ConvNext3Egocentric = 'ConvNext3Egocentric'
    EffHandEgoNet_FPHAB = 'EffHandEgoNet_FPHAB'
    CustomEgocentric3D = 'CustomEgocentric3D'
    CustomEgocentric = 'CustomEgocentric'
    CustomEgocentric3Depth = 'CustomEgocentric3Depth'
    CustomEgocentric3Depth2 = 'CustomEgocentric3Depth2'
    CustomEgocentric3Depth3 = 'CustomEgocentric3Depth3'
    CustomEgocentric3DepthEstimatedSingleFrame = 'CustomEgocentric3DepthEstimatedSingleFrame'


@spock
class TrainingConfig:
    #     max_epochs: int
    #     batch_size: int
    debug: bool = False
    device: int
    lr: float
    max_epochs: int
    early_stopping: int
    weight_decay: float
    momentum: float
    grad_clipping: float
    load_model: bool
    load_model_path: str
    model_type: Model
    run_name: str
    train_flag: bool
    test_flag: bool
    one_batch_flag: bool
    experiment_name: str
    criterion: str
    num_workers: int
    data_type: str
    dataset: str
#     early_stopping: int
#     emplying_objs: bool = False
#     checkpoint_pth: str


@spock
class Data:
    path: str
    batch_size: int
    img_size: List[int]
    max_train_samples: int = 55742
    max_val_samples: int = 11638
    norm_mean: List[float]
    norm_std: List[float]
