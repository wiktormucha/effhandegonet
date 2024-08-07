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


@spock
class Data:
    path: str
    batch_size: int
    img_size: List[int]
    max_train_samples: int = 55742
    max_val_samples: int = 11638
    norm_mean: List[float]
    norm_std: List[float]


class HandPoseType(Enum):
    gt_hand_pose = 'gt_hand_pose'
    own_hand_pose = 'own_hand_pose'
    mediapipe_hand_pose = 'mediapipe_hand_pose'
    ego_handpoints = 'ego_handpoints'
    hand_resnet50 = 'hand_resnet50'


class ObjPoseType(Enum):
    gt = 'GT'
    yolov7 = 'YoloV7'
    own = 'own'


class DataDimension(Enum):
    two_d = '2D'
    thre_d = '3D'


class Optimizer(Enum):
    sgd = 'SGD'
    adam = 'Adam'
    adamw = 'AdamW'


class Model(Enum):
    simple_lstm = 'SimpleLSTM'
    lstm_3d_6d_objinclasss = 'LSTM_3D_6D_OBJINCLASSS'
    lstm_2d = "LSTM_2D"
    lstm_2d_2d_objinclasss = "LSTM_2D_2D_OBJINCLASSS"
    lstm_2d_2dobj = 'LSTM_2D_2DOBJ'
    lstm_2dbb_2dobj = 'LSTM_2DBB_2DOBJ'
    lstm_2d_2dobj_embed = 'LSTM_2D_2DOBJ_EMBED'
    simple_transformer = 'TransfomerKeypoints'
    simple_transformer1 = 'TransfomerKeypoints_1'
    simple_transformer2 = 'TransfomerKeypoints_2'
    simple_transformer3 = 'TransfomerKeypoints_3'
    simple_transformer_loop = 'ActionTransformer'
    ActionTransformer_NoLinear = 'ActionTransformer_NoLinear'
    lstm_2d_2d_objinclass = 'LSTM_2D_2DOBJINCLASS'


@spock
class ModelConfig():
    model_type: Model
    input_dim: int
    hidden_layers: int
    out_dim: int
    dropout: float
    dropout_att: float
    trans_num_layers: int
    trans_num_heads: int
    seq_length: int
    load_checkpoint: bool
    checkpoint_path: str


@spock
class OptimizerConfig:
    type: Optimizer = 'SGD'
    lr: float = 0.01
    weight_decay: Optional[float] = 0.0
    momentum: Optional[float] = 0.0


@spock
class TrainingConfigAction:
    seed_num: int
    max_epochs: int
    batch_size: int
    device: int
    early_stopping: int
    wandb_name: str
    scheduler_steps: List[int]
    run_type: str
    debug: bool = False


@spock
class DataConfig:
    dataset: str
    data_dir: str = 'Asdf'
    annotation_train: str
    no_of_input_frames: int
    using_obj_bb: bool
    using_obj_label: bool
    hand_pose_type: HandPoseType
    obj_pose_type: ObjPoseType
    apply_vanishing: bool
    vanishing_proability: float
    obj_to_vanish: int
    albumentation_pth: str
