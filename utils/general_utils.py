import numpy as np
import cv2 as cv2
import torch
import torch.nn as nn
from models import models
import torch.optim as optim
import importlib.util
import matplotlib.pyplot as plt
from enum import Enum
import random
import yaml
from constants import COLORMAP


class EgocentricModelType(Enum):
    mediapipe = 'MediaPipe'
    effhandegonet = 'EffHandEgoNet'
    effhandnet = 'EffHandNet'
    poseresnet50 = 'PoseResNet50'


def freeze_seeds(seed_num=42, max_num_threads=16):

    torch.set_num_threads(max_num_threads)
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)
    print('Seed set to:', seed_num)


def define_optimizer(model, optimizer_cfg):

    if optimizer_cfg.type == 'SGD':
        optimizer = optim.SGD(model.parameters(
        ), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay, momentum=optimizer_cfg.momentum)
    elif optimizer_cfg.type == 'AdamW':
        optimizer = optim.AdamW(
            model.parameters(), lr=optimizer_cfg.lr, weight_decay=optimizer_cfg.weight_decay)
    return optimizer


def count_parameters(model: nn.Module) -> int:
    """
    Counts parameters for training in a given model.

    Args:
        model (nn.Module): Input model

    Returns:
        int: No. of trainable parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vector_to_heatmaps(keypoints: np.array, scale_factor: int = 1, out_size: int = 128, n_keypoints: int = 21) -> np.array:
    """
    Creates 2D heatmaps from keypoint locations for a single image.

    Args:
        keypoints (np.array): array of size N_KEYPOINTS x 2
        scale_factor (int, optional): Factor to scale keypoints (factor = 1 when keypoints are org size). Defaults to 1.
        out_size (int, optional): Size of output heatmap. Defaults to MODEL_IMG_SIZE.

    Returns:
        np.array: Heatmap
    """
    heatmaps = np.zeros([n_keypoints, out_size, out_size])
    for k, (x, y) in enumerate(keypoints):
        x, y = int(x * scale_factor), int(y * scale_factor)
        if (0 <= x < out_size) and (0 <= y < out_size):
            heatmaps[k, int(y), int(x)] = 1

    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


def blur_heatmaps(heatmaps: np.array) -> np.array:
    """
    Blurs heatmaps using GaussinaBlur of defined size

    Args:
        heatmaps (np.array): Input heatmap

    Returns:
        np.array: Output heatmap
    """
    heatmaps_blurred = heatmaps.copy()
    for k in range(len(heatmaps)):
        if heatmaps_blurred[k].max() == 1:
            heatmaps_blurred[k] = cv2.GaussianBlur(heatmaps[k], (51, 51), 3)
            heatmaps_blurred[k] = heatmaps_blurred[k] / \
                heatmaps_blurred[k].max()
    return heatmaps_blurred


def project_points_3D_to_2D(xyz: np.array, K: np.array) -> np.array:
    """
    Projects 3D coordinates into 2D space. Taken from FreiHAND dataset repository.

    Args:
        xyz (np.array): 3D keypoints
        K (np.array): camera intrinsic

    Returns:
        np.array: 2D keypoints
    """
    uv = np.matmul(xyz, K.T)
    return uv[:, :2] / uv[:, -1:]


def heatmaps_to_coordinates(heatmaps: np.array, img_size: int) -> np.array:
    """
    Transforms heat,aps to 2d keypoints

    Args:
        heatmaps (np.array): Input heatmap

    Returns:
        np.array: Output points
    """
    # heatmaps = heatmaps.cpu().detach().numpy()
    batch_size = heatmaps.shape[0]
    # sums = heatmaps.sum(axis=-1)
    sums = heatmaps.sum(axis=-1).sum(axis=-1)

    sums = np.expand_dims(sums, [2, 3])
    normalized = heatmaps / sums

    x_prob = normalized.sum(axis=2)

    y_prob = normalized.sum(axis=3)

    arr = np.tile(np.float32(np.arange(0, img_size)), [batch_size, 21, 1])

    x = (arr * x_prob).sum(axis=2)

    y = (arr * y_prob).sum(axis=2)
    keypoints = np.stack([x, y], axis=-1)

    return keypoints / img_size


def make_model(model_cfg, device='cpu', parameter_info=True):

    model = getattr(models, model_cfg.model_type)()
    # model = getattr(models, model_cfg.model_type)(
    #     model_cfg, device=device)

    model = model.to(device)

    print(f'Model created on device: {device}')

    # If loading weights from checkpoin
    if model_cfg.load_model:
        model.load_state_dict(torch.load(
            model_cfg.load_model_path, map_location=torch.device(device)))
        print("Model's checkpoint loaded")

    if parameter_info:
        print('Number of parameters to learn:', count_parameters(model))

    return model


def load_cfg(path):
    print('loading configuration file', path)
    spec = importlib.util.spec_from_file_location('cfg_file', path)
    cfg_file = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg_file)
    return cfg_file


def draw_keypoints_on_single_hand(pts):
    for finger, params in COLORMAP.items():
        plt.plot(
            pts[params["ids"], 0],
            pts[params["ids"], 1],
            params["color"],
        )


def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return img


def make_model_action(model_cfg, device, dataset, parameter_info=True):
    model = getattr(models, model_cfg.model_type)(
        model_cfg, device=device, dataset=dataset)
    model = model.to(device)

    print(f'Model created on device: {device}')

    # If loading weights from checkpoin
    if model_cfg.load_checkpoint:
        model.load_state_dict(torch.load(
            model_cfg.checkpoint_path, map_location=torch.device(device)))
        print("Model's checkpoint loaded")

    if parameter_info:
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def get_wandb_cfg(wandbcfg_pth):
    with open(wandbcfg_pth, 'r') as stream:
        try:
            # Converts yaml document to python object
            wandbcfg = yaml.safe_load(stream)
        # Program to convert yaml file to dictionary
        except yaml.YAMLError as e:
            print(e)

    return wandbcfg
