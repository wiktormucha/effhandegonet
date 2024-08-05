import numpy as np
import cv2 as cv2
import torch
import torch.nn as nn
from models import models
import torch.optim as optim
import importlib.util
import matplotlib.pyplot as plt
from enum import Enum


class EgocentricModelType(Enum):
    mediapipe = 'MediaPipe'
    effhandegonet = 'EffHandEgoNet'
    effhandnet = 'EffHandNet'
    poseresnet50 = 'PoseResNet50'


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


def vector_to_heatmaps_tensor(keypoints: torch.Tensor, scale_factor: int = 1, out_size: int = 128, n_keypoints: int = 21) -> torch.Tensor:
    """
    Creates 2D heatmaps from keypoint locations for a batch of images.

    Args:
        keypoints (torch.Tensor): tensor of size BATCH_SIZE x N_KEYPOINTS x 2
        scale_factor (int, optional): Factor to scale keypoints (factor = 1 when keypoints are org size). Defaults to 1.
        out_size (int, optional): Size of output heatmap. Defaults to MODEL_IMG_SIZE.

    Returns:
        torch.Tensor: Heatmap
    """
    batch_size = keypoints.size(0)
    heatmaps = torch.zeros(
        [batch_size, n_keypoints, out_size, out_size], device=keypoints.device)

    # Scale keypoints in-place
    keypoints = (keypoints * scale_factor).long()

    # Ensure keypoints are within bounds
    keypoints = keypoints.clamp(0, out_size - 1)

    # Use advanced indexing to set the heatmap values
    heatmaps[torch.arange(batch_size)[:, None, None], torch.arange(
        n_keypoints), keypoints[..., 1], keypoints[..., 0]] = 1

    # Assuming blur_heatmaps can handle batched input
    heatmaps = blur_heatmaps(heatmaps)
    return heatmaps


# def blur_heatmaps(heatmaps: np.array) -> np.array:
#     """
#     Blurs heatmaps using GaussianBlur of defined size

#     Args:
#         heatmaps (np.array): Input heatmap of shape (batch_size, channels, height, width)

#     Returns:
#         np.array: Output heatmap of same shape as input
#     """
#     heatmaps_blurred = heatmaps.clone()
#     for i in range(heatmaps.shape[0]):  # iterate over batch size
#         for j in range(heatmaps.shape[1]):  # iterate over channels
#             if heatmaps_blurred[i, j].max() == 1:
#                 heatmaps_blurred[i, j] = cv2.GaussianBlur(
#                     heatmaps[i, j], (51, 51), 3)
#                 heatmaps_blurred[i, j] = heatmaps_blurred[i,
#                                                           j] / heatmaps_blurred[i, j].max()
#     return heatmaps_blurred
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


# def project_points_3D_to_2D(xyz: list, K: list) -> np.array:
#     """
#     Projects 3D coordinates into 2D space. Taken from FreiHAND dataset repository.

#     Args:
#         xyz (list): 3D keypoints
#         K (list): camera intrinsic

#     Returns:
#         np.array: 2D keypoints
#     """
#     xyz = np.array(xyz)
#     K = np.array(K)
#     uv = np.matmul(K, xyz.T).T
#     return uv[:, :2] / uv[:, -1:]

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


def project_points_2D_to_3D(xy: np.array, z: np.array, K: list) -> np.array:
    """
    Projects 2D coordinates into 3D space.

    Args:
        xy (np.array): 2D keypoints
        z (np.array): estimated depth
        K (list): camera intrinsic

    Returns:
        np.array: 3D keypoints
    """
    xy = np.concatenate((xy, np.ones((xy.shape[0], 1))), axis=-1)
    K_inv = np.linalg.inv(np.array(K))
    xyz = np.matmul(K_inv, xy.T).T
    xyz *= z.reshape(21, 1)
    return xyz
# def project_points_2D_to_3D(uv: list, K: list, depth: float) -> np.array:
#     """
#     Projects 2D coordinates into 3D space using camera intrinsic.

#     Args:
#         uv (list): 2D keypoints
#         K (list): camera intrinsic
#         depth (float): depth of the 3D points

#     Returns:
#         np.array: 3D keypoints
#     """
#     uv = np.array(uv)
#     K_inv = np.linalg.inv(np.array(K))
#     dummy_z = np.ones((uv.shape[0], 21, 1))
#     hstack = np.append(uv, dummy_z, axis=2).T
#     # hstack = np.hstack((uv, dummy_z)).T
#     xyz = np.matmul(K_inv, hstack)
#     xyz = (depth / xyz[-1, :]) * xyz[:-1, :]
#     return xyz.T


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


def heatmaps_to_coordinates_tensor(heatmaps: torch.Tensor, img_size: int) -> torch.Tensor:
    """
    Transforms heatmaps to 2d keypoints

    Args:
        heatmaps (torch.Tensor): Input heatmap

    Returns:
        torch.Tensor: Output points
    """
    batch_size = heatmaps.shape[0]
    sums = heatmaps.sum(dim=-1).sum(dim=-1)

    sums = sums.unsqueeze(-1).unsqueeze(-1)
    normalized = heatmaps / sums

    x_prob = normalized.sum(dim=2)
    y_prob = normalized.sum(dim=3)

    arr = torch.arange(0, img_size, device=heatmaps.device).float().repeat(
        batch_size, 21, 1)

    x = (arr * x_prob).sum(dim=2)
    y = (arr * y_prob).sum(dim=2)

    keypoints = torch.stack([x, y], dim=-1)

    return keypoints / img_size


# def heatmaps_to_coordinates_tensors(heatmaps: np.array, img_size: int, device=0) -> np.array:
#     """
#     Transforms heat,aps to 2d keypoints

#     Args:
#         heatmaps (np.array): Input heatmap

#     Returns:
#         np.array: Output points
#     """
#     # heatmaps = heatmaps.cpu().detach().numpy()
#     batch_size = heatmaps.shape[0]
#     sums = heatmaps.sum(-1).sum(-1).to(device)

#     # sums = sums.expand((2, 3))  # torch.expand(sums, [2, 3])
#     sums = sums.unsqueeze(axis=2)
#     sums = sums.unsqueeze(axis=3)
#     normalized = heatmaps / sums

#     x_prob = normalized.sum(axis=2)

#     y_prob = normalized.sum(axis=3)

#     arr = torch.tile(torch.arange(0, img_size), [batch_size, 21, 1]).to(device)
#     x = (arr * x_prob).sum(axis=2)

#     y = (arr * y_prob).sum(axis=2)
#     keypoints = torch.stack([x, y], axis=-1)
#     return (keypoints / img_size)


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


def make_model2(model_name, device='cpu', model_path=None, parameter_info=True):

    model = getattr(models, model_name)()
    # model = getattr(models, model_cfg.model_type)(
    #     model_cfg, device=device)

    model = model.to(device)

    print('Model created on device: {}', device)

    # If loading weights from checkpoin
    if model_path:
        model.load_state_dict(torch.load(
            model_path, map_location=torch.device(device)))
        print("Model's checkpoint loaded")

    if parameter_info:
        print('Number of parameters to learn:', count_parameters(model))

    return model


def make_optimiser(model, training_cfg):

    optimiser = optim.SGD(model.parameters(
    ), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay, momentum=training_cfg.momentum)

    # optimiser = optim.AdamW(
    #         model.parameters(), lr=training_cfg.lr, weight_decay=training_cfg.weight_decay)

    return optimiser


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
