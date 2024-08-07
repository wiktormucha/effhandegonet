from utils.general_utils import heatmaps_to_coordinates
from utils.metrics import keypoint_pck_accuracy, keypoint_epe, keypoint_auc
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
# sys.path.append("../")

from constants import COLORMAP, MODEL_IMG_SIZE, RAW_IMG_SIZE


def get_bb_w_and_h(gt_keypoints: np.array, bb_factor: int = 1) -> np.array:
    """
    Returns width and height of bounding box

    Args:
        gt_keypoints (np.array): GT keypoints
        bb_factor (int, optional): Bounding box margin factor. Defaults to 1.

    Returns:
        np.array: (batch_size, (bb_width, bb_height))
    """

    normalize = np.zeros((gt_keypoints.shape[0], 2))

    # normalize = get_bb_batch(true_keypoints)
    for idx, img in enumerate(gt_keypoints):

        xmax, ymax = img.max(axis=0)
        xmin, ymin = img.min(axis=0)

        width = xmax - xmin
        height = ymax - ymin
        normalize[idx][0] = width * bb_factor
        normalize[idx][1] = height * bb_factor

    return normalize


def batch_epe_calculation(pred_keypoints: np.array, true_keypoints: np.array, batch_mask: np.array = None, mask: np.array = None, input_img_size=MODEL_IMG_SIZE) -> float:
    """
    Calculates End Point Error (EPE) for a batch in pixels.

    Args:
        pred_keypoints (np.array): Predicted keypoints.
        true_keypoints (np.array): GT keypoints
        mask (np.array, optional): Mask with information which points to hide from calculation (0 skipped; 1 used). Defaults to None.

    Returns:
        float: EPE
    """

    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21), dtype=int)

    epe = keypoint_epe(pred_keypoints,
                       true_keypoints, mask, batch_mask)

    return epe


def batch_auc_calculation(pred_keypoints: np.array, true_keypoints: np.array, num_step: int = 20, mask: np.array = None, normalize: np.array = None):
    """
    Calculates Area Under the Curve for a batch.

    Args:
        pred_keypoints (np.array): Predicted keypoints.
        true_keypoints (np.array): GT keypoints
        num_step (int, optional): How dense is treshold. Defaults to 20.
        mask (np.array, optional): Mask with information which points to hide from calculation (0 skipped; 1 used). Defaults to None.
        normalize (np.array, optional): Width and height to normalise. Defaults to None.

    Returns:
        float: AUC
    """

    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21), dtype=int)

    if normalize == None:
        normalize = get_bb_w_and_h(true_keypoints)

    auc = keypoint_auc(pred=pred_keypoints, gt=true_keypoints,
                       mask=mask, normalize=normalize, num_step=num_step)

    return auc


def batch_pck_calculation(pred_keypoints: np.array, true_keypoints: np.array, treshold: float = 0.2, mask: np.array = None, normalize: np.array = None) -> float:
    """
    Calculates PCK for a batch.

    Args:
        pred_keypoints (np.array): Predicted keypoints.
        true_keypoints (np.array): GT keypoints
        treshold (float, optional): PCK treshold. Defaults to 0.2.
        mask (np.array, optional): Mask with information which points to hide from calculation (0 skipped; 1 used). Defaults to None.
        normalize (np.array, optional): _description_. Defaults to None.

    Returns:
        float: PCK
    """

    if mask == None:
        mask = np.ones((true_keypoints.shape[0], 21), dtype=int)

    if normalize == None:
        normalize = get_bb_w_and_h(true_keypoints)
    # print(normalize[:,])
    _, avg_acc, _ = keypoint_pck_accuracy(
        pred=pred_keypoints, gt=true_keypoints, mask=mask, thr=treshold, normalize=normalize)

    return avg_acc


def show_batch_predictions(batch_data: dict, model: torch.nn.Module) -> None:
    """
    Visualizes image, image with actual keypoints and image with predicted keypoints. Finger colors are in COLORMAP.

    Args:
        batch_data (dict): Batch from dataloader
        model (torch.nn.Module): Trained model
    """

    inputs = batch_data["image"]
    true_keypoints = batch_data["keypoints"].numpy()
    batch_size = true_keypoints.shape[0]
    pred_heatmaps = model(inputs)
    pred_heatmaps = pred_heatmaps.detach().numpy()
    pred_keypoints = heatmaps_to_coordinates(pred_heatmaps)
    images = batch_data["image_raw"].numpy()
    images = np.moveaxis(images, 1, -1)

    plt.figure(figsize=[12, 4 * batch_size])
    for i in range(batch_size):
        image = images[i]
        true_keypoints_img = true_keypoints[i] * RAW_IMG_SIZE
        pred_keypoints_img = pred_keypoints[i] * RAW_IMG_SIZE

        plt.subplot(batch_size, 3, i * 3 + 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis("off")

        plt.subplot(batch_size, 3, i * 3 + 2)
        plt.imshow(image)
        plt.scatter(true_keypoints_img[:, 0],
                    true_keypoints_img[:, 1], c="k", alpha=0.5)
        for finger, params in COLORMAP.items():
            plt.plot(
                true_keypoints_img[params["ids"], 0],
                true_keypoints_img[params["ids"], 1],
                params["color"],
            )
        plt.title("True Keypoints")
        plt.axis("off")

        plt.subplot(batch_size, 3, i * 3 + 3)
        plt.imshow(image)
        plt.scatter(pred_keypoints_img[:, 0],
                    pred_keypoints_img[:, 1], c="k", alpha=0.5)
        for finger, params in COLORMAP.items():
            plt.plot(
                pred_keypoints_img[params["ids"], 0],
                pred_keypoints_img[params["ids"], 1],
                params["color"],
            )
        plt.title("Predicted Keypoints")
        plt.axis("off")
    plt.tight_layout()
