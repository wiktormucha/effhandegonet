import cv2 as cv2
# from config import *
from utils.general_utils import heatmaps_to_coordinates
from torchvision import transforms
import torch
import copy as copy
import numpy as np
import os
from utils.general_utils import (
    heatmaps_to_coordinates,
    tensor_to_numpy
)
import pandas as pd
from utils.hand_detector import get_mediapipe_kpts, crop_image


def run_model_on_hands(model: torch.nn.Module, imgs: np.array, device, model_type: str) -> np.array:
    """
    Function to run hand predicotr in egocentric data

    Args:
        model (torch.nn.Module): Prediction model for a single hand.
        imgs (list): Left and right hands segmented images.

    Returns:
        np.array: predicted keypoints
    """

    if model_type == 'custom':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (128, 128), antialias=True),
                transforms.Normalize(
                    mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS)
            ]
        )

        heatmap_out_size = 128
    elif model_type == 'resnet50':
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        heatmap_out_size = 56

    img_trans = []
    for img in imgs:
        img_trans.append(transform(img))

    inpt = torch.stack(img_trans, dim=0).to(device)

    if model_type == 'custom':
        pred_heatmaps = model(inpt)
    elif model_type == 'resnet50':
        pred_heatmaps = model(inpt, data_samples=None)

    pred_keypoints = heatmaps_to_coordinates(
        pred_heatmaps.cpu().detach().numpy(), img_size=heatmap_out_size)

    return pred_keypoints


def preds_to_full_image(predictions: np.array, hands_bb: list, scale: list) -> list:
    """
    Function taking predictions and moving coordinates to full size image with two hands

    Args:
        predictions (np.array): left and right hand prediction
        hands_bb (list): List of bb of hands
        scale (list): Size of imaes used for prediction

    Returns:
        list: Points transformed to input image
    """
    full_scale_preds = []

    for idx, bb in enumerate(hands_bb):

        pts = predictions[idx] * scale[idx]
        pts[:, 0] = pts[:, 0] + bb['x_min']
        pts[:, 1] = pts[:, 1] + bb['y_min']
        full_scale_preds.append(pts)

    return full_scale_preds


def get_egocentric_prediction_mediapipe(img, hand_model, pred_hand_left, pred_hand_right):

    if img.shape[0] != 1:
        raise ValueError(
            "Wrong input shape, tensor requires batch size =1: ", img.shape)
    frame = tensor_to_numpy(img[0])
    result = hand_model.process(frame)
    hand_landmarks = result.multi_hand_landmarks
    hand_class = result.multi_handedness

    # From here continue only if there are predictions in this model:
    if hand_class is None:

        pred_hand_left.append(0)
        pred_hand_right.append(0)

        return pred_hand_left, pred_hand_right, np.zeros((1, 21, 2)), np.zeros((1, 21, 2))

    pose_dict = get_mediapipe_kpts(hand_class, hand_landmarks, mirror=False)

    if "Left" in pose_dict:
        pts_left = pose_dict["Left"].reshape(21, 2) * (1280, 720)
        pred_hand_left.append(1)

    else:
        pts_left = np.zeros((21, 2))
        pred_hand_left.append(0)

    if "Right" in pose_dict:
        pts_right = pose_dict["Right"].reshape(21, 2) * (1280, 720)
        pred_hand_right.append(1)
    else:

        pts_right = np.zeros((21, 2))
        pred_hand_right.append(0)

    pred_left = np.reshape(pts_left, (1, 21, 2))
    pred_right = np.reshape(pts_right, (1, 21, 2))

    return pred_hand_left, pred_hand_right, pred_left, pred_right


def get_egocentric_predictions_effhandegonet(img, hand_model, activation, pred_hand_left, pred_hand_right):
    # activation = torch.nn.Softmax(dim=1)

    if img.shape[0] != 1:
        raise ValueError(
            "Wrong input shape, tensor requires batch size =1: ", img.shape)

    out = hand_model(img)
    #   'left_handness': self.left_hand(flatten),
    #         'right_handness': self.right_hand(flatten),
    #         'left_2D_pose': self.left_pose(features),
    #         'right_2D_pose': self.right_pose(features),
    pred_left_flag = out['left_handness']
    pred_right_flag = out['right_handness']
    pred_left = out['left_2D_pose']
    pred_right = out['right_2D_pose']

    left_batch_mask = torch.argmax(
        activation(pred_left_flag), dim=1)
    right_batch_mask = torch.argmax(
        activation(pred_right_flag), dim=1)

    pred_hand_left.append(left_batch_mask.cpu().detach().tolist()[0])
    pred_hand_right.append(right_batch_mask.cpu().detach().tolist()[0])

    # Convert preds to points
    pred_left_np = heatmaps_to_coordinates(
        pred_left.cpu().detach().numpy(), img_size=pred_left.shape[3]) * (1280, 720)
    pred_right_np = heatmaps_to_coordinates(
        pred_right.cpu().detach().numpy(), img_size=pred_right.shape[3]) * (1280, 720)

    if pred_hand_left[-1] == 0:
        pred_left_np = np.zeros((1, 21, 2))
    if pred_hand_right[-1] == 0:
        pred_right_np = np.zeros((1, 21, 2))

    return pred_left_flag, pred_right_flag, pred_left_np, pred_right_np


def get_egocentric_predictions_effhandnet(img, hand_model, device, yolo_labels_file, factor, pred_hand_left, pred_hand_right):

    if img.shape[0] != 1:
        raise ValueError(
            "Wrong input shape, tensor requires batch size =1: ", img.shape)
    frame = tensor_to_numpy(img[0])

    yolo_labels = read_yolo_labels(yolo_labels=yolo_labels_file, bb_factor=1.1)

    hands_bbs = []
    if 8 in yolo_labels:

        if len(yolo_labels[8]) > 1:
            yolo_labels[8] = [yolo_labels[8][0]]

        left_hand_bb = yolo_labels[8][0]
        hands_bbs.append(left_hand_bb)
    if 9 in yolo_labels:
        if len(yolo_labels[9]) > 1:
            yolo_labels[9] = [yolo_labels[9][0]]
        right_hand_bb = yolo_labels[9][0]
        hands_bbs.append(right_hand_bb)

    # When there is no hands in the image
    if 8 not in yolo_labels and 9 not in yolo_labels:
        pred_hand_left.append(0)
        pred_hand_right.append(0)

        return pred_hand_left, pred_hand_right, np.zeros((1, 21, 2)), np.zeros((1, 21, 2))

    segmented_hands_imgs, bb_dict_list, scale = get_hands_regions(
        img=frame, hands_bbs=hands_bbs, factor=factor)

    pred = run_model_on_hands(
        hand_model, segmented_hands_imgs, device=device, model_type='custom')

    pred = preds_to_full_image(
        predictions=pred, hands_bb=bb_dict_list, scale=scale)

    if len(pred) == 1:
        if 8 in yolo_labels:
            pred_right_np = np.zeros((21, 2))
            pred_left_np = pred[0]
            pred_hand_left.append(1)
            pred_hand_right.append(0)
        else:
            pred_left_np = np.zeros((21, 2))
            pred_right_np = pred[0]
            pred_hand_left.append(0)
            pred_hand_right.append(1)

    elif len(pred) == 2:

        pred_left_np = pred[0].reshape(1, 21, 2)
        pred_right_np = pred[1].reshape(1, 21, 2)
        pred_hand_left.append(1)
        pred_hand_right.append(1)

    return pred_hand_left, pred_hand_right, pred_left_np, pred_right_np


def get_egocentric_predictions_poseresnet50(img, hand_model, device, yolo_labels_file, factor, pred_hand_left, pred_hand_right):

    if img.shape[0] != 1:
        raise ValueError(
            "Wrong input shape, tensor requires batch size =1: ", img.shape)
    frame = tensor_to_numpy(img[0])

    yolo_labels = read_yolo_labels(yolo_labels=yolo_labels_file, bb_factor=1.1)

    hands_bbs = []
    if 8 in yolo_labels:

        if len(yolo_labels[8]) > 1:
            yolo_labels[8] = [yolo_labels[8][0]]

        left_hand_bb = yolo_labels[8][0]
        hands_bbs.append(left_hand_bb)
    if 9 in yolo_labels:
        if len(yolo_labels[9]) > 1:
            yolo_labels[9] = [yolo_labels[9][0]]
        right_hand_bb = yolo_labels[9][0]
        hands_bbs.append(right_hand_bb)

    # When there is no hands in the image
    if 8 not in yolo_labels and 9 not in yolo_labels:
        pred_hand_left.append(0)
        pred_hand_right.append(0)

        return pred_hand_left, pred_hand_right, np.zeros((1, 21, 2)), np.zeros((1, 21, 2))

    segmented_hands_imgs, bb_dict_list, scale = get_hands_regions(
        img=frame, hands_bbs=hands_bbs, factor=factor)

    pred = run_model_on_hands(
        hand_model, segmented_hands_imgs, device=device, model_type='resnet50')

    pred = preds_to_full_image(
        predictions=pred, hands_bb=bb_dict_list, scale=scale)

    if len(pred) == 1:
        if 8 in yolo_labels:
            pred_right_np = np.zeros((21, 2))
            pred_left_np = pred[0]
            pred_hand_left.append(1)
            pred_hand_right.append(0)
        else:
            pred_left_np = np.zeros((21, 2))
            pred_right_np = pred[0]
            pred_hand_left.append(0)
            pred_hand_right.append(1)

    elif len(pred) == 2:

        pred_left_np = pred[0].reshape(1, 21, 2)
        pred_right_np = pred[1].reshape(1, 21, 2)
        pred_hand_left.append(1)
        pred_hand_right.append(1)

    return pred_hand_left, pred_hand_right, pred_left_np, pred_right_np


def get_hands_regions(img, hands_bbs, factor):
    """
    Extracts regions of interest (hands) from an egocentric image based on bounding boxes.

    Args:
        img (numpy.ndarray): The egocentric image.
        hands_bbs (list): A list of bounding boxes for the hands.
        factor (int): The factor by which to expand the bounding box.

    Returns:
        tuple: A tuple containing:
            - A list of numpy arrays, each containing a segmented hand image.
            - A list of dictionaries, each containing the coordinates of the bounding box for the corresponding hand.
            - A list of tuples, each containing the dimensions of the corresponding segmented hand image.
    """
    segmented_hands_imgs = []
    bb_dict_list = []
    for bb in hands_bbs:
        bb_dict = {
            'y_min': int((bb.yc - (bb.height/2)) * 720) - factor,
            'y_max': int((bb.yc + (bb.height/2)) * 720) + factor,
            'x_min': int((bb.xc - (bb.width/2)) * 1280) - factor,
            'x_max': int((bb.xc + (bb.width/2)) * 1280) + factor,
        }
        bb_dict_list.append(bb_dict)
        segmented_hands_imgs.append(crop_image(np.array(img), bb_dict))

    if len(segmented_hands_imgs) == 2:
        scale = [(segmented_hands_imgs[0].shape[1], segmented_hands_imgs[0].shape[0]),
                 (segmented_hands_imgs[1].shape[1],
                 segmented_hands_imgs[1].shape[0])
                 ]
    elif len(segmented_hands_imgs) == 1:
        scale = [(segmented_hands_imgs[0].shape[1], segmented_hands_imgs[0].shape[0])
                 ]

    return segmented_hands_imgs, bb_dict_list, scale


def save_predictions(pred_left_np, pred_right_np, pth_to_save, folder_to_save, filename_to_save):
    """
    Saves the predictions of left and right hands to a file.

    Args:
    pred_left_np (numpy.ndarray): The predicted values for the left hand.
    pred_right_np (numpy.ndarray): The predicted values for the right hand.
    pth_to_save (str): The path to save the file.
    folder_to_save (str): The folder to save the file.
    filename_to_save (str): The name of the file to save.

    Returns:
    None
    """
    merged = np.stack([pred_left_np, pred_right_np])
    dirname = os.path.join(pth_to_save, folder_to_save)
    if os.path.isdir(dirname) == False:
        os.mkdir(dirname)

    pred_to_file = merged.reshape(42, 2)
    np.savetxt(os.path.join(pth_to_save, folder_to_save,
               filename_to_save), pred_to_file)


class YoloLabel:
    """
    A class representing a YOLO label for an object detection task.

    Attributes:
    -----------
    label : int
        The label of the object.
    xc : float
        The x-coordinate of the center of the object.
    yc : float
        The y-coordinate of the center of the object.
    width : float
        The width of the object.
    height : float
        The height of the object.
    """
    label: int
    xc: float
    yc: float
    width: float
    height: float


def read_yolo_labels(yolo_labels: pd.DataFrame, bb_factor=1):
    """
    Reads YOLO labels from a pandas DataFrame and returns a dictionary of YoloLabel objects.

    Args:
        yolo_labels (pd.DataFrame): A pandas DataFrame containing YOLO labels.
        bb_factor (float): A scaling factor for bounding box dimensions. Default is 1.

    Returns:
        dict: A dictionary of YoloLabel objects, where the keys are the label integers and the values are lists of YoloLabel objects.
    """
    ret_dict = {}
    for _, row in yolo_labels.iterrows():
        temp_obj = YoloLabel()
        temp_obj.label = int(row[0])
        temp_obj.xc = float(row[1])
        temp_obj.yc = float(row[2])
        temp_obj.width = float(row[3]) * bb_factor
        temp_obj.height = float(row[4]) * bb_factor

        if temp_obj.label not in ret_dict:
            ret_dict[temp_obj.label] = [temp_obj]
        else:
            ret_dict[temp_obj.label].append(temp_obj)

    return ret_dict
