import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from utils.general_utils import project_points_3D_to_2D

from constants import BB_FACTOR


def pil_to_cv(pil_image: Image) -> cv2:
    """
    Converts PIL image to OpenCV

    Args:
        pil_image (Image): Input in PIL format
    Returns:
        cv2: Output in cv2/np.array format
    """
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    return open_cv_image


def cv_to_pil(cv_image: cv2) -> Image:
    """
    Converts OpenCV image to PIL format

    Args:
        cv_image (cv2): Input in cv2/np.array 

    Returns:
        Image: Output in PIL format
    """
    # Convert BGR to RGB
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image)
    return pil_image


def crop_image(image: cv2, bbox: dict) -> cv2:
    """
    Segments givven bounding box from input image.

    Args:
        image (cv2): Input image
        bbox (dict): Bounding box to segment image

    Returns:
        cv2: Segmented image
    """

    cropped_image = image[bbox['y_min']                          :bbox['y_max'], bbox['x_min']:bbox['x_max']]

    h, w, c = cropped_image.shape

    # In case when hand is to low i nthe image and cropped size is not square we reduce size of square
    if h < w:

        padding = np.zeros((w, w, 3), dtype=np.uint8)
        # cropped_image = cropped_image[:,:h]
        padding[:h, :w] = cropped_image

        return padding
        # cropped_image += padding

    return cropped_image


def gt_to_segments(gt: np.array, bbox: dict) -> np.array:
    """
    Function applyes segmentic keypoints from main image using given boundigboxes
    Args:
        gt (np.array): Ground truth keypoints
        bbox (dict): Bounding box

    Returns:
        np.array: Transformed GT keypoints
    """
    '''
    
    '''
    gt[:, 0] = gt[:, 0] - bbox['x_min']
    gt[:, 1] = gt[:, 1] - bbox['y_min']

    return gt


def compress_gt_pts(gt: np.array, img_dimm: int) -> np.array:
    """
    Normalises GT points to <0,1>

    Args:
        gt (np.array): Ground truth keypoints
        img_dimm (int): Image width/height
    Returns:
        np.array: Normalised GT keypoints
    """

    compressed_gt = gt / img_dimm

    return compressed_gt


def get_bb(hand_info: dict, hand_landmarks: dict, w: int, h: int, factor: int = BB_FACTOR, mirror: bool = False) -> dict:
    """
    Function to get bounding boxes from mediapipe predictions.

    Args:
        hand_info (dict): _description_
        hand_landmarks (dict): _description_
        w (int): Width
        h (int): Height
        factor (int, optional): Factor adding margin to boundig box. Defaults to BB_FACTOR.
        mirror (bool, optional): Flag to flip left with right hand. Defaults to False.

    Returns:
        dict: _description_
    """

    x_max = 0
    y_max = 0
    x_min = w
    y_min = h

    hand_dict = {'index': hand_info.classification[0].index,
                 'label': hand_info.classification[0].label

                 }

    if mirror == True:

        if hand_dict['label'] == 'Right':
            hand_dict['label'] = 'Left'
        else:
            hand_dict['label'] = 'Right'

    for lm in hand_landmarks.landmark:
        x, y = int(lm.x * w), int(lm.y * h)
        if x > x_max:
            x_max = x
        if x < x_min:
            x_min = x
        if y > y_max:
            y_max = y
        if y < y_min:
            y_min = y

    # Here BBs are increased by a given factor
    hand_dict['x_min'] = x_min - factor
    hand_dict['x_max'] = x_max + factor

    hand_dict['y_min'] = y_min - factor
    hand_dict['y_max'] = y_max + factor

    bb_w = hand_dict['x_max'] - hand_dict['x_min']
    bb_h = hand_dict['y_max'] - hand_dict['y_min']

    # Here the BBs are modified to be square shape
    diff = bb_w - bb_h

    # if w > h
    if diff > 0:
        hand_dict['y_min'] -= int(diff/2)
        if diff % 2 == 0:
            hand_dict['y_max'] += int(diff/2)
        else:
            hand_dict['y_max'] += int(diff/2) + 1
    # if h > w
    elif diff < 0:
        diff = abs(diff)
        hand_dict['x_min'] -= int(diff/2)
        if diff % 2 == 0:
            hand_dict['x_max'] += int(diff/2)
        else:
            hand_dict['x_max'] += int(diff/2) + 1

    return hand_dict


def get_hands_bb(img: np.array, hand_landmarks, hand_class) -> list:
    """
    Function detecitng hands and returning bounding boxes

    Args:
        img (np.array): Input image
        hands (_type_): Hand model

    Returns:
        list: List of bounding boxes
    """
    hand_bb_list = []
    h, w, c = img.shape

    if hand_landmarks:
        for handLMs, hand in zip(hand_landmarks, hand_class):
            hand_bb = get_bb(hand, handLMs, w=w, h=h, mirror=True)
            hand_bb_list.append(hand_bb)
            # cv2.rectangle(frame, (hand_bb['x_min'], hand_bb['y_min']), (hand_bb['x_max'], hand_bb['y_max']), (0, 255, 0), 2)
            # cv2.putText(frame, hand_bb['label'], (hand_bb['x_min'], hand_bb['y_min']-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return hand_bb_list


def get_hands_img(img: np.array, gt_pts: np.array, hand_landmarks, hand_class, cam_instr: np.array) -> dict:
    """
    This function segments hands from main image and return dicitonary with image hand and label (left/right)

    Args:
        img (np.array): Input image with two hands
        gt_pts (np.array): GT points
        hand_model (_type_): Hand model to predict hands in the image
        cam_instr (np.array): Camera intrinsic to transform image

    Returns:
        dict:   -'hands_seg': Images of segmented hands
                -'gt': gt, GT keypoints
                -'hand_type': Label Left/Right
                -'hands_bb': List of bounding boxes
    """

    img = pil_to_cv(img)

    hand_bb_list = get_hands_bb(img, hand_landmarks, hand_class)

    gt_pts = np.split(gt_pts, [1, 64, 65, 128])

    hand1 = np.reshape(gt_pts[1], (21, 3))

    hand2 = np.reshape(gt_pts[3], (21, 3))
    ptsL = project_points_3D_to_2D(hand1, cam_instr)
    ptsP = project_points_3D_to_2D(hand2, cam_instr)

    # For each hand
    hands_seg = []
    gt = []
    hand_type = []
    hand_pts = [ptsL, ptsP]

    if len(hand_bb_list) > 1:

        if hand_bb_list[1]['x_max'] < hand_bb_list[0]['x_max']:
            hand_bb_list.reverse()

    for i, hand_bb in enumerate(hand_bb_list):

        hand = crop_image(img, hand_bb)
        pts_segm = gt_to_segments(hand_pts[i], hand_bb)
        compress_pts = compress_gt_pts(pts_segm, hand.shape[0])
        hand = cv_to_pil(hand)
        # Convert back to PIL
        hands_seg.append(hand)
        gt.append(compress_pts)
        hand_type.append(hand_bb['label'])

    return {
        'hands_seg': hands_seg,
        'gt': gt,
        'hand_type': hand_type,
        'hands_bb': hand_bb_list
    }


def get_mediapipe_kpts(hand_class, hand_landmarks, mirror: bool = True) -> dict:

    # mirror = True
    pose_dict = {}

    # Todo assert type

    # Loop over detected hands
    if isinstance(hand_class, type(None)):
        print("Hand class: ", hand_class)
    if isinstance(hand_landmarks, type(None)):
        print("hand_landmarks: ", hand_landmarks)

    for hand, hand_landmark in zip(hand_class, hand_landmarks):
        # print(hand.classification[0].label)
        temp_key = hand.classification[0].label
        if mirror == True:
            if temp_key == "Left":
                temp_key = "Right"
            else:
                temp_key = "Left"
        temp_list = []
        # Loop over landarm in each hand
        for lm in hand_landmark.landmark:
            # print(lm)
            temp_list.append(lm.x)
            temp_list.append(lm.y)
        pose_dict[temp_key] = np.array(temp_list)

    return pose_dict
