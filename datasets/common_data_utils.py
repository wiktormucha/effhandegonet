import numpy as np
import torch
import random
import pytorchvideo.transforms.functional as T
import albumentations as A


def sample2(input_frames: list, no_of_outframes, sampling_type: str):

    if len(input_frames) >= no_of_outframes:
        indxs_to_sample = np.arange(len(input_frames))

        if sampling_type == "uniform":

            indxs_to_sample = T.uniform_temporal_subsample(
                torch.tensor(indxs_to_sample), no_of_outframes, 0).tolist()

        elif sampling_type == "random":

            # randomly susample the frames to match the no_of_outframes
            indxs_to_sample = list(range(len(input_frames)))
            indxs_to_sample = random.sample(
                indxs_to_sample, no_of_outframes)
            indxs_to_sample.sort()

    else:
        indxs_to_sample = np.trunc(
            np.arange(0, no_of_outframes) * len(input_frames)/no_of_outframes).astype(int)

    return indxs_to_sample


def albumentation_to_sequence(image, keypoints, albumentations, transform_replay, frame_idx):

    if frame_idx == 0:
        transformed = albumentations(
            image=image, keypoints=keypoints)

    # For rest offrames apply same
    else:
        transformed = A.ReplayCompose.replay(
            transform_replay, image=image, keypoints=keypoints)

    return {
        'image': transformed['image'],
        'keypoints': transformed['keypoints'],
        'replay': transformed["replay"]
    }
