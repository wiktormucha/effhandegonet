from constants import RAW_IMG_SIZE, MODEL_IMG_SIZE
import numpy as np
import os
from PIL import Image
import json
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from utils.general_utils import vector_to_heatmaps, project_points_3D_to_2D
import albumentations as A
from constants import ALBUMENTATION_VAL, ALBUMENTATION_TRAIN
from torch.utils.data import DataLoader

ONLY_GREENSCREEN_IMAGES = False

class FreiHAND_albu(Dataset):
    """
    Class to load FreiHAND dataset. This dataset class is using only training part.
    It devides the dataste into three subsets in proportion 80/10/10. It is using albumentations for data augumentation.

    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config: dict, albumetations: A.Compose, set_type: str = "train"):
        """
        Initialisation of the dataset

        Args:
            config (dict): Config dictionary with needed data for training.
            albumetations (A.Compose): Albumentation transforms for augumentation.
            set_type (str, optional): Set tye type "train" or "val" or other for test. Defaults to "train".
        """

        self.device = config["device"]
        self.image_dir = os.path.join(config["data_dir"], "training/rgb")
        self.image_names = np.sort(os.listdir(self.image_dir))

        fn_K_matrix = os.path.join(config["data_dir"], "training_K.json")
        with open(fn_K_matrix, "r") as f:
            K_matrix_temp = np.array(json.load(f))

        fn_anno = os.path.join(config["data_dir"], "training_xyz.json")
        with open(fn_anno, "r") as f:
            anno_temp = np.array(json.load(f))

        self.K_matrix = np.concatenate(
            (K_matrix_temp, K_matrix_temp, K_matrix_temp, K_matrix_temp), axis=0)
        self.anno = np.concatenate(
            (anno_temp, anno_temp, anno_temp, anno_temp), axis=0)

        assert len(self.K_matrix) == len(self.anno) == len(self.image_names)

        # Case to switch between smaller (only greenscreen images) and bigger dataset
        if ONLY_GREENSCREEN_IMAGES:
            train_end = 26048
            val_end = 29304
        else:
            train_end = 104192
            val_end = 117216

        if set_type == "train":
            n_start = 0
            n_end = train_end
            # n_end = 26048
        elif set_type == "val":
            n_start = train_end
            n_end = val_end
        else:
            n_start = val_end
            n_end = len(self.anno)

        self.image_names = self.image_names[n_start:n_end]
        self.K_matrix = self.K_matrix[n_start:n_end]
        self.anno = self.anno[n_start:n_end]

        print(f'Number of {set_type} samples: {len(self.image_names)}')

        self.image_raw_transform = transforms.ToTensor()
        self.albumetations = albumetations

    def __len__(self):
        """
        Return length of the dataset

        Returns:
            _type_: Dataset length
        """
        return len(self.anno)

    def __getitem__(self, idx):
        """
        Returns item from dataset

        Args:
            idx (int): Given index in dataset

        Returns:
            dict: Dictionary with item containing:
                    - image
                    - keypoints
                    - heatmaps
                    - image name
                    - raw image

        """

        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))

        keypoints = project_points_3D_to_2D(self.anno[idx], self.K_matrix[idx])

        transformed = self.albumetations(
            image=np.asarray(image_raw), keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = np.asarray(transformed['keypoints'])

        heatmaps = vector_to_heatmaps(
            transformed_keypoints, scale_factor=1, out_size=MODEL_IMG_SIZE)

        # Convert to tensors
        image_raw = self.image_raw_transform(image_raw)

        keypoints = torch.from_numpy(transformed_keypoints / MODEL_IMG_SIZE)

        return {
            "img": transformed_image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "img_name": image_name,
            "img_raw": image_raw,
        }


class FreiHAND_evaluation(Dataset):
    """
    Class to load FreiHAND evaluation subset.

    Link to dataset:
    https://lmb.informatik.uni-freiburg.de/resources/datasets/FreihandDataset.en.html
    """

    def __init__(self, config: dict, albumetations: A.Compose):
        """
        Initialisation of the dataset

        Args:
            config (dict): Config dictionary with needed data for training.
            albumetations (A.Compose): Albumentation transforms for augumentation.
        """

        self.image_dir = os.path.join(
            (config["data_dir"] + '_evaluation'), "evaluation/rgb")

        fn_K_matrix = os.path.join(
            (config["data_dir"] + '_evaluation'), "evaluation_K.json")
        with open(fn_K_matrix, "r") as f:
            self.K_matrix = np.array(json.load(f))

        fn_anno = os.path.join(
            config["data_dir"] + '_evaluation', "evaluation_xyz.json")
        with open(fn_anno, "r") as f:
            self.anno = np.array(json.load(f))

        self.device = config["device"]
        self.image_names = np.sort(os.listdir(self.image_dir))

        self.image_raw_transform = transforms.ToTensor()
        self.albumetations = albumetations

    def __len__(self):
        """
        Return length of the dataset

        Returns:
            _type_: Dataset length
        """
        return len(self.anno)

    def __getitem__(self, idx):
        """
        Returns item from dataset

        Args:
            idx (int): Given index in dataset

        Returns:
            dict: Dictionary with item containing:
                    - image
                    - keypoints
                    - heatmaps
                    - image name
                    - raw image
        """

        image_name = self.image_names[idx]
        image_raw = Image.open(os.path.join(self.image_dir, image_name))

        keypoints = project_points_3D_to_2D(self.anno[idx], self.K_matrix[idx])

        transformed = self.albumetations(
            image=np.asarray(image_raw), keypoints=keypoints)
        transformed_image = transformed['image']
        transformed_keypoints = np.asarray(transformed['keypoints'])

        heatmaps = vector_to_heatmaps(
            transformed_keypoints, scale_factor=1, out_size=MODEL_IMG_SIZE)

        # Convert to tensors
        image_raw = self.image_raw_transform(image_raw)
        keypoints = torch.from_numpy(transformed_keypoints / MODEL_IMG_SIZE)

        return {
            "img": transformed_image,
            "keypoints": keypoints,
            "heatmaps": heatmaps,
            "img_name": image_name,
            "img_raw": image_raw,
        }


def get_FreiHAND_dataloaders(config: dict):

    train_dataset = FreiHAND_albu(
        config=config, set_type="train", albumetations=ALBUMENTATION_TRAIN)

    test_dataset = FreiHAND_albu(
        config=config, set_type="test", albumetations=ALBUMENTATION_VAL)

    val_dataset = FreiHAND_albu(
        config=config, set_type="val", albumetations=ALBUMENTATION_VAL)

    test_final = FreiHAND_evaluation(
        config=config, albumetations=ALBUMENTATION_VAL)

    dataloader_train = DataLoader(
        train_dataset,
        config["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=config['num_workers'],
    )
    dataloader_test = DataLoader(
        test_dataset,
        config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers'],
    )

    dataloader_val = DataLoader(
        val_dataset,
        config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers'],
    )

    dataloader_test_final = DataLoader(
        test_final,
        config["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=config['num_workers'],
    )

    return {
        "train": dataloader_train,
        "val": dataloader_val,
        "test": dataloader_test,
        "test_final": dataloader_test_final
    }
