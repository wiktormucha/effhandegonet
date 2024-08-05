import warnings
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from utils.general_utils import project_points_3D_to_2D, vector_to_heatmaps
import cv2 as cv2
import albumentations as A
from torch.utils.data import DataLoader

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


CAM_INTRS = np.array([[636.6593017578125, 0.00000000e+00, 635.283881879317],
                      [0.00000000e+00, 636.251953125, 366.8740353496978],
                      [0.00000000e+00, 0.00000000e+00, 1.0]])


class H2O_Dataset_hand_train(Dataset):
    """
    H2O Dataset for egocentric hand tests only
    """

    def __init__(self, config, type="train", albu_transform=None):
        """
        Initialisation of the dataset

        Args:
            config (dict): Config dictionary with needed data for training.
        """
        self.subset_type = type
        if type == "train":
            path_list = os.path.join(config.path, "train_pose.txt")
            max_samples = config.max_train_samples
        elif type == "val":
            path_list = os.path.join(config.path, "val_pose.txt")
            max_samples = config.max_val_samples
        elif type == "test":
            path_list = os.path.join(config.path, "test_pose.txt")
            max_samples = 30000

        self.imgs_paths = pd.read_csv(path_list, header=None).to_numpy()
        self.imgs_paths = self.imgs_paths[0:max_samples]
        self.img_size = (config.img_size[0], config.img_size[1])
        self.albu_transform = albu_transform
        self.data_dimension = "2D"

        # self.imgs_paths = [os.path.join(
        #     "/data/wmucha/datasets/h2o_ego/h2o_ego", path[0]) for path in self.imgs_paths[0]]
        for i in range(len(self.imgs_paths)):
            temp = os.path.join(
                "/data/wmucha/datasets/h2o_ego/h2o_ego", self.imgs_paths[i][0])

            self.imgs_paths[i] = temp

        #     if not os.path.exists(self.imgs_paths[i][0]):
        #         raise FileNotFoundError(self.imgs_paths[i])

        hand_pose_pths = []
        hand_pose_list = []
        hand_pose3D_list = []
        left_hand_flag_temp_list = []
        right_hand_flag_temp_list = []

        # Assuming CAM_INTRS is a constant that doesn't change per iteration
        cam_instr = CAM_INTRS

        # Transform paths using list comprehension
        temp_hand_poses = [path[0].replace("rgb", "hand_pose").replace(
            ".png", ".txt") for path in self.imgs_paths]

        # Batch check for file existence
        missing_files = [
            path for path in temp_hand_poses if not os.path.exists(path)]
        if missing_files:
            # Raise the first missing file found
            raise FileNotFoundError(missing_files[0])

        # Process each hand pose
        for temp_hand_pose in temp_hand_poses:
            # cam_instr_pth = os.path.join(
            #     temp_hand_pose[0:60], "cam_intrinsics.txt")

            # Read hand pose
            hand_pose, hand_pose3D, left_hand_flag_temp, right_hand_flag_temp = self.__read_hand_pose(
                temp_hand_pose, cam_instr)

            # Append results to lists
            hand_pose_pths.append(temp_hand_pose)
            hand_pose_list.append(hand_pose)
            hand_pose3D_list.append(hand_pose3D)
            left_hand_flag_temp_list.append(left_hand_flag_temp)
            right_hand_flag_temp_list.append(right_hand_flag_temp)

        # obj_pose_pths = []
        # Transform paths using list comprehension
        temp_obj_poses = [path[0].replace("rgb", "obj_pose").replace(
            ".png", ".txt") for path in self.imgs_paths]

        # Batch check for file existence
        missing_files = [
            path for path in temp_obj_poses if not os.path.exists(path)]
        if missing_files:
            # Raise the first missing file found
            raise FileNotFoundError(missing_files[0])

        # Assignments to class attributes
        # self.obj_pose_pths = np.array(temp_obj_poses)
        # self.hand_pose_pths = np.array(hand_pose_pths)
        self.hand_pose2D = np.array(hand_pose_list)
        # self.hand_pose3D = np.array(hand_pose3D_list)
        self.left_hand_flag = np.array(left_hand_flag_temp_list)
        self.right_hand_flag = np.array(right_hand_flag_temp_list)

    def __len__(self) -> int:
        """
        Return length of the dataset

        Returns:
            int: Dataset length
        """
        return len(self.imgs_paths)

    def __read_hand_pose(self, path, cam_instr):
        hand_pose = np.genfromtxt(path)
        gt_pts = np.split(hand_pose, [1, 64, 65, 128])
        left_hand = np.reshape(gt_pts[1], (21, 3))
        right_hand = np.reshape(gt_pts[3], (21, 3))
        merged3d = np.stack([left_hand, right_hand])

        # Put to 2D
        if self.data_dimension == "2D":
            left_hand = project_points_3D_to_2D(left_hand, cam_instr)
            right_hand = project_points_3D_to_2D(right_hand, cam_instr)

        merged = np.stack([left_hand, right_hand])
        if self.data_dimension == "2D":
            merged = merged.reshape(42, 2)
        else:
            merged = merged.reshape(42, 3)

        left_hand_flag = gt_pts[0]
        right_hand_flag = gt_pts[2]

        return merged, merged3d, left_hand_flag, right_hand_flag

    def __getitem__(self, idx: int) -> dict:
        """
        Returns item from dataset

        Args:
            idx (int): Given index in dataset

        Returns:
            dict: Dictionary with item containing:
                    - Raw image
                    - Camera position
                    - Hand position
                    - Hand position mano
                    - Camera intrinsic
                    - Image name
        """
        img_path = self.imgs_paths[idx][0]
        img = np.asarray(Image.open(img_path))

        hand_pose = self.hand_pose2D[idx]
        left_hand_flag_temp = self.left_hand_flag[idx]
        right_hand_flag_temp = self.right_hand_flag[idx]

        if self.albu_transform:
            transformed = self.albu_transform(
                image=img, keypoints=hand_pose)
            img = transformed['image']
            keypoints = np.array(transformed['keypoints'])

            if 'replay' in transformed:
                applied_aug = transformed['replay']

                horizontal_flip_flag = applied_aug["transforms"][0]["transforms"][0]["applied"]
                vertical_flip_flag = applied_aug["transforms"][0]["transforms"][1]["applied"]
            else:
                horizontal_flip_flag = False
                vertical_flip_flag = False

        else:
            img = img
            keypoints = hand_pose

        if horizontal_flip_flag or vertical_flip_flag:
            ptsL = keypoints[21:42]
            ptsR = keypoints[0:21]
            right_hand_flag = left_hand_flag_temp
            left_hand_flag = right_hand_flag_temp

        else:
            ptsL = keypoints[0:21]
            ptsR = keypoints[21:42]
            left_hand_flag = left_hand_flag_temp
            right_hand_flag = right_hand_flag_temp

        ptsL = ptsL * left_hand_flag
        ptsR = ptsR * right_hand_flag

        heatmap_dim = 128

        if int(left_hand_flag) == 1:
            heatmap_left = vector_to_heatmaps(
                ptsL/self.img_size, scale_factor=heatmap_dim, out_size=heatmap_dim)

        else:
            heatmap_left = np.zeros([21, heatmap_dim, heatmap_dim])

        if int(right_hand_flag) == 1:
            heatmap_right = vector_to_heatmaps(
                ptsR/self.img_size, scale_factor=heatmap_dim, out_size=heatmap_dim)
        else:
            heatmap_right = np.zeros([21, heatmap_dim, heatmap_dim])

        return {
            "img_path": img_path,
            "img": img,
            "left_hand_flag": int(left_hand_flag),
            "right_hand_flag": int(right_hand_flag),
            "keypoints_left": ptsL / self.img_size,
            "keypoints_right": ptsR / self.img_size,
            'heatmap_left':  heatmap_left,
            'heatmap_right':  heatmap_right,
        }


def get_H2O_dataloader(config):

    albumentations_train = A.ReplayCompose(
        [
            A.OneOf([
                A.HorizontalFlip(always_apply=True, p=0.33),
                A.VerticalFlip(always_apply=True, p=0.33),
                A.Compose([
                    A.HorizontalFlip(always_apply=True, p=1.0),
                    A.VerticalFlip(always_apply=True, p=1.0),
                ], p=0.33),
                A.Rotate(always_apply=True, p=0.5, limit=(-180, 180), interpolation=0, border_mode=4,
                         value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
            ], p=0.3),
            A.OneOf([
                A.Resize(
                    config.Data.img_size[0], config.Data.img_size[1]),
                A.RandomResizedCrop(always_apply=True, p=0.5, height=config.Data.img_size[0],
                                    width=config.Data.img_size[1], scale=(0.8, 1.0), ratio=(1, 1), interpolation=0),
                A.Compose([
                    A.Rotate(always_apply=True, p=0.5, limit=(-30, 30), interpolation=0, border_mode=4,
                             value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
                    A.RandomResizedCrop(always_apply=True, p=0.5, height=config.Data.img_size[0],
                                        width=config.Data.img_size[1], scale=(0.7, 1.0), ratio=(1, 1), interpolation=0),
                ]),
                A.Compose([
                    A.Rotate(always_apply=True, p=0.5, limit=(-30, 30), interpolation=0, border_mode=4,
                             value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
                    A.Resize(
                        config.Data.img_size[0], config.Data.img_size[1]),
                ]),
            ], p=1.0),
            A.OneOf([
                A.OneOf([
                    A.PadIfNeeded(always_apply=True, p=1.0, min_height=2000, min_width=2000, pad_height_divisor=None,
                        pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None),
                    A.PadIfNeeded(always_apply=True, p=1.0, min_height=3000, min_width=3000, pad_height_divisor=None,
                        pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None),
                ]),
                A.OneOf([
                    A.Resize(
                        config.Data.img_size[0], config.Data.img_size[1]),
                    A.PadIfNeeded(always_apply=True, p=1.0, min_height=700, min_width=700, pad_height_divisor=None,
                                  pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None),
                    A.PadIfNeeded(always_apply=True, p=1.0, min_height=800, min_width=800, pad_height_divisor=None,
                                  pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None),
                    A.PadIfNeeded(always_apply=True, p=1.0, min_height=1000, min_width=1000, pad_height_divisor=None,
                                  pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None),
                    A.PadIfNeeded(always_apply=True, p=1.0, min_height=1200, min_width=1200, pad_height_divisor=None,
                                  pad_width_divisor=None, border_mode=0, value=(0, 0, 0), mask_value=None),
                ])
            ], p=0.5),
            A.Rotate(always_apply=False, p=0.5, limit=(-180, 180), interpolation=0, border_mode=4,
                     value=(0, 0, 0), mask_value=None, rotate_method='largest_box', crop_border=False),
            A.Resize(
                config.Data.img_size[0], config.Data.img_size[1]),
            A.Normalize(mean=config.Data.norm_mean,
                        std=config.Data.norm_std),
            ToTensorV2()
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    albumentation_val = A.Compose(
        [
            A.Resize(config.Data.img_size[0],
                     config.Data.img_size[1]),
            A.Normalize(mean=config.Data.norm_mean,
                        std=config.Data.norm_std),
            ToTensorV2()

        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
    )

    dataset_train = H2O_Dataset_hand_train(
        config=config.Data, type="train", albu_transform=albumentations_train)
    dataset_val = H2O_Dataset_hand_train(
        config=config.Data, type="val", albu_transform=albumentation_val)
    dataset_test = H2O_Dataset_hand_train(
        config=config.Data, type="test", albu_transform=albumentation_val)

    dataloader_train = DataLoader(
        dataset_train,
        config.Data.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.TrainingConfig.num_workers,
        pin_memory=True
    )

    dataloader_val = DataLoader(
        dataset_val,
        config.Data.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.TrainingConfig.num_workers,
        pin_memory=True
    )

    dataloader_test = DataLoader(
        dataset_test,
        config.Data.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.TrainingConfig.num_workers,
        pin_memory=True
    )

    return {
        "train": dataloader_train,
        "val": dataloader_val,
        "test": dataloader_test,
        "albumentation_train": albumentations_train,
        "albumentation_val": albumentation_val
    }
