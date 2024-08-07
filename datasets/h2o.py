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
import torch
from torch.nn.utils.rnn import pad_sequence
import random
from datasets.common_data_utils import sample2, albumentation_to_sequence
from utils.egocentric import read_yolo_labels


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=np.VisibleDeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


CAM_INTRS_H2O = np.array([[636.6593017578125, 0.00000000e+00, 635.283881879317],
                          [0.00000000e+00, 636.251953125, 366.8740353496978],
                          [0.00000000e+00, 0.00000000e+00, 1.0]])


class CustomCollate:
    def __init__(self, pad_idx=0):
        self.pad_idx = pad_idx

    def __call__(self, batch):

        action_id = [item["action_id"] for item in batch]
        positions = [item["positions"] for item in batch]
        positions = pad_sequence(
            positions, batch_first=True, padding_value=self.pad_idx)
        action_label = [item["action_label"] for item in batch]

        return {
            'action_id': torch.tensor(action_id),
            'positions': positions.float(),
            'action_label': torch.tensor(action_label),
        }


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

        # Assuming CAM_INTRS_H2O is a constant that doesn't change per iteration
        cam_instr = CAM_INTRS_H2O

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

        left_hand = project_points_3D_to_2D(left_hand, cam_instr)
        right_hand = project_points_3D_to_2D(right_hand, cam_instr)

        merged = np.stack([left_hand, right_hand])

        merged = merged.reshape(42, 2)

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


class H2O_actions(Dataset):

    def __init__(self, data_cfg, albumentations: A = None, subset_type: str = 'train') -> None:
        super().__init__()

        # Cfg parser

        self.no_of_input_frames = data_cfg.no_of_input_frames
        annotation_pth = data_cfg.annotation_train
        data_dir = data_cfg.data_dir

        self.using_obj_bb = data_cfg.using_obj_bb
        self.using_obj_label = data_cfg.using_obj_label
        self.subset_type = subset_type
        self.hand_pose_type = data_cfg.hand_pose_type
        self.obj_pose_type = data_cfg.obj_pose_type
        self.apply_vanishing = data_cfg.apply_vanishing
        self.vanishing_proability = data_cfg.vanishing_proability
        self.obj_to_vanish = data_cfg.obj_to_vanish
        # self.own_obj_path = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands/exp2/labels"

        if self.subset_type == 'train':
            self.sample = 'random'
            label_file = 'action_train.txt'

            # self.own_obj_path = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands_train/exp/labels"
            # self.path_to_own_pose = '/data/wmucha/datasets/h2o_ego/train_own_pose'
        elif self.subset_type == 'val':
            self.sample = 'uniform'
            label_file = 'action_val.txt'

            # self.own_obj_path = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands/exp2/labels"
            # self.path_to_own_pose = '/data/wmucha/datasets/h2o_ego/val_own_pose'
        elif self.subset_type == 'test':
            self.sample = 'uniform'
            label_file = 'action_test.txt'

            # self.own_obj_path = "/caa/Homes01/wmucha/repos/yolov7/h2o_bb_hands_train/exp/labels"
            # self.path_to_own_pose = '/data/wmucha/datasets/h2o_ego/train_own_pose'

        final_pth = os.path.join(annotation_pth, label_file)

        df = pd.DataFrame()
        if os.path.exists(final_pth):
            df = pd.read_csv(final_pth, sep=' ')
        else:
            raise ValueError("File not found: ", final_pth)

        self.id = df['id']
        self.path = df['path']
        if self.subset_type != 'test':
            self.action_label = df['action_label']
        self.start_act = df['start_act']
        self.end_act = df['end_act']
        self.start_frame = df['start_frame']
        self.end_frame = df['end_frame']
        self.data_path = data_dir
        self.width = 1280
        self.height = 720

        self.vid_clip_flag = False
        self.hand_pose_flag = False
        self.objs_flag = False
        self.frames_pths = []
        self.hand_pose_pths = []
        self.objs_pths = []
        self.action_length = []
        self.starting_point = []
        self.albumentations = albumentations
        self.cam_instr = CAM_INTRS_H2O
        # start_pt = 0

        if False:
            self.vid_clip_flag = True
            self.frames_pths = [self.__read_sequence_paths(
                init_pth=self.path[idx], start_idx=self.start_act[idx], end_idx=self.end_act[idx]) for idx in range(len(self.id))]

        if True:
            self.hand_pose_flag = True
            self.hand_pose_pths = [self.__read_sequence_paths(init_pth=self.path[idx], start_idx=self.start_act[idx],
                                                              end_idx=self.end_act[idx], seq_type='hand_pose', file_format='.txt') for idx in range(len(self.id))]

        if True:
            self.objs_flag = True
            self.objs_pths = [self.__read_sequence_paths(init_pth=self.path[idx], start_idx=self.start_act[idx],
                                                         end_idx=self.end_act[idx], seq_type='obj_pose', file_format='.txt') for idx in range(len(self.id))]

    def __read_sequence_paths(self, init_pth, start_idx, end_idx, cam_no='cam4', seq_type='rgb', file_format='.png'):
        final_pth = os.path.join(self.data_path, init_pth, cam_no, seq_type)
        if not os.path.isdir(final_pth):
            raise ValueError("The path does not exist: ", final_pth)

        return [f'{final_pth}/{frame_id:06d}{file_format}' for frame_id in range(start_idx, end_idx+1)]

    def __len__(self):
        return len(self.id)

    def __getitem__(self, idx):

        action_id = self.id[idx]

        if self.subset_type == "test":
            action = 999
        else:
            action = self.action_label[idx]

        if self.hand_pose_flag:

            indxs_to_sample = sample2(
                input_frames=self.hand_pose_pths[idx], no_of_outframes=self.no_of_input_frames, sampling_type=self.sample)
            positions = self.__load_handpose_to_tensor(
                frames_list=np.array(self.hand_pose_pths[idx]), obj_list=np.array(self.objs_pths[idx]), indxs_to_sample=indxs_to_sample, cam_instr=self.cam_instr)

        return {
            "action_id": action_id,
            "action_label": action,
            "positions": positions,
        }

    def __load_handpose_to_tensor(self, frames_list: np.array, obj_list: np.array, indxs_to_sample, cam_instr) -> torch.tensor:

        # Subsample idxs to load frames
        frames_list = frames_list[indxs_to_sample]
        obj_list = obj_list[indxs_to_sample]

        pts = []
        background = np.zeros((720, 1280, 3))

        transform_replay = None
        vanish_fag = False
        obj_to_vanish = None

        # if True:
        if (random.randint(0, 100) < (self.vanishing_proability*100)) & (self.apply_vanishing) and (self.subset_type == 'train'):
            vanish_fag = True
            obj_to_vanish = random.randint(0, 3)

        i = 0

        horizontal_flip_flag = False
        vertical_flip_flag = False

        for frame, obj_pth in zip(frames_list, obj_list):

            if not os.path.isfile(frame):
                raise ValueError('File does not exist... ', frame)
            # Get hands
            if self.hand_pose_type == 'gt_hand_pose':

                hand_pose = np.loadtxt(frame)
                gt_pts = np.split(hand_pose, [1, 64, 65, 128])
                left_hand = np.reshape(gt_pts[1], (21, 3))
                right_hand = np.reshape(gt_pts[3], (21, 3))

                # Put to 2D

                is_left = int(gt_pts[0].tolist()[0])
                is_right = int(gt_pts[2].tolist()[0])

                merged = np.concatenate([left_hand, right_hand], axis=0)
                merged = project_points_3D_to_2D(merged, self.cam_instr)

                if is_left == 0:
                    merged[:21] = 0

                if is_right == 0:
                    merged[21:] = 0

            elif self.hand_pose_type == 'own_hand_pose':
                # TODo load own data

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_ownmodel'))
                merged = hand_pose.reshape(42, 2)

            elif self.hand_pose_type == 'mediapipe_hand_pose':
                # TODo load own data

                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_mediapipe'))
                merged = hand_pose.reshape(42, 2)

            elif self.hand_pose_type == 'ego_handpoints':
                # TODo load own data
                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_ownmodel_ego'))
                merged = hand_pose.reshape(42, 2)

            elif self.hand_pose_type == 'hand_resnet50':
                # TODo load own data
                hand_pose = np.loadtxt(frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                    'hand_pose', 'hand_pose_resnet50'))
                merged = hand_pose.reshape(42, 2)

            # Getting OBJ label
            if self.using_obj_bb or self.using_obj_label:

                obj_pose = np.loadtxt(obj_pth)
                pts_3d = obj_pose[1:64].reshape((21, 3))
                obj_label = obj_pose[0]

                if self.using_obj_bb:

                    if self.obj_pose_type == 'GT':
                        pts_2d = project_points_3D_to_2D(pts_3d, cam_instr)
                        xmax, ymax = pts_2d.max(axis=0)
                        xmin, ymin = pts_2d.min(axis=0)
                        obj_bb = np.array([xmin, ymin,
                                           xmax, ymin, xmin, ymax, xmax, ymax])

                        obj_bb = obj_bb.reshape(4, 2)

                    elif self.obj_pose_type == 'YoloV7':
                        objs_key_list = [0, 1, 2, 3, 4, 5, 6, 7]
                        obj_pose_pth = frame.replace('subject1', 'subject1_ego').replace('subject2', 'subject2_ego').replace('subject3', 'subject3_ego').replace('subject4', 'subject4_ego').replace(
                            'hand_pose', 'obj_pose_ownmodel')

                        yolo_labels_file = pd.read_csv(
                            obj_pose_pth, sep=" ", header=None, index_col=None)
                        yolo_objs = read_yolo_labels(
                            yolo_labels=yolo_labels_file)

                        obj_bb = 0
                        for key in yolo_objs:
                            if key in objs_key_list:
                                obj = yolo_objs[key][0]

                                ymin = ((obj.yc - (obj.height/2)))
                                ymax = ((obj.yc + (obj.height/2)))
                                xmin = ((obj.xc - (obj.width/2)))
                                xmax = ((obj.xc + (obj.width/2)))

                                obj_bb = np.array([xmin, ymin,
                                                   xmax, ymin, xmin, ymax, xmax, ymax])

                                obj_bb = obj_bb.reshape(
                                    4, 2) * (self.width, self.height)

                                obj_label = (obj.label + 1)
                        if type(obj_bb) is int:

                            obj_bb = np.zeros((4, 2))

                    merged = np.concatenate((merged, obj_bb), axis=0)

            # Apply albumentations to the sequence
            if self.albumentations and vanish_fag == False:

                transformed = albumentation_to_sequence(
                    background, merged, self.albumentations, transform_replay, len(pts))
                if len(pts) == 0:
                    transform_replay = transformed["replay"]

                    # Get flags
                    horizontal_flip_flag = transform_replay["transforms"][0]["transforms"][0]["applied"]
                    # vertical_flip_flag = transform_replay["transforms"][0]["transforms"][1]["applied"]

                transformed_keypoints = np.array(transformed["keypoints"])

                if horizontal_flip_flag:

                    ptsL = transformed_keypoints[21:42]
                    ptsR = transformed_keypoints[0:21]
                    obj = transformed_keypoints[42:]
                    # concatenate ptsL and ptsR to shape (42,2)
                    transformed_keypoints = np.concatenate(
                        (ptsL, ptsR, obj), axis=0)

            # # No albumentations:
            else:
                transformed_keypoints = merged

            transformed_keypoints = transformed_keypoints / (
                self.width, self.height)
            transformed_keypoints = transformed_keypoints.flatten()
            if self.using_obj_bb or self.using_obj_label:
                final_keypoints = np.empty(
                    transformed_keypoints.size + 1)
                final_keypoints[:transformed_keypoints.size] = transformed_keypoints
                final_keypoints[transformed_keypoints.size:] = obj_label
                transformed_keypoints = final_keypoints
            # Apply reduction of information:
            if vanish_fag:
                transformed_keypoints = vanish_keypoints(
                    transformed_keypoints, obj_to_vanish)

            pts.append(torch.tensor(transformed_keypoints))

            i += 1

        return torch.stack(pts)


def vanish_keypoints(keypoints: np.array, obj_to_vanish: int) -> np.array:
    # type = 2
    if obj_to_vanish == 0:
        # Make left hand equal to 0
        keypoints[0:42] = 0
    elif obj_to_vanish == 1:
        # Make right hand equal to 0
        keypoints[42:84] = 0
    elif obj_to_vanish == 2:
        # Make obj pose equal to 0
        keypoints[84:92] = 0
    elif obj_to_vanish == 3:
        # Make obj label equal to 0
        keypoints[92] = 0

    return keypoints


def get_H2O_actions_dataloader(config, albumentations=None):

    train_dataset = H2O_actions(
        data_cfg=config.DataConfig, albumentations=albumentations)

    print("Len of train: ", len(train_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        config.TrainingConfigAction.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=12,
        pin_memory=True,
        collate_fn=CustomCollate(),
    )

    val_dataset = H2O_actions(data_cfg=config.DataConfig,
                              albumentations=None,
                              subset_type="val")

    print("Len of val: ", len(val_dataset))

    val_dataloader = DataLoader(
        val_dataset,
        config.TrainingConfigAction.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=12,
        pin_memory=True,
        collate_fn=CustomCollate(),
    )

    test_dataset = H2O_actions(data_cfg=config.DataConfig,
                               albumentations=None,
                               subset_type="test")

    print("Len of val: ", len(test_dataset))

    test_dataloader = DataLoader(
        test_dataset,
        config.TrainingConfigAction.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=12,
        pin_memory=True,
        collate_fn=CustomCollate(),
    )

    return {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
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
