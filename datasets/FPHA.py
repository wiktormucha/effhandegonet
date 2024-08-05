import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
import cv2 as cv2
from natsort import natsorted
from PIL import Image
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.general_utils import vector_to_heatmaps

FPHA_CAM_INSTR = np.array([[1395.749023, 0, 935.732544],
                           [0, 1395.749268, 540.681030], [0, 0, 1]])

FPHA_CAM_EXTR = np.array([[0.999988496304, -0.00468848412856, 0.000982563360594,
                           25.7], [0.00469115935266, 0.999985218048, -0.00273845880292, 1.22],
                          [-0.000969709653873, 0.00274303671904, 0.99999576807,
                           3.902], [0, 0, 0, 1]])

FPHA_RENDER_IDX = np.array(
    [0, 1, 6, 7, 8, 2, 9, 10, 11, 3, 12, 13, 14, 4, 15, 16, 17, 5, 18, 19, 20])

FPHA_OBJ_DICT = {
    "unknown": 0,
    "juice": 1,
    "peanut_butter": 2,
    "fork": 3,
    "spoon": 4,
    "milk": 5,
    "mug": 6,
    "tea_bag": 7,
    "salt": 8,
    "liquid_soap": 9,
    "sponge": 10,
    "soda_can": 11,
    "spray": 12,
    "pen": 13,
    "paper": 14,
    "letter": 15,
    "book": 16,
    "calculator": 17,
    "match": 18,
    "cell_charger": 19,
    "glasses": 20,
    "wallet": 21,
    "coin": 22,
    "card": 23,
    "wine_bottle": 24,
    "wine_glass": 25,
    "hand": 26
}
FPHA_ACTIONSTR_TO_OBJ = {
    "open_juice_bottle": "juice",
    "close_juice_bottle": "juice",
    "pour_juice_bottle": "juice",
    "open_peanut_butter": "peanut_butter",
    "close_peanut_butter": "peanut_butter",
    "prick": "fork",
    "sprinkle": "spoon",
    "scoop_spoon": "spoon",
    "put_sugar": "spoon",
    "stir": "spoon",
    "open_milk": "milk",
    "close_milk": "milk",
    "pour_milk": "milk",
    "drink_mug": "mug",
    "put_tea_bag": "tea_bag",
    "put_salt": "salt",
    "open_liquid_soap": "liquid_soap",
    "close_liquid_soap": "liquid_soap",
    "pour_liquid_soap": "liquid_soap",
    "wash_sponge": "sponge",
    "flip_sponge": "sponge",
    "scratch_sponge": "sponge",
    "squeeze_sponge": "sponge",
    "open_soda_can": "soda_can",
    "use_flash": "spray",
    "write": "pen",
    "tear_paper": "paper",
    "squeeze_paper": "paper",
    "open_letter": "letter",
    "take_letter_from_enveloppe": "letter",
    "read_letter": "paper",
    "flip_pages": "book",
    "use_calculator": "calculator",
    "light_candle": "match",
    "charge_cell_phone": "cell_charger",
    "unfold_glasses": "glasses",
    "clean_glasses": "glasses",
    "open_wallet": "wallet",
    "give_coin": "coin",
    "receive_coin": "coin",
    "give_card": "card",
    "pour_wine": "wine_bottle",
    "toast_wine": "wine_glass",
    "handshake": "hand",
    "high_five": "hand"}

FPHA_ACTION_TO_OBJ = {
    1: {"action": "open_juice_bottle", "object": "juice"},
    2: {"action": "close_juice_bottle", "object": "juice"},
    3: {"action": "pour_juice_bottle", "object": "juice"},
    4: {"action": "open_peanut_butter", "object": "peanut_butter"},
    5: {"action": "close_peanut_butter", "object": "peanut_butter"},
    6: {"action": "prick", "object": "fork"},
    7: {"action": "sprinkle", "object": "spoon"},
    8: {"action": "scoop_spoon", "object": "spoon"},
    9: {"action": "put_sugar", "object": "spoon"},
    10: {"action": "stir", "object": "spoon"},
    11: {"action": "open_milk", "object": "milk"},
    12: {"action": "close_milk", "object": "milk"},
    13: {"action": "pour_milk", "object": "milk"},
    14: {"action": "drink_mug", "object": "mug"},
    15: {"action": "put_tea_bag", "object": "tea_bag"},
    16: {"action": "put_salt", "object": "salt"},
    17: {"action": "open_liquid_soap", "object": "liquid_soap"},
    18: {"action": "close_liquid_soap", "object": "liquid_soap"},
    19: {"action": "pour_liquid_soap", "object": "liquid_soap"},
    20: {"action": "wash_sponge", "object": "sponge"},
    21: {"action": "flip_sponge", "object": "sponge"},
    22: {"action": "scratch_sponge", "object": "sponge"},
    23: {"action": "squeeze_sponge", "object": "sponge"},
    24: {"action": "open_soda_can", "object": "soda_can"},
    25: {"action": "use_flash", "object": "spray"},
    26: {"action": "write", "object": "pen"},
    27: {"action": "tear_paper", "object": "paper"},
    28: {"action": "squeeze_paper", "object": "paper"},
    29: {"action": "open_letter", "object": "letter"},
    30: {"action": "take_letter_from_enveloppe", "object": "letter"},
    31: {"action": "read_letter", "object": "paper"},
    32: {"action": "flip_pages", "object": "book"},
    33: {"action": "use_calculator", "object": "calculator"},
    34: {"action": "light_candle", "object": "match"},
    35: {"action": "charge_cell_phone", "object": "cell_charger"},
    36: {"action": "unfold_glasses", "object": "glasses"},
    37: {"action": "clean_glasses", "object": "glasses"},
    38: {"action": "open_wallet", "object": "wallet"},
    39: {"action": "give_coin", "object": "coin"},
    40: {"action": "receive_coin", "object": "coin"},
    41: {"action": "give_card", "object": "card"},
    42: {"action": "pour_wine", "object": "wine_bottle"},
    43: {"action": "toast_wine", "object": "wine_glass"},
    44: {"action": "handshake", "object": "hand"},
    45: {"action": "high_five", "object": "hand"}
}


class FPHA_pose(Dataset):

    def __init__(self, data_cfg, type='train', albu=None) -> None:
        super().__init__()

        base = data_cfg.path
        pth = os.path.join(base, 'data_split_action_recognition.txt')
        frame_root_pth = os.path.join(base, 'Video_files')
        hand_pose_root_pth = os.path.join(base, 'Hand_pose_annotation_v1')

        self.albu = albu
        self.subset_type = type
        self.img_size = data_cfg.img_size

        df = pd.read_csv(pth, header=None)

        df_train = df[1:601]
        df_test = df[602:]

        if self.img_size[0] == 224:
            self.heatmap_dim = 56
        elif self.img_size[0] == 512:
            self.heatmap_dim = 128

        if self.subset_type == 'train':
            self.action_list = np.array(df_train)

        elif self.subset_type == 'val':
            self.action_list = np.array(df_test)

        else:
            raise ValueError('Wrong subset type given: ', self.subset_type)

        action_labels = []
        action_frames = []
        action_hand_pose = []
        all_frames = []
        all_poses = []
        action_labels_total = []

        action_id_num = []
        for i, action in enumerate(self.action_list):
            temp = action[0]
            temp = temp.split()

            # Get action labels
            action_labels.append(int(temp[1]))
            action_id_num.append(i)
            # Get frames
            frames_in_action = []
            frames_temp_pth = os.path.join(frame_root_pth, temp[0], 'color')

            for filename in os.listdir(frames_temp_pth):
                if 'jpeg' in filename:
                    f = os.path.join(frames_temp_pth, filename)
                else:
                    continue

                # checking if it is a file
                if os.path.isfile(f):
                    frames_in_action.append(f)
                    all_frames.append(f)
                    action_labels_total.append(int(temp[1]))

            action_frames.append(
                np.array(natsorted(frames_in_action), dtype=str))

            # Get handpose
            hand_temp_pth = os.path.join(
                hand_pose_root_pth, temp[0], 'skeleton.txt')
            hand_pose_df = pd.read_csv(
                hand_temp_pth, sep=' ', header=None, index_col=False)
            hand_pose_df.pop(hand_pose_df.columns[0])
            action_hand_pose.append(np.array(hand_pose_df.values.tolist()))

        self.action_label = np.array(action_labels)
        self.action_frames = action_frames
        self.action_hand_pose = action_hand_pose
        self.action_id_num = np.array(action_id_num)
        self.all_frames = np.array(all_frames)
        self.depth_img_type = "est"
        self.used_masked = True

        imgs_raw = []
        # Transform hand poses:
        keypoints_list = []
        keypoints3d_list = []
        print('Loading hand poses..')
        c = 0
        objs_labels = []
        for img_pth in tqdm(self.all_frames):

            imgs_raw.append(0)

            skeleton_pth = img_pth.split("/")
            action_label = skeleton_pth[7]

            obj_name = FPHA_ACTIONSTR_TO_OBJ[action_label]

            obj_label = FPHA_OBJ_DICT[obj_name]
            objs_labels.append(obj_label)

            file_index = skeleton_pth[-1][6:10]
            skeleton_pth = os.path.join('/', skeleton_pth[1], skeleton_pth[2], skeleton_pth[3], skeleton_pth[4],
                                        skeleton_pth[5], skeleton_pth[6], skeleton_pth[7], skeleton_pth[8], 'skeleton.txt')
            skeleton_pth = skeleton_pth.replace(
                'Video_files', 'Hand_pose_annotation_v1')

            df = pd.read_csv(skeleton_pth, sep=' ', header=None)

            loaded_keypoint = df.iloc[int(file_index)].tolist()

            assert loaded_keypoint[0] == int(file_index)

            loaded_keypoint = loaded_keypoint[1:]
            all_poses.append(loaded_keypoint)
            hand_pose = np.array(loaded_keypoint)
            skel = hand_pose.reshape(21, -1)

            ske3d_cam = wrlds_2_cam(FPHA_CAM_EXTR, skel)[FPHA_RENDER_IDX]
            skel2d_img = cam_2_image(FPHA_CAM_INSTR, ske3d_cam)

            keypoints_list.append(skel2d_img)
            keypoints3d_list.append(ske3d_cam)

            # if c == 1000:
            #     break
            c += 1

        self.all_poses = np.array(all_poses)
        self.keypoints = np.array(keypoints_list)
        self.keypoints3d = np.array(keypoints3d_list)
        self.imgs_raw = np.array(imgs_raw)
        self.action_labels_total = np.array(action_labels_total)
        self.objs_labels = np.array(objs_labels)

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index) -> None:

        img_pth = self.all_frames[index]
        img_raw = np.array(Image.open(img_pth))
        img = img_raw.copy()
        obj_label = self.objs_labels[index]
        hand_pose_raw = self.keypoints3d[index]
        hand_pose = self.keypoints[index]

        if self.albu:

            transformed = self.albu(
                image=img, keypoints=hand_pose)
            transformed_image = transformed['image']
            transformed_keypoints = np.asarray(
                transformed['keypoints']).reshape(21, 2)

        else:
            raise ValueError('No albumentation provided')

        heatmaps = vector_to_heatmaps(
            transformed_keypoints / self.img_size, scale_factor=self.heatmap_dim, out_size=self.heatmap_dim)

        return {
            'img_pth': img_pth,
            'img_raw': img_raw,
            'img': transformed_image,
            'keypoints': transformed_keypoints / self.img_size,
            'keypoints3d': hand_pose_raw.reshape(21, -1),
            'obj': obj_label,
            'heatmaps': heatmaps
        }


def wrlds_2_cam(cam_extr, skel):
    skel_hom = np.concatenate([skel, np.ones([skel.shape[0], 1])], 1)
    skel_camcoords = cam_extr.dot(skel_hom.transpose()).transpose()[
        :, :3].astype(np.float32)
    return skel_camcoords


def cam_2_image(cam_intr, skel_camcoords):
    hom_2d = np.array(cam_intr).dot(skel_camcoords.transpose()).transpose()
    skel2d = (hom_2d / hom_2d[:, 2:])[:, :2]
    return skel2d


def get_FPHAB_dataset(config):

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

    train_dataset = FPHA_pose(
        data_cfg=config.Data, type='train', albu=albumentations_train)

    dataloader_train = DataLoader(
        train_dataset,
        config.Data.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=config.TrainingConfig.num_workers,
        pin_memory=True
    )

    val_dataset = FPHA_pose(
        data_cfg=config.Data, type='val', albu=albumentation_val)

    dataloader_val = DataLoader(
        val_dataset,
        config.Data.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.TrainingConfig.num_workers,
        pin_memory=True
    )

    return {
        "train": dataloader_train,
        "val": dataloader_val,
        "albumentation_train": albumentations_train,
        "albumentation_val": albumentation_val
    }
