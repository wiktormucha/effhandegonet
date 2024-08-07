import albumentations as A
from albumentations.pytorch import ToTensorV2

TRAIN_DATASET_MEANS = [0.4532, 0.4522, 0.4034]
TRAIN_DATASET_STDS = [0.2218, 0.2186, 0.2413]
RAW_IMG_SIZE = 224
MODEL_IMG_SIZE = 128
BB_FACTOR = 25

COLORMAP = {
    "thumb": {"ids": [0, 1, 2, 3, 4], "color": "g"},
    "index": {"ids": [0, 5, 6, 7, 8], "color": "c"},
    "middle": {"ids": [0, 9, 10, 11, 12], "color": "b"},
    "ring": {"ids": [0, 13, 14, 15, 16], "color": "m"},
    "little": {"ids": [0, 17, 18, 19, 20], "color": "r"},
}

ALBUMENTATION_VAL = A.Compose(
    [
        A.Resize(MODEL_IMG_SIZE, MODEL_IMG_SIZE),
        A.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
        ToTensorV2()

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)
ALBUMENTATION_TRAIN = A.Compose(
    [
        A.SafeRotate(always_apply=False, p=0.5, limit=(-20, 20),
                     interpolation=0, border_mode=1, value=(0, 0, 0), mask_value=None),
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.VerticalFlip(always_apply=False, p=0.5),
        A.RandomResizedCrop(always_apply=True, p=1, height=MODEL_IMG_SIZE, width=MODEL_IMG_SIZE, scale=(
            0.3, 1.0), ratio=(1, 1), interpolation=0),
        A.MotionBlur(always_apply=False, p=0.2,
                     blur_limit=(3, 7), allow_shifted=True),
        # A.RandomGridShuffle(always_apply=False, p=0.2, grid=(2, 2)),
        A.Downscale(always_apply=False, p=0.2,
                    scale_min=0.9, scale_max=0.99),
        A.Normalize(mean=TRAIN_DATASET_MEANS, std=TRAIN_DATASET_STDS),
        ToTensorV2()

    ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)
)


action_to_verb = {
    0: 0,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 2,
    10: 2,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 3,
    18: 3,
    19: 3,
    20: 4,
    21: 4,
    22: 4,
    23: 5,
    24: 6,
    25: 6,
    26: 6,
    27: 6,
    28: 7,
    29: 7,
    30: 7,
    31: 8,
    32: 8,
    33: 9,
    34: 9,
    35: 10,
    36: 11,
    999: 999
}