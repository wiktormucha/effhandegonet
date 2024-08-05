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
