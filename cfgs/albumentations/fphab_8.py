import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

albumentations = A.ReplayCompose([
    A.OneOf([
            A.HorizontalFlip(always_apply=True, p=0.4),
            # A.VerticalFlip(always_apply=True, p=0.4),
            A.SafeRotate(always_apply=True, p=0.4, limit=(-45, 45),
                         interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            A.RandomCropFromBorders(always_apply=True, p=0.4, crop_left=0.2,
                                    crop_right=0.2, crop_top=0.2, crop_bottom=0.2),
            A.RandomSizedCrop(always_apply=True, p=0.5, min_max_height=(
                500, 720), height=1080, width=1920, w2h_ratio=1.0, interpolation=0),
            ], p=0.8),
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
