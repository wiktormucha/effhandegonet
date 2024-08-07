import albumentations as A

albumentations = A.ReplayCompose([

    A.OneOf([
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.VerticalFlip(always_apply=False, p=0.5),
            A.SafeRotate(always_apply=False, p=0.4, limit=(-45, 45),
                         interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
            A.RandomSizedCrop(always_apply=False, p=0.5, min_max_height=(
                500, 720), height=720, width=1280, w2h_ratio=1.0, interpolation=0),
            ], p=0.8),

], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))
