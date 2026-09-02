import albumentations as A
from albumentations.pytorch import ToTensorV2

from configs.dataset_config import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def get_train_transforms(image_size=IMAGE_SIZE):
    """
    Data augmentation for training.
    """

    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.HorizontalFlip(p=0.5),

            A.VerticalFlip(p=0.5),

            A.RandomRotate90(p=0.5),

            A.Affine(
                scale=(0.9, 1.1),
                rotate=(-20, 20),
                translate_percent=(-0.05, 0.05),
                p=0.5,
            ),

            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),

            ToTensorV2(),
        ]
    )


def get_val_transforms(image_size=IMAGE_SIZE):
    """
    Validation / Test preprocessing.
    """

    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),

            ToTensorV2(),
        ]
    )