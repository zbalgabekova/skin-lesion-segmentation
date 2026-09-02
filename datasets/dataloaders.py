from torch.utils.data import DataLoader

from configs.dataset_config import (
    SPLITS_DIR,
    TRAIN_SPLIT,
    VAL_SPLIT,
    TEST_SPLIT,
)

from configs.training_config import (
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
)

from datasets.isic_dataset import ISICDataset
from datasets.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def get_dataloader(
    split,
    batch_size=BATCH_SIZE,
    image_size=512,
    return_metadata=False,
):
    """
    Create a DataLoader for a dataset split.

    Args:
        split: "train", "val" or "test"
        batch_size: Batch size
        image_size: Resize dimension
        return_metadata: Whether to return metadata

    Returns:
        torch.utils.data.DataLoader
    """

    if split not in {TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT}:
        raise ValueError(
            f"Unknown split '{split}'. "
            f"Expected one of: "
            f"{TRAIN_SPLIT}, {VAL_SPLIT}, {TEST_SPLIT}"
        )

    csv_file = SPLITS_DIR / f"{split}.csv"

    if split == TRAIN_SPLIT:
        transform = get_train_transforms(image_size)
        shuffle = True
    else:
        transform = get_val_transforms(image_size)
        shuffle = False

    dataset = ISICDataset(
        csv_file=csv_file,
        transform=transform,
        return_metadata=return_metadata,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    return loader