from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from configs.dataset_config import IMAGE_DIR, MASK_DIR
from datasets.transforms import get_val_transforms


class ISICDataset(Dataset):

    def __init__(
        self,
        csv_file,
        transform=None,
        return_metadata=False,
    ):

        self.df = pd.read_csv(csv_file)

        self.image_dir = IMAGE_DIR
        self.mask_dir = MASK_DIR

        self.return_metadata = return_metadata

        self.transform = transform or get_val_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = cv2.imread(str(self.image_dir / row["image_name"]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(
            str(self.mask_dir / row["mask_name"]),
            cv2.IMREAD_GRAYSCALE,
        )

        mask = (mask > 0).astype(np.float32)

        transformed = self.transform(
            image=image,
            mask=mask,
        )

        image = transformed["image"]
        mask = transformed["mask"].unsqueeze(0).float()

        if self.return_metadata:

            metadata = {
                "image_id": row["image_id"],
                "lesion_percentage": row["lesion_percentage"],
                "lesion_bin": row["lesion_bin"],
                "split": row.get("split", None),
            }

            return image, mask, metadata

        return image, mask