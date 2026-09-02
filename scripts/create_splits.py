from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==========================================================
# Configuration
# ==========================================================

DATA_DIR = Path("data")

IMAGE_DIR = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"

SPLIT_DIR = DATA_DIR / "splits"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def print_distribution(df, name):
    print(f"\n{name}")

    dist = (
        df["lesion_bin"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    for lesion_bin, percentage in dist.items():
        print(f"{lesion_bin}: {percentage:.1f}%")


def main():

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGE_DIR.glob("*.jpg"))

    records = []

    missing_masks = []

    print("=" * 70)
    print("Creating stratified dataset splits")
    print("=" * 70)

    print("\nScanning dataset...\n")

    for image_path in tqdm(image_files):

        image_id = image_path.stem

        mask_path = MASK_DIR / f"{image_id}_segmentation.png"

        if not mask_path.exists():
            missing_masks.append(image_id)
            continue

        mask = np.array(Image.open(mask_path))

        lesion_pixels = int(np.sum(mask > 0))

        image_pixels = mask.shape[0] * mask.shape[1]

        lesion_percentage = lesion_pixels / image_pixels * 100

        records.append({
            "image_id": image_id,
            "image_name": image_path.name,
            "mask_name": mask_path.name,
            "lesion_pixels": lesion_pixels,
            "image_pixels": image_pixels,
            "lesion_percentage": lesion_percentage
        })

    print(f"\nImages found : {len(image_files)}")
    print(f"Valid pairs  : {len(records)}")
    print(f"Missing masks: {len(missing_masks)}")

    if len(records) == 0:
        raise RuntimeError("No valid image-mask pairs found.")

    df = pd.DataFrame(records)

    # -------------------------------------------------------
    # Quartile bins
    # -------------------------------------------------------

    df["lesion_bin"] = pd.qcut(
        df["lesion_percentage"],
        q=4,
        labels=["Q1", "Q2", "Q3", "Q4"]
    )

    print("\nLesion bins")

    print(df["lesion_bin"].value_counts().sort_index())

    # -------------------------------------------------------
    # Train / Temp
    # -------------------------------------------------------

    train_df, temp_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=df["lesion_bin"]
    )

    # -------------------------------------------------------
    # Validation / Test
    # -------------------------------------------------------

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=temp_df["lesion_bin"]
    )

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    summary_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    # -------------------------------------------------------
    # Overlap verification
    # -------------------------------------------------------

    train_ids = set(train_df["image_id"])
    val_ids = set(val_df["image_id"])
    test_ids = set(test_df["image_id"])

    print("\n" + "=" * 70)
    print("Overlap check")
    print("=" * 70)

    print(f"Train ∩ Validation : {len(train_ids & val_ids)}")
    print(f"Train ∩ Test       : {len(train_ids & test_ids)}")
    print(f"Validation ∩ Test  : {len(val_ids & test_ids)}")

    # -------------------------------------------------------
    # Distribution report
    # -------------------------------------------------------

    print("\n" + "=" * 70)
    print("Lesion bin distributions")
    print("=" * 70)

    print_distribution(train_df, "Train")
    print_distribution(val_df, "Validation")
    print_distribution(test_df, "Test")

    # -------------------------------------------------------
    # Sort for readability
    # -------------------------------------------------------

    train_df = train_df.sort_values("image_id")
    val_df = val_df.sort_values("image_id")
    test_df = test_df.sort_values("image_id")
    summary_df = summary_df.sort_values("image_id")

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    train_df.to_csv(SPLIT_DIR / "train.csv", index=False)
    val_df.to_csv(SPLIT_DIR / "val.csv", index=False)
    test_df.to_csv(SPLIT_DIR / "test.csv", index=False)
    summary_df.to_csv(SPLIT_DIR / "dataset_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("Dataset summary")
    print("=" * 70)

    print(f"Training   : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")

    print("\nFiles saved to:")
    print(SPLIT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()