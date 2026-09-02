from pathlib import Path
from collections import Counter
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# ==========================================================
# Dataset paths
# ==========================================================

DATA_DIR = Path("data")

IMAGE_DIR = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"

# ==========================================================

def main():

    start_time = time.time()

    image_files = sorted(IMAGE_DIR.glob("*.jpg"))
    mask_files = sorted(MASK_DIR.glob("*.png"))

    print("=" * 70)
    print("ISIC 2018 DATASET INSPECTION")
    print("=" * 70)

    print(f"Training images : {len(image_files)}")
    print(f"Training masks  : {len(mask_files)}")

    image_ids = {x.stem for x in image_files}
    mask_ids = {x.stem.replace("_segmentation", "") for x in mask_files}

    common_ids = sorted(image_ids & mask_ids)

    print(f"Matching pairs  : {len(common_ids)}")
    print(f"Images missing masks : {len(image_ids-mask_ids)}")
    print(f"Masks missing images : {len(mask_ids-image_ids)}")

    if len(common_ids) == 0:
        print("\nERROR: No matching pairs found.")
        return

    image_sizes = Counter()
    mask_sizes = Counter()

    lesion_areas = []
    lesion_percentages = []

    empty_masks = 0
    size_mismatches = 0

    all_mask_values = set()

    print("\nInspecting files...\n")

    for image_id in tqdm(common_ids):

        image_path = IMAGE_DIR / f"{image_id}.jpg"
        mask_path = MASK_DIR / f"{image_id}_segmentation.png"

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        image_sizes[image.size] += 1
        mask_sizes[mask.size] += 1

        if image.size != mask.size:
            size_mismatches += 1

        mask_np = np.array(mask)

        all_mask_values.update(np.unique(mask_np))

        lesion_pixels = np.sum(mask_np > 0)

        if lesion_pixels == 0:
            empty_masks += 1

        lesion_areas.append(lesion_pixels)

        total_pixels = mask_np.shape[0] * mask_np.shape[1]

        lesion_percentages.append(
            lesion_pixels / total_pixels * 100
        )

    lesion_areas = np.array(lesion_areas)
    lesion_percentages = np.array(lesion_percentages)

    print("\n" + "=" * 70)
    print("IMAGE INFORMATION")
    print("=" * 70)

    print(f"Unique image resolutions : {len(image_sizes)}")
    print(f"Unique mask resolutions  : {len(mask_sizes)}")
    print(f"Image-mask size mismatches : {size_mismatches}")

    print("\nMost common image resolutions:")

    for size, count in image_sizes.most_common(10):
        print(f"{size} : {count}")

    print("\n" + "=" * 70)
    print("MASK INFORMATION")
    print("=" * 70)

    print(f"Mask values : {sorted(all_mask_values)}")
    print(f"Empty masks : {empty_masks}")

    if all_mask_values == {0, 255}:
        print("✓ Masks are binary.")
    else:
        print("WARNING: Masks are not binary!")

    print("\n" + "=" * 70)
    print("LESION STATISTICS")
    print("=" * 70)

    print(f"Minimum lesion area : {lesion_areas.min():,} pixels")
    print(f"Maximum lesion area : {lesion_areas.max():,} pixels")
    print(f"Mean lesion area    : {lesion_areas.mean():,.1f}")
    print(f"Median lesion area  : {np.median(lesion_areas):,.1f}")
    print(f"Std lesion area     : {lesion_areas.std():,.1f}")

    print()

    print(f"Minimum lesion percentage : {lesion_percentages.min():.3f}%")
    print(f"Maximum lesion percentage : {lesion_percentages.max():.3f}%")
    print(f"Mean lesion percentage    : {lesion_percentages.mean():.3f}%")

    print("\n" + "=" * 70)
    print("VISUALIZATION")
    print("=" * 70)

    sample_ids = random.sample(common_ids, 4)

    fig, axes = plt.subplots(4, 2, figsize=(8, 16))

    for i, image_id in enumerate(sample_ids):

        image = Image.open(
            IMAGE_DIR / f"{image_id}.jpg"
        )

        mask = Image.open(
            MASK_DIR / f"{image_id}_segmentation.png"
        )

        axes[i,0].imshow(image)
        axes[i,0].set_title(image_id)
        axes[i,0].axis("off")

        axes[i,1].imshow(mask, cmap="gray")
        axes[i,1].set_title("Mask")
        axes[i,1].axis("off")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,5))
    plt.hist(lesion_percentages, bins=30)
    plt.xlabel("Lesion Percentage (%)")
    plt.ylabel("Number of Images")
    plt.title("Distribution of Lesion Sizes")
    plt.show()

    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Inspection finished in {elapsed:.1f} seconds.")
    print("Dataset inspection completed successfully.")


if __name__ == "__main__":
    main()