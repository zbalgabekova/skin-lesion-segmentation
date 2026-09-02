from pathlib import Path

# ==========================================================
# Paths
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"

IMAGE_DIR = DATA_DIR / "ISIC2018_Task1-2_Training_Input"
MASK_DIR = DATA_DIR / "ISIC2018_Task1_Training_GroundTruth"

SPLITS_DIR = DATA_DIR / "splits"

OUTPUT_DIR = PROJECT_ROOT / "outputs"

# ==========================================================
# Dataset
# ==========================================================

IMAGE_SIZE = 512

NUM_CLASSES = 1

# ==========================================================
# ImageNet normalization
# ==========================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ==========================================================
# Dataset splits
# ==========================================================

TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
TEST_SPLIT = "test"