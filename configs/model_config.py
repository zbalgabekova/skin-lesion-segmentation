"""
Model configuration.
"""

# ==========================================================
# Available models
# ==========================================================

UNET = "unet"
ATTENTION_UNET = "attention_unet"
SEGFORMER = "segformer"

# ==========================================================
# Default model
# ==========================================================

MODEL_NAME = UNET

# ==========================================================
# Encoder
# ==========================================================

ENCODER_NAME = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# ==========================================================
# Segmentation
# ==========================================================

IN_CHANNELS = 3
NUM_CLASSES = 1
ACTIVATION = None

# ==========================================================
# SegFormer
# ==========================================================

SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"

SEGFORMER_PRETRAINED = True

SEGFORMER_FREEZE_ENCODER = False