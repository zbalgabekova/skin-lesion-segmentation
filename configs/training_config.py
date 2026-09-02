import torch


# ==========================================================
# DataLoader
# ==========================================================

BATCH_SIZE = 8

NUM_WORKERS = 4

PIN_MEMORY = True

SHUFFLE_TRAIN = True

SHUFFLE_VAL = False

SHUFFLE_TEST = False

# ==========================================================
# Training
# ==========================================================

NUM_EPOCHS = 50


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ==========================================================
# Loss
# ==========================================================

BCE_LOSS = "bce"
DICE_LOSS = "dice"
DICE_BCE_LOSS = "dice_bce"
FOCAL_LOSS = "focal"

LOSS_FUNCTION = DICE_BCE_LOSS

# ==========================================================
# Metrics
# ==========================================================

TRAIN_METRICS = [
    "loss",
    "dice",
    "iou",
    "precision",
    "recall",
    "f1",
    "pixel_accuracy",
]

# ==========================================================
# Optimizer
# ==========================================================

ADAM = "adam"
ADAMW = "adamw"
SGD = "sgd"

OPTIMIZER = ADAMW

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MOMENTUM = 0.9

# ==========================================================
# Scheduler
# ==========================================================

NONE = "none"
COSINE = "cosine"
PLATEAU = "plateau"
STEP = "step"
ONE_CYCLE = "one_cycle"

SCHEDULER = COSINE

# CosineAnnealingLR
T_MAX = NUM_EPOCHS
ETA_MIN = 1e-6

# StepLR
STEP_SIZE = 20
GAMMA = 0.1

# ReduceLROnPlateau
PATIENCE = 5
FACTOR = 0.5

# OneCycleLR
MAX_LR = LEARNING_RATE

RESUME = False

# ==========================================================
# Mixed Precision
# ==========================================================

USE_AMP = False

# ==========================================================
# Reproducibility
# ==========================================================

SEED = 42


# ==========================================================
# Checkpoints
# ==========================================================

CHECKPOINT_DIR = "checkpoints"

BEST_MODEL_NAME = "best_model.pth"

LATEST_MODEL_NAME = "latest_model.pth"

# ==========================================================
# Early Stopping
# ==========================================================

EARLY_STOPPING = True

EARLY_STOPPING_PATIENCE = 10

MIN_DELTA = 0.0


# ==========================================================
# Gradient Clipping
# ==========================================================

GRADIENT_CLIP = 1.0


# ==========================================================
# Monitoring
# ==========================================================

MONITOR = "dice"

MODE = "max"


PRINT_EVERY = 1

SAVE_HISTORY = True