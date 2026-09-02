"""
Main training script.
"""
import random

import numpy as np
import torch

from configs.model_config import MODEL_NAME
from configs.training_config import (
    DEVICE,
    NUM_EPOCHS,
    RESUME,
)

from datasets.dataloaders import get_dataloader
from models.model_factory import create_model

from trainer import Trainer

from utils.losses import create_loss
from utils.optimizers import create_optimizer
from utils.schedulers import create_scheduler


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed=42):
    """
    Set random seed for reproducibility.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 70)
    print("Medical Image Segmentation")
    print("=" * 70)

    set_seed()

    device = torch.device(DEVICE)

    print(f"Device : {device}")

    # ------------------------------------------------------
    # Data
    # ------------------------------------------------------

    train_loader = get_dataloader(
        split="train",
    )

    val_loader = get_dataloader(
        split="val",
    )

    print(
        f"Training samples   : "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples : "
        f"{len(val_loader.dataset)}"
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    model = create_model(MODEL_NAME)

    print(f"Model : {MODEL_NAME}")

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    criterion = create_loss()

    # ------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------

    optimizer = create_optimizer(model)

    # ------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------

    scheduler = create_scheduler(
        optimizer,
        steps_per_epoch=len(train_loader),
    )

    # ------------------------------------------------------
    # Trainer
    # ------------------------------------------------------

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=NUM_EPOCHS,
    )
    
    if RESUME:
        trainer.load_checkpoint()

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    trainer.fit()

    print()

    print("=" * 70)
    print("Training finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()