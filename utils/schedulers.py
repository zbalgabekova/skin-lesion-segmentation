"""
Learning rate scheduler factory.
"""

import torch

from configs.training_config import (
    NONE,
    COSINE,
    PLATEAU,
    STEP,
    ONE_CYCLE,
    SCHEDULER,
    NUM_EPOCHS,
    T_MAX,
    ETA_MIN,
    STEP_SIZE,
    GAMMA,
    PATIENCE,
    FACTOR,
    MAX_LR,
)


def create_scheduler(
    optimizer,
    scheduler_name=SCHEDULER,
    num_epochs=NUM_EPOCHS,
    steps_per_epoch=None,
):
    """
    Create a learning rate scheduler.

    Args:
        optimizer:
            Optimizer.

        scheduler_name:
            Scheduler name.

        num_epochs:
            Number of training epochs.

        steps_per_epoch:
            Required only for OneCycleLR.

    Returns:
        torch.optim.lr_scheduler._LRScheduler | None
    """

    scheduler_name = scheduler_name.lower()

    if scheduler_name == NONE:
        return None

    if scheduler_name == COSINE:

        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_MAX,
            eta_min=ETA_MIN,
        )

    if scheduler_name == PLATEAU:

        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=FACTOR,
            patience=PATIENCE,
        )

    if scheduler_name == STEP:

        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=STEP_SIZE,
            gamma=GAMMA,
        )

    if scheduler_name == ONE_CYCLE:

        if steps_per_epoch is None:
            raise ValueError(
                "steps_per_epoch must be provided "
                "for OneCycleLR."
            )

        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=MAX_LR,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
        )

    raise ValueError(
        f"Unknown scheduler: {scheduler_name}"
    )
    
def step_scheduler(scheduler, metric=None):
    """
    Advance the scheduler by one step.
    """

    if scheduler is None:
        return

    if isinstance(
        scheduler,
        torch.optim.lr_scheduler.ReduceLROnPlateau,
    ):
        scheduler.step(metric)
    else:
        scheduler.step()