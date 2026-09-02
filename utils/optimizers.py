"""
Optimizer factory.
"""

import torch

from configs.training_config import (
    ADAM,
    ADAMW,
    SGD,
    OPTIMIZER,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MOMENTUM,
)


def create_optimizer(
    model,
    optimizer_name=OPTIMIZER,
    learning_rate=LEARNING_RATE,
):
    """
    Create optimizer.

    Args:
        model:
            PyTorch model.

        optimizer_name:
            Optimizer name.

        learning_rate:
            Learning rate.

    Returns:
        torch.optim.Optimizer
    """

    optimizer_name = optimizer_name.lower()

    parameters = model.parameters()

    if optimizer_name == ADAM:

        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

    elif optimizer_name == ADAMW:

        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

    elif optimizer_name == SGD:

        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )

    else:

        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )