"""
Factory for creating segmentation models.
"""

from configs.model_config import (
    UNET,
    ATTENTION_UNET,
    SEGFORMER,
)

from models.unet import UNetModel
from models.attention_unet import AttentionUNetModel
from models.segformer import SegFormerModel


def create_model(model_name):
    """
    Create a segmentation model.

    Args:
        model_name (str):
            Model name.

    Returns:
        torch.nn.Module
    """

    model_name = model_name.lower()

    if model_name == UNET:
        return UNetModel().get_model()

    elif model_name == ATTENTION_UNET:
        return AttentionUNetModel().get_model()

    elif model_name == SEGFORMER:
        return SegFormerModel().get_model()
        

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: "
            f"{UNET}, "
            f"{ATTENTION_UNET}, "
            f"{SEGFORMER}"
        )