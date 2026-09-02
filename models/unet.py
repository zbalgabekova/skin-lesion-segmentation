"""
U-Net model based on Segmentation Models PyTorch.
"""

import segmentation_models_pytorch as smp

from configs.model_config import (
    ENCODER_NAME,
    ENCODER_WEIGHTS,
    IN_CHANNELS,
    NUM_CLASSES,
    ACTIVATION,
)


class UNetModel:
    """
    Wrapper around segmentation_models_pytorch.Unet.
    """

    def __init__(
        self,
        encoder_name=ENCODER_NAME,
        encoder_weights=ENCODER_WEIGHTS,
        in_channels=IN_CHANNELS,
        classes=NUM_CLASSES,
        activation=ACTIVATION,
    ):

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
            activation=activation,
        )

    def get_model(self):
        return self.model