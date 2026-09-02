"""
SegFormer model for medical image segmentation.

Wrapper around Hugging Face Transformers.
"""

import torch
import torch.nn as nn

from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
)

from configs.model_config import (
    SEGFORMER_MODEL,
    SEGFORMER_PRETRAINED,
    SEGFORMER_FREEZE_ENCODER,
    NUM_CLASSES,
)


class SegFormerModel(nn.Module):
    """
    Wrapper around Hugging Face SegFormer.

    Returns
    -------
    Tensor
        Segmentation logits of shape:

            (B, C, H, W)
    """

    def __init__(
        self,
        model_name=SEGFORMER_MODEL,
        pretrained=SEGFORMER_PRETRAINED,
        num_classes=NUM_CLASSES,
    ):

        super().__init__()

        self.model_name = model_name
        
        self.pretrained = pretrained

        self.num_classes = num_classes

        self.model = None

        self._build_model()

    def _build_model(self):
        """
        Create SegFormer model.
        """

        id2label = {
            0: "lesion",
        }

        label2id = {
            "lesion": 0,
        }

        if self.pretrained:

            self.model = (
                SegformerForSemanticSegmentation
                .from_pretrained(
                    self.model_name,
                    num_labels=self.num_classes,
                    id2label=id2label,
                    label2id=label2id,
                    ignore_mismatched_sizes=True,
                )
            )

        else:

            config = SegformerConfig(
                num_labels=self.num_classes,
                id2label=id2label,
                label2id=label2id,
            )

            self.model = (
                SegformerForSemanticSegmentation(
                    config
                )
            )

        if SEGFORMER_FREEZE_ENCODER:

            self.freeze_encoder()
            
    def freeze_encoder(self):
        """
        Freeze encoder parameters.
        """

        for parameter in self.model.segformer.encoder.parameters():

            parameter.requires_grad = False
            
            
    def unfreeze_encoder(self):
        """
        Unfreeze encoder parameters.
        """

        for parameter in self.model.segformer.encoder.parameters():

            parameter.requires_grad = True
            
            
    @property
    def num_parameters(self):
        """
        Number of trainable parameters.
        """

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )
    
    
    
    def print_model_info(self):
        """
        Print model information.
        """

        print("=" * 60)

        print("SegFormer")

        print("=" * 60)

        print(f"Model: {self.model_name}")

        print(f"Pretrained: {self.pretrained}")

        print(f"Classes: {self.num_classes}")

        print(
            f"Trainable parameters: "
            f"{self.num_parameters:,}"
        )

        print("=" * 60)

        

    def forward(
        self,
        images,
    ):
        """
        Forward pass.

        Args:
            images:
                Tensor of shape (B, 3, H, W)

        Returns:
            Tensor of shape (B, C, H, W)
        """

        # Original image size
        height, width = images.shape[-2:]

        # Hugging Face forward pass
        outputs = self.model(
            pixel_values=images,
        )

        logits = outputs.logits

        # Upsample logits to the original resolution
        logits = nn.functional.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )

        return logits

    def get_model(self):
        """
        Return initialized model.
        """

        return self