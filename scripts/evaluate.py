"""
Evaluate a trained segmentation model on the test set.

Usage examples:

    python evaluate.py --model unet --checkpoint checkpoints/unet/best_model.pth

    python evaluate.py --model attention_unet --checkpoint checkpoints/attention_unet/best_model.pth

    python evaluate.py --model segformer --checkpoint checkpoints/segformer/best_model.pth
"""

import argparse
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from configs.training_config import DEVICE
from datasets.dataloaders import get_dataloader
from models.model_factory import create_model


# ==========================================================
# Argument Parser
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate segmentation model on test set."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name: unet, attention_unet, or segformer",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for binary segmentation.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Directory for evaluation results.",
    )

    return parser.parse_args()


# ==========================================================
# Checkpoint Loading
# ==========================================================

def load_checkpoint(model, checkpoint_path, device):
    """
    Load model weights from checkpoint.

    Supports:
        - plain state_dict
        - checkpoint dictionaries containing
          'model_state_dict' or 'state_dict'
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            # Assume the dictionary itself is a state dict
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # Handle checkpoints saved from DataParallel
    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    missing, unexpected = model.load_state_dict(
        cleaned_state_dict,
        strict=False,
    )

    if missing:
        print(
            f"Warning: {len(missing)} missing keys."
        )

    if unexpected:
        print(
            f"Warning: {len(unexpected)} unexpected keys."
        )

    return model


# ==========================================================
# Metrics
# ==========================================================

def calculate_metrics(
    predictions,
    targets,
    epsilon=1e-7,
):
    """
    Calculate binary segmentation metrics.

    Args:
        predictions:
            Binary predictions, shape (B, 1, H, W)

        targets:
            Binary ground truth masks.

    Returns:
        Dictionary of metrics.
    """

    predictions = predictions.float()
    targets = targets.float()

    predictions = predictions.view(
        predictions.size(0),
        -1,
    )

    targets = targets.view(
        targets.size(0),
        -1,
    )

    true_positive = (
        predictions * targets
    ).sum(dim=1)

    false_positive = (
        predictions * (1 - targets)
    ).sum(dim=1)

    false_negative = (
        (1 - predictions) * targets
    ).sum(dim=1)

    true_negative = (
        (1 - predictions)
        * (1 - targets)
    ).sum(dim=1)

    # ------------------------------------------------------
    # Dice
    # ------------------------------------------------------

    dice = (
        2 * true_positive + epsilon
    ) / (
        2 * true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    # ------------------------------------------------------
    # IoU
    # ------------------------------------------------------

    iou = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    # ------------------------------------------------------
    # Precision
    # ------------------------------------------------------

    precision = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + epsilon
    )

    # ------------------------------------------------------
    # Recall
    # ------------------------------------------------------

    recall = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_negative
        + epsilon
    )

    # ------------------------------------------------------
    # F1
    # ------------------------------------------------------

    f1 = (
        2 * precision * recall
        + epsilon
    ) / (
        precision
        + recall
        + epsilon
    )

    # ------------------------------------------------------
    # Pixel Accuracy
    # ------------------------------------------------------

    pixel_accuracy = (
        true_positive
        + true_negative
    ) / (
        true_positive
        + true_negative
        + false_positive
        + false_negative
        + epsilon
    )

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item(),
        "pixel_accuracy": pixel_accuracy.mean().item(),
    }


# ==========================================================
# Parameter Count
# ==========================================================

def count_parameters(model):
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ==========================================================
# Evaluation
# ==========================================================

def evaluate(
    model,
    dataloader,
    device,
    threshold=0.5,
):
    """
    Evaluate model on the complete test set.
    """

    model.eval()

    metric_sums = {
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "pixel_accuracy": 0.0,
    }

    total_images = 0

    total_inference_time = 0.0

    if device.type == "cuda":

        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():

        progress = tqdm(
            dataloader,
            desc="Evaluating",
        )

        for images, masks in progress:

            images = images.to(
                device,
                non_blocking=True,
            )

            masks = masks.to(
                device,
                non_blocking=True,
            )

            # --------------------------------------------------
            # Inference timing
            # --------------------------------------------------

            if device.type == "cuda":

                torch.cuda.synchronize()

            start_time = time.perf_counter()

            logits = model(images)

            if device.type == "cuda":

                torch.cuda.synchronize()

            elapsed = (
                time.perf_counter()
                - start_time
            )

            total_inference_time += elapsed

            # --------------------------------------------------
            # Convert logits to probabilities
            # --------------------------------------------------

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities >= threshold
            ).float()

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            batch_metrics = calculate_metrics(
                predictions,
                masks,
            )

            batch_size = images.size(0)

            for key in metric_sums:

                metric_sums[key] += (
                    batch_metrics[key]
                    * batch_size
                )

            total_images += batch_size

    # ------------------------------------------------------
    # Average metrics
    # ------------------------------------------------------

    results = {}

    for key in metric_sums:

        results[key] = (
            metric_sums[key]
            / total_images
        )

    results["num_images"] = total_images

    results["total_inference_time_sec"] = (
        total_inference_time
    )

    results["inference_time_per_image_sec"] = (
        total_inference_time
        / total_images
    )

    results["inference_fps"] = (
        total_images
        / total_inference_time
        if total_inference_time > 0
        else 0.0
    )

    if device.type == "cuda":

        results["peak_gpu_memory_mb"] = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

    else:

        results["peak_gpu_memory_mb"] = None

    return results


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------
    # Device
    # ------------------------------------------------------

    device = torch.device(DEVICE)

    print(f"Device     : {device}")

    if device.type == "cuda":

        print(
            f"GPU        : "
            f"{torch.cuda.get_device_name(0)}"
        )

    # ------------------------------------------------------
    # Test DataLoader
    # ------------------------------------------------------

    test_loader = get_dataloader(
        split="test",
    )

    print(
        f"Test images: "
        f"{len(test_loader.dataset)}"
    )

    # ------------------------------------------------------
    # Model
    # ------------------------------------------------------

    print(
        f"Model      : {args.model}"
    )

    model = create_model(
        args.model
    )

    model = model.to(device)

    print(
        f"Parameters : "
        f"{count_parameters(model):,}"
    )

    # ------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------

    checkpoint_path = Path(
        args.checkpoint
    )

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        f"Checkpoint : "
        f"{checkpoint_path}"
    )

    model = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    # ------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------

    print()

    print(
        f"Threshold  : "
        f"{args.threshold}"
    )

    print()

    results = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        threshold=args.threshold,
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print()

    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    print(
        f"Dice             : "
        f"{results['dice']:.4f}"
    )

    print(
        f"IoU              : "
        f"{results['iou']:.4f}"
    )

    print(
        f"Precision        : "
        f"{results['precision']:.4f}"
    )

    print(
        f"Recall           : "
        f"{results['recall']:.4f}"
    )

    print(
        f"F1               : "
        f"{results['f1']:.4f}"
    )

    print(
        f"Pixel Accuracy   : "
        f"{results['pixel_accuracy']:.4f}"
    )

    print()

    print(
        f"Inference Time   : "
        f"{results['total_inference_time_sec']:.2f} sec"
    )

    print(
        f"Time / Image     : "
        f"{results['inference_time_per_image_sec']:.4f} sec"
    )

    print(
        f"Inference FPS    : "
        f"{results['inference_fps']:.2f}"
    )

    if results["peak_gpu_memory_mb"] is not None:

        print(
            f"Peak GPU Memory  : "
            f"{results['peak_gpu_memory_mb']:.1f} MB"
        )

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / f"{args.model}_results.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print()

    print(
        f"Results saved to: "
        f"{output_file}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()