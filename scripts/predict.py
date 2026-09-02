"""
Qualitative and per-image evaluation of segmentation models.

Models:
    - U-Net
    - Attention U-Net
    - SegFormer-B0

The script:

1. Loads the test dataset.
2. Runs all three trained models on every test image.
3. Calculates per-image Dice and IoU.
4. Finds interesting test cases:
       - best overall
       - worst overall
       - largest Attention U-Net improvement
       - largest SegFormer improvement
       - largest model disagreement
5. Generates comparison figures with:
       - Original image
       - Ground truth
       - Model predictions
       - Error maps
       - Prediction overlays

Outputs:
    predictions/
        summary.csv
        selected_cases.txt
        best/
        worst/
        attention_advantage/
        segformer_advantage/
        disagreement/

Usage:

    python predict.py

    python predict.py --threshold 0.5

    python predict.py --output-dir predictions

    python predict.py --num-cases 5
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from configs.training_config import DEVICE
from datasets.dataloaders import get_dataloader
from models.model_factory import create_model


# ==========================================================
# Configuration
# ==========================================================

CHECKPOINTS = {
    "U-Net": Path(
        "checkpoints/unet/best_model.pth"
    ),
    "Attention U-Net": Path(
        "checkpoints/attention_unet/best_model.pth"
    ),
    "SegFormer-B0": Path(
        "checkpoints/segformer/best_model.pth"
    ),
}

MODEL_NAMES = {
    "U-Net": "unet",
    "Attention U-Net": "attention_unet",
    "SegFormer-B0": "segformer",
}


# ==========================================================
# Arguments
# ==========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Generate qualitative segmentation "
            "comparisons."
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Segmentation threshold.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Output directory.",
    )

    parser.add_argument(
        "--num-cases",
        type=int,
        default=5,
        help=(
            "Number of cases to save for each "
            "category."
        ),
    )

    return parser.parse_args()


# ==========================================================
# Checkpoint Loading
# ==========================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load model checkpoint.

    Supports:
        - plain state_dict
        - model_state_dict
        - state_dict

    Also removes a possible 'module.' prefix.
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

            state_dict = checkpoint

    else:

        state_dict = checkpoint

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
            f"Warning: {len(missing)} "
            f"missing checkpoint keys."
        )

    if unexpected:

        print(
            f"Warning: {len(unexpected)} "
            f"unexpected checkpoint keys."
        )

    return model


# ==========================================================
# Image Conversion
# ==========================================================

def tensor_to_image(image):
    """
    Convert image tensor to displayable NumPy image.

    Input:
        (3, H, W)

    Output:
        (H, W, 3)

    The normalization here is only for visualization.
    """

    image = image.detach().cpu()

    image = image.permute(
        1,
        2,
        0,
    ).numpy()

    image_min = image.min()
    image_max = image.max()

    if image_min < 0 or image_max > 1:

        image = (
            image - image_min
        ) / (
            image_max - image_min + 1e-8
        )

    image = np.clip(
        image,
        0,
        1,
    )

    return image


def tensor_to_mask(mask):
    """
    Convert mask tensor to NumPy array.

    Input:
        (1, H, W)

    Output:
        (H, W)
    """

    mask = mask.detach().cpu()

    if mask.ndim == 3:

        mask = mask.squeeze(0)

    return mask.numpy()


# ==========================================================
# Per-Image Metrics
# ==========================================================

def calculate_dice(
    prediction,
    target,
    epsilon=1e-7,
):
    """
    Calculate Dice for one image.
    """

    prediction = prediction.astype(
        np.float32
    )

    target = target.astype(
        np.float32
    )

    intersection = (
        prediction * target
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    # Both masks are empty.
    if denominator == 0:

        return 1.0

    return float(
        (
            2.0 * intersection
            + epsilon
        )
        / (
            denominator
            + epsilon
        )
    )


def calculate_iou(
    prediction,
    target,
    epsilon=1e-7,
):
    """
    Calculate IoU for one image.
    """

    prediction = prediction.astype(
        np.float32
    )

    target = target.astype(
        np.float32
    )

    intersection = (
        prediction * target
    ).sum()

    union = (
        prediction.sum()
        + target.sum()
        - intersection
    )

    # Both masks are empty.
    if union == 0:

        return 1.0

    return float(
        (
            intersection
            + epsilon
        )
        / (
            union
            + epsilon
        )
    )


# ==========================================================
# Error Map
# ==========================================================

def create_error_map(
    prediction,
    target,
):
    """
    Create RGB error map.

    Colors:

        White = True Positive
        Red   = False Positive
        Blue  = False Negative
        Black = True Negative
    """

    prediction = (
        prediction >= 0.5
    )

    target = (
        target >= 0.5
    )

    true_positive = (
        prediction
        & target
    )

    false_positive = (
        prediction
        & ~target
    )

    false_negative = (
        ~prediction
        & target
    )

    true_negative = (
        ~prediction
        & ~target
    )

    height, width = prediction.shape

    error_map = np.zeros(
        (
            height,
            width,
            3,
        ),
        dtype=np.float32,
    )

    # True negative = black
    error_map[
        true_negative
    ] = [0.0, 0.0, 0.0]

    # True positive = white
    error_map[
        true_positive
    ] = [1.0, 1.0, 1.0]

    # False positive = red
    error_map[
        false_positive
    ] = [1.0, 0.0, 0.0]

    # False negative = blue
    error_map[
        false_negative
    ] = [0.0, 0.0, 1.0]

    return error_map


# ==========================================================
# Overlay
# ==========================================================

def create_overlay(
    image,
    prediction,
    target,
):
    """
    Create an overlay showing:

        Ground truth:
            green

        Prediction:
            red

    Areas where both overlap appear yellow.
    """

    overlay = image.copy()

    prediction = (
        prediction >= 0.5
    )

    target = (
        target >= 0.5
    )

    # Slightly darken the original image
    # so masks are easier to see.
    overlay *= 0.65

    # Ground truth = green
    overlay[target, 0] += 0.0
    overlay[target, 1] += 0.35
    overlay[target, 2] += 0.0

    # Prediction = red
    overlay[prediction, 0] += 0.35
    overlay[prediction, 1] += 0.0
    overlay[prediction, 2] += 0.0

    return np.clip(
        overlay,
        0,
        1,
    )


# ==========================================================
# Prediction
# ==========================================================

def predict_batch(
    model,
    images,
    device,
    threshold,
):
    """
    Generate binary predictions for a batch.
    """

    images = images.to(
        device,
        non_blocking=True,
    )

    with torch.no_grad():

        logits = model(images)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= threshold
        ).float()

    return predictions.cpu()


# ==========================================================
# Evaluate One Model
# ==========================================================

def evaluate_model(
    model_name,
    model,
    test_loader,
    device,
    threshold,
):
    """
    Run one model over the complete test set.

    Returns:
        predictions
        dice scores
        IoU scores
    """

    model.eval()

    predictions = {}

    dice_scores = {}

    iou_scores = {}

    image_index = 0

    for images, masks in tqdm(
        test_loader,
        desc=f"Evaluating {model_name}",
    ):

        batch_predictions = predict_batch(
            model,
            images,
            device,
            threshold,
        )

        batch_size = images.size(0)

        for i in range(batch_size):

            prediction = (
                tensor_to_mask(
                    batch_predictions[i]
                )
            )

            target = (
                tensor_to_mask(
                    masks[i]
                )
            )

            target = (
                target >= 0.5
            ).astype(
                np.float32
            )

            prediction = (
                prediction >= 0.5
            ).astype(
                np.float32
            )

            dice = calculate_dice(
                prediction,
                target,
            )

            iou = calculate_iou(
                prediction,
                target,
            )

            predictions[
                image_index
            ] = prediction.astype(
                np.uint8
            )

            dice_scores[
                image_index
            ] = dice

            iou_scores[
                image_index
            ] = iou

            image_index += 1

    return (
        predictions,
        dice_scores,
        iou_scores,
    )


# ==========================================================
# Build Dataset Cache
# ==========================================================

def cache_test_data(
    test_loader,
):
    """
    Cache test images and ground-truth masks.

    This allows us to generate selected figures
    after all models have been evaluated.
    """

    images = {}

    masks = {}

    image_index = 0

    print()
    print("Caching test images...")

    for batch_images, batch_masks in tqdm(
        test_loader,
        desc="Caching",
    ):

        batch_size = (
            batch_images.size(0)
        )

        for i in range(batch_size):

            images[
                image_index
            ] = tensor_to_image(
                batch_images[i]
            )

            masks[
                image_index
            ] = (
                tensor_to_mask(
                    batch_masks[i]
                )
                >= 0.5
            ).astype(
                np.float32
            )

            image_index += 1

    return images, masks


# ==========================================================
# Create Results Table
# ==========================================================

def create_results_table(
    predictions,
    dice_scores,
    iou_scores,
):
    """
    Create per-image comparison table.
    """

    indices = sorted(
        dice_scores["U-Net"].keys()
    )

    rows = []

    for index in indices:

        unet_dice = dice_scores[
            "U-Net"
        ][index]

        attention_dice = dice_scores[
            "Attention U-Net"
        ][index]

        segformer_dice = dice_scores[
            "SegFormer-B0"
        ][index]

        unet_iou = iou_scores[
            "U-Net"
        ][index]

        attention_iou = iou_scores[
            "Attention U-Net"
        ][index]

        segformer_iou = iou_scores[
            "SegFormer-B0"
        ][index]

        rows.append(
            {
                "index": index,

                "unet_dice": unet_dice,
                "attention_unet_dice": attention_dice,
                "segformer_dice": segformer_dice,

                "unet_iou": unet_iou,
                "attention_unet_iou": attention_iou,
                "segformer_iou": segformer_iou,

                "attention_vs_unet": (
                    attention_dice
                    - unet_dice
                ),

                "segformer_vs_unet": (
                    segformer_dice
                    - unet_dice
                ),

                "max_dice": max(
                    unet_dice,
                    attention_dice,
                    segformer_dice,
                ),

                "min_dice": min(
                    unet_dice,
                    attention_dice,
                    segformer_dice,
                ),

                "disagreement": (
                    max(
                        unet_dice,
                        attention_dice,
                        segformer_dice,
                    )
                    -
                    min(
                        unet_dice,
                        attention_dice,
                        segformer_dice,
                    )
                ),
            }
        )

    return rows


# ==========================================================
# Save CSV
# ==========================================================

def save_results_csv(
    rows,
    output_file,
):
    """
    Save per-image results.
    """

    if not rows:

        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(rows)


# ==========================================================
# Select Interesting Cases
# ==========================================================

def select_cases(
    rows,
    num_cases,
):
    """
    Select interesting test images.

    Categories:

        best
        worst
        attention_advantage
        segformer_advantage
        disagreement
    """

    best = sorted(
        rows,
        key=lambda row: row[
            "max_dice"
        ],
        reverse=True,
    )[:num_cases]

    worst = sorted(
        rows,
        key=lambda row: row[
            "min_dice"
        ],
    )[:num_cases]

    attention_advantage = sorted(
        rows,
        key=lambda row: row[
            "attention_vs_unet"
        ],
        reverse=True,
    )[:num_cases]

    segformer_advantage = sorted(
        rows,
        key=lambda row: row[
            "segformer_vs_unet"
        ],
        reverse=True,
    )[:num_cases]

    disagreement = sorted(
        rows,
        key=lambda row: row[
            "disagreement"
        ],
        reverse=True,
    )[:num_cases]

    return {
        "best": best,
        "worst": worst,
        "attention_advantage": (
            attention_advantage
        ),
        "segformer_advantage": (
            segformer_advantage
        ),
        "disagreement": disagreement,
    }


# ==========================================================
# Save Selected Cases
# ==========================================================

def save_selected_cases(
    selected_cases,
    output_file,
):
    """
    Save selected image indices to a text file.
    """

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        for category, cases in (
            selected_cases.items()
        ):

            file.write(
                f"{category.upper()}\n"
            )

            file.write(
                "-" * 60
                + "\n"
            )

            for case in cases:

                file.write(
                    f"Image {case['index']}: "
                    f"U-Net Dice="
                    f"{case['unet_dice']:.4f}, "
                    f"Attention Dice="
                    f"{case['attention_unet_dice']:.4f}, "
                    f"SegFormer Dice="
                    f"{case['segformer_dice']:.4f}\n"
                )

            file.write("\n")


# ==========================================================
# Comparison Figure
# ==========================================================

def create_comparison_figure(
    image,
    target,
    predictions,
    metrics,
    index,
    category,
    output_file,
):
    """
    Create a detailed qualitative comparison.

    Layout:

        Original
        Ground Truth
        U-Net
        Attention U-Net
        SegFormer-B0

    Each prediction also gets an error map and overlay.
    """

    model_names = [
        "U-Net",
        "Attention U-Net",
        "SegFormer-B0",
    ]

    columns = 5

    rows = 3

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            18,
            11,
        ),
    )

    # ------------------------------------------------------
    # Row 1: original + masks
    # ------------------------------------------------------

    axes[0, 0].imshow(
        image
    )

    axes[0, 0].set_title(
        "Original Image"
    )

    axes[0, 0].axis("off")

    axes[0, 1].imshow(
        target,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[0, 1].set_title(
        "Ground Truth"
    )

    axes[0, 1].axis("off")

    for column, model_name in enumerate(
        model_names,
        start=2,
    ):

        prediction = predictions[
            model_name
        ][index]

        axes[0, column].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
        )

        axes[0, column].set_title(
            (
                f"{model_name}\n"
                f"Dice: "
                f"{metrics[model_name]['dice']:.4f}\n"
                f"IoU: "
                f"{metrics[model_name]['iou']:.4f}"
            )
        )

        axes[0, column].axis(
            "off"
        )

    # ------------------------------------------------------
    # Row 2: error maps
    # ------------------------------------------------------

    axes[1, 0].axis("off")

    axes[1, 1].axis("off")

    for column, model_name in enumerate(
        model_names,
        start=2,
    ):

        prediction = predictions[
            model_name
        ][index]

        error_map = create_error_map(
            prediction,
            target,
        )

        axes[1, column].imshow(
            error_map
        )

        axes[1, column].set_title(
            (
                f"{model_name} Error Map\n"
                "White=TP  Red=FP  Blue=FN"
            )
        )

        axes[1, column].axis(
            "off"
        )

    # ------------------------------------------------------
    # Row 3: overlays
    # ------------------------------------------------------

    axes[2, 0].imshow(
        image
    )

    axes[2, 0].set_title(
        "Original"
    )

    axes[2, 0].axis("off")

    axes[2, 1].imshow(
        target,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2, 1].set_title(
        "Ground Truth"
    )

    axes[2, 1].axis("off")

    for column, model_name in enumerate(
        model_names,
        start=2,
    ):

        prediction = predictions[
            model_name
        ][index]

        overlay = create_overlay(
            image,
            prediction,
            target,
        )

        axes[2, column].imshow(
            overlay
        )

        axes[2, column].set_title(
            (
                f"{model_name} Overlay"
            )
        )

        axes[2, column].axis(
            "off"
        )

    # ------------------------------------------------------
    # Figure title
    # ------------------------------------------------------

    fig.suptitle(
        (
            f"ISIC Segmentation Comparison — "
            f"Test Image {index} — "
            f"{category.replace('_', ' ').title()}"
        ),
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ==========================================================
# Generate Selected Figures
# ==========================================================

def generate_selected_figures(
    selected_cases,
    images,
    masks,
    predictions,
    dice_scores,
    iou_scores,
    output_dir,
):
    """
    Generate figures for all selected cases.
    """

    generated = set()

    for category, cases in (
        selected_cases.items()
    ):

        category_dir = (
            output_dir / category
        )

        category_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for case in cases:

            index = case["index"]

            # Avoid generating the same figure
            # twice within one category.
            key = (
                category,
                index,
            )

            if key in generated:

                continue

            generated.add(key)

            metrics = {}

            for model_name in MODEL_NAMES:

                metrics[
                    model_name
                ] = {
                    "dice": dice_scores[
                        model_name
                    ][index],

                    "iou": iou_scores[
                        model_name
                    ][index],
                }

            output_file = (
                category_dir
                / f"{index:04d}_comparison.png"
            )

            create_comparison_figure(
                image=images[index],
                target=masks[index],
                predictions=predictions,
                metrics=metrics,
                index=index,
                category=category,
                output_file=output_file,
            )


# ==========================================================
# Print Summary
# ==========================================================

def print_summary(
    rows,
):
    """
    Print overall per-image averages.
    """

    print()
    print("=" * 80)
    print("PER-IMAGE SUMMARY")
    print("=" * 80)

    for model_name, dice_key, iou_key in [
        (
            "U-Net",
            "unet_dice",
            "unet_iou",
        ),
        (
            "Attention U-Net",
            "attention_unet_dice",
            "attention_unet_iou",
        ),
        (
            "SegFormer-B0",
            "segformer_dice",
            "segformer_iou",
        ),
    ]:

        dice_mean = np.mean(
            [
                row[dice_key]
                for row in rows
            ]
        )

        iou_mean = np.mean(
            [
                row[iou_key]
                for row in rows
            ]
        )

        print(
            f"{model_name:<20}"
            f"Dice: {dice_mean:.4f}    "
            f"IoU: {iou_mean:.4f}"
        )

    print("=" * 80)


# ==========================================================
# Main
# ==========================================================

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        DEVICE
    )

    print("=" * 80)
    print("QUALITATIVE SEGMENTATION EVALUATION")
    print("=" * 80)

    print(
        f"Device    : {device}"
    )

    if device.type == "cuda":

        print(
            f"GPU       : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Threshold : {args.threshold}"
    )

    # ------------------------------------------------------
    # Test loader
    # ------------------------------------------------------

    test_loader = get_dataloader(
        split="test"
    )

    test_size = len(
        test_loader.dataset
    )

    print(
        f"Test size : {test_size}"
    )

    # ------------------------------------------------------
    # Cache test data
    # ------------------------------------------------------

    images, masks = cache_test_data(
        test_loader
    )

    # ------------------------------------------------------
    # Store predictions and metrics
    # ------------------------------------------------------

    predictions = {}

    dice_scores = {}

    iou_scores = {}

    # ------------------------------------------------------
    # Evaluate models one at a time
    # ------------------------------------------------------

    for model_name, model_type in (
        MODEL_NAMES.items()
    ):

        print()
        print("=" * 80)
        print(
            f"MODEL: {model_name}"
        )
        print("=" * 80)

        checkpoint_path = CHECKPOINTS[
            model_name
        ]

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                (
                    f"Checkpoint not found:\n"
                    f"{checkpoint_path}"
                )
            )

        print(
            f"Checkpoint: "
            f"{checkpoint_path}"
        )

        model = create_model(
            model_type
        )

        model = model.to(
            device
        )

        model = load_checkpoint(
            model,
            checkpoint_path,
            device,
        )

        (
            model_predictions,
            model_dice,
            model_iou,
        ) = evaluate_model(
            model_name=model_name,
            model=model,
            test_loader=test_loader,
            device=device,
            threshold=args.threshold,
        )

        predictions[
            model_name
        ] = model_predictions

        dice_scores[
            model_name
        ] = model_dice

        iou_scores[
            model_name
        ] = model_iou

        # Free GPU memory before loading
        # the next model.
        del model

        if device.type == "cuda":

            torch.cuda.empty_cache()

        print(
            f"{model_name} evaluation complete."
        )

    # ------------------------------------------------------
    # Create results table
    # ------------------------------------------------------

    rows = create_results_table(
        predictions,
        dice_scores,
        iou_scores,
    )

    # ------------------------------------------------------
    # Save CSV
    # ------------------------------------------------------

    csv_file = (
        output_dir
        / "summary.csv"
    )

    save_results_csv(
        rows,
        csv_file,
    )

    print(
        f"\nPer-image results saved to:"
        f" {csv_file}"
    )

    # ------------------------------------------------------
    # Print summary
    # ------------------------------------------------------

    print_summary(
        rows
    )

    # ------------------------------------------------------
    # Select cases
    # ------------------------------------------------------

    selected_cases = select_cases(
        rows,
        args.num_cases,
    )

    # ------------------------------------------------------
    # Save selected cases
    # ------------------------------------------------------

    selected_file = (
        output_dir
        / "selected_cases.txt"
    )

    save_selected_cases(
        selected_cases,
        selected_file,
    )

    print(
        f"Selected cases saved to:"
        f" {selected_file}"
    )

    # ------------------------------------------------------
    # Print selected indices
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("SELECTED CASES")
    print("=" * 80)

    for category, cases in (
        selected_cases.items()
    ):

        print()
        print(
            category.upper()
        )

        for case in cases:

            print(
                f"  Image {case['index']:3d} | "
                f"U-Net={case['unet_dice']:.4f} | "
                f"Attention="
                f"{case['attention_unet_dice']:.4f} | "
                f"SegFormer="
                f"{case['segformer_dice']:.4f}"
            )

    # ------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("GENERATING QUALITATIVE FIGURES")
    print("=" * 80)

    generate_selected_figures(
        selected_cases=selected_cases,
        images=images,
        masks=masks,
        predictions=predictions,
        dice_scores=dice_scores,
        iou_scores=iou_scores,
        output_dir=output_dir,
    )

    print()
    print("=" * 80)
    print("PREDICTION ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Results directory: "
        f"{output_dir}"
    )

    print(
        f"CSV: "
        f"{csv_file}"
    )

    print(
        f"Selected cases: "
        f"{selected_file}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()