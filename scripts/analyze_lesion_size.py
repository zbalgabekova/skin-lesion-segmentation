"""
analyze_lesion_size.py

Analyze segmentation performance according to lesion size.

This script uses the files produced by the current project pipeline:

    predictions/summary.csv
    splits/test.csv
    data/ISIC2018_Task1_Training_GroundTruth/

The prediction CSV produced by predict.py contains:
    index
    unet_dice
    attention_unet_dice
    segformer_dice
    unet_iou
    attention_unet_iou
    segformer_iou

The "index" column is mapped to the corresponding row in splits/test.csv.

Default usage:
    python analyze_lesion_size.py

Optional:
    python analyze_lesion_size.py --small-threshold 5 --large-threshold 20
    python analyze_lesion_size.py --no-plots
"""

from pathlib import Path
import argparse
import json
import re

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import matplotlib.pyplot as plt


# ============================================================
# DEFAULT PATHS
# ============================================================

TEST_CSV = Path("splits/test.csv")
PREDICTIONS_CSV = Path("predictions/summary.csv")
MASK_DIR = Path("data/ISIC2018_Task1_Training_GroundTruth")
OUTPUT_DIR = Path("evaluation/lesion_size")


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "unet": {
        "name": "U-Net",
        "dice": "unet_dice",
        "iou": "unet_iou",
    },
    "attention_unet": {
        "name": "Attention U-Net",
        "dice": "attention_unet_dice",
        "iou": "attention_unet_iou",
    },
    "segformer": {
        "name": "SegFormer-B0",
        "dice": "segformer_dice",
        "iou": "segformer_iou",
    },
}


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze segmentation performance by lesion size."
    )

    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV,
        help="Path to splits/test.csv",
    )

    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=PREDICTIONS_CSV,
        help="Path to predictions/summary.csv",
    )

    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=MASK_DIR,
        help="Directory containing ground-truth masks.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for analysis results.",
    )

    parser.add_argument(
        "--small-threshold",
        type=float,
        default=5.0,
        help="Lesion area below this percentage is Small.",
    )

    parser.add_argument(
        "--large-threshold",
        type=float,
        default=20.0,
        help="Lesion area above this percentage is Large.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not generate plots.",
    )

    return parser.parse_args()


# ============================================================
# HELPERS
# ============================================================

def normalize_column(name):
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(name).lower().strip(),
    ).strip("_")


def find_image_column(df):
    """
    Automatically find the image-ID column in test.csv.
    """
    preferred = [
        "image_id",
        "image",
        "filename",
        "file_name",
        "id",
    ]

    normalized = {
        normalize_column(c): c
        for c in df.columns
    }

    for candidate in preferred:
        if candidate in normalized:
            return normalized[candidate]

    # Conservative fallback: a column containing image/file/id.
    for column in df.columns:
        name = normalize_column(column)
        if (
            "image" in name
            or "filename" in name
            or name == "id"
        ):
            return column

    raise ValueError(
        "Could not find image-ID column in test.csv.\n"
        f"Available columns: {list(df.columns)}"
    )


def clean_image_id(value):
    """
    Convert an entry from test.csv to the ISIC image ID.
    """
    value = str(value).strip()

    # Remove directory and extension.
    value = Path(value).stem

    # Ground-truth masks normally use *_segmentation.
    if value.endswith("_segmentation"):
        value = value[:-len("_segmentation")]

    return value


def find_mask(mask_dir, image_id):
    """
    Find the ground-truth mask for an image.
    """
    candidates = [
        mask_dir / f"{image_id}_segmentation.png",
        mask_dir / f"{image_id}.png",
        mask_dir / f"{image_id}_segmentation.jpg",
        mask_dir / f"{image_id}.jpg",
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def lesion_area_percent(mask_path):
    """
    Calculate lesion area as a percentage of the image.

    Ground-truth foreground pixels are all pixels > 0.
    """
    mask = np.asarray(
        Image.open(mask_path).convert("L")
    )

    foreground = mask > 0

    total_pixels = foreground.size
    lesion_pixels = int(foreground.sum())

    if total_pixels == 0:
        raise ValueError(
            f"Empty mask: {mask_path}"
        )

    return (
        lesion_pixels,
        total_pixels,
        100.0 * lesion_pixels / total_pixels,
    )


def size_group(area, small_threshold, large_threshold):
    if area < small_threshold:
        return "Small"

    if area <= large_threshold:
        return "Medium"

    return "Large"


# ============================================================
# LOAD DATA
# ============================================================

def load_test_ids(test_csv):
    test_df = pd.read_csv(test_csv)

    image_column = find_image_column(test_df)

    image_ids = [
        clean_image_id(value)
        for value in test_df[image_column]
    ]

    return test_df, image_column, image_ids


def load_predictions(predictions_csv):
    predictions = pd.read_csv(predictions_csv)

    required = [
        "index",
        "unet_dice",
        "attention_unet_dice",
        "segformer_dice",
        "unet_iou",
        "attention_unet_iou",
        "segformer_iou",
    ]

    missing = [
        column
        for column in required
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            "predictions/summary.csv is missing columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n{list(predictions.columns)}"
        )

    predictions["index"] = pd.to_numeric(
        predictions["index"],
        errors="raise",
    ).astype(int)

    if predictions["index"].duplicated().any():
        raise ValueError(
            "Duplicate values found in predictions.csv 'index' column."
        )

    return predictions


# ============================================================
# CREATE PER-IMAGE DATA
# ============================================================

def create_per_image_results(
    test_ids,
    predictions,
    mask_dir,
    small_threshold,
    large_threshold,
):
    prediction_map = predictions.set_index("index")

    rows = []
    missing_masks = []
    missing_predictions = []

    for index, image_id in tqdm(
        enumerate(test_ids),
        total=len(test_ids),
        desc="Analyzing test images",
    ):
        if index not in prediction_map.index:
            missing_predictions.append(index)
            continue

        mask_path = find_mask(
            mask_dir,
            image_id,
        )

        if mask_path is None:
            missing_masks.append(image_id)
            continue

        (
            lesion_pixels,
            image_pixels,
            area_percent,
        ) = lesion_area_percent(mask_path)

        row = {
            "index": index,
            "image_id": image_id,
            "lesion_pixels": lesion_pixels,
            "image_pixels": image_pixels,
            "lesion_area_percent": area_percent,
            "size_group": size_group(
                area_percent,
                small_threshold,
                large_threshold,
            ),
        }

        prediction_row = prediction_map.loc[index]

        for model in MODELS.values():
            row[model["dice"]] = float(
                prediction_row[model["dice"]]
            )
            row[model["iou"]] = float(
                prediction_row[model["iou"]]
            )

        rows.append(row)

    if not rows:
        raise RuntimeError(
            "No images could be analyzed."
        )

    results = pd.DataFrame(rows)

    print()
    print("=" * 72)
    print("DATASET CHECK")
    print("=" * 72)
    print(f"Test images in test.csv : {len(test_ids)}")
    print(f"Prediction rows         : {len(predictions)}")
    print(f"Images analyzed         : {len(results)}")
    print(f"Missing masks           : {len(missing_masks)}")
    print(f"Missing predictions     : {len(missing_predictions)}")
    print("=" * 72)

    if missing_masks:
        print("\nFirst missing masks:")
        for image_id in missing_masks[:10]:
            print(f"  {image_id}")

    if missing_predictions:
        print("\nFirst missing prediction indices:")
        for index in missing_predictions[:10]:
            print(f"  {index}")

    return results


# ============================================================
# SUMMARY
# ============================================================

def make_summary(results):
    rows = []

    for group in ["Small", "Medium", "Large"]:
        subset = results[
            results["size_group"] == group
        ]

        for model in MODELS.values():
            dice = pd.to_numeric(
                subset[model["dice"]],
                errors="coerce",
            )

            iou = pd.to_numeric(
                subset[model["iou"]],
                errors="coerce",
            )

            rows.append({
                "size_group": group,
                "model": model["name"],
                "n_images": len(subset),

                "dice_mean": dice.mean(),
                "dice_std": dice.std(),
                "dice_median": dice.median(),

                "iou_mean": iou.mean(),
                "iou_std": iou.std(),
                "iou_median": iou.median(),
            })

    return pd.DataFrame(rows)


def calculate_correlations(results):
    rows = []

    lesion_size = results[
        "lesion_area_percent"
    ]

    for model in MODELS.values():
        dice = results[
            model["dice"]
        ]

        iou = results[
            model["iou"]
        ]

        rows.append({
            "model": model["name"],
            "lesion_size_vs_dice": lesion_size.corr(dice),
            "lesion_size_vs_iou": lesion_size.corr(iou),
        })

    return pd.DataFrame(rows)


# ============================================================
# PRINT RESULTS
# ============================================================

def print_distribution(results):
    print()
    print("=" * 72)
    print("LESION SIZE DISTRIBUTION")
    print("=" * 72)

    print(
        f"Mean lesion area   : "
        f"{results['lesion_area_percent'].mean():.2f}%"
    )

    print(
        f"Median lesion area : "
        f"{results['lesion_area_percent'].median():.2f}%"
    )

    print(
        f"Minimum            : "
        f"{results['lesion_area_percent'].min():.2f}%"
    )

    print(
        f"Maximum            : "
        f"{results['lesion_area_percent'].max():.2f}%"
    )

    print()

    counts = (
        results["size_group"]
        .value_counts()
        .reindex(["Small", "Medium", "Large"])
        .fillna(0)
        .astype(int)
    )

    for group, count in counts.items():
        percentage = 100.0 * count / len(results)

        print(
            f"{group:<10}: "
            f"{count:>4} images "
            f"({percentage:5.1f}%)"
        )

    print("=" * 72)


def print_performance(summary):
    print()
    print("=" * 72)
    print("PERFORMANCE BY LESION SIZE")
    print("=" * 72)

    for group in ["Small", "Medium", "Large"]:
        print(f"\n{group.upper()}")

        subset = summary[
            summary["size_group"] == group
        ]

        if subset.empty:
            print("  No images.")
            continue

        for _, row in subset.iterrows():
            print(
                f"  {row['model']:<18} "
                f"Dice: {row['dice_mean']:.4f} "
                f"+/- {row['dice_std']:.4f} | "
                f"IoU: {row['iou_mean']:.4f} "
                f"+/- {row['iou_std']:.4f} | "
                f"N: {int(row['n_images'])}"
            )

    print("=" * 72)


def print_best_models(summary):
    print()
    print("=" * 72)
    print("BEST MODEL BY LESION SIZE")
    print("=" * 72)

    for group in ["Small", "Medium", "Large"]:
        subset = summary[
            summary["size_group"] == group
        ]

        if subset.empty:
            continue

        best_dice = subset.loc[
            subset["dice_mean"].idxmax()
        ]

        best_iou = subset.loc[
            subset["iou_mean"].idxmax()
        ]

        print(
            f"{group:<10} "
            f"Dice: {best_dice['model']} "
            f"({best_dice['dice_mean']:.4f}) | "
            f"IoU: {best_iou['model']} "
            f"({best_iou['iou_mean']:.4f})"
        )

    print("=" * 72)


def print_correlations(correlations):
    print()
    print("=" * 72)
    print("LESION SIZE vs MODEL PERFORMANCE")
    print("=" * 72)

    for _, row in correlations.iterrows():
        print(
            f"{row['model']:<18} "
            f"Dice correlation: "
            f"{row['lesion_size_vs_dice']:+.4f} | "
            f"IoU correlation: "
            f"{row['lesion_size_vs_iou']:+.4f}"
        )

    print("=" * 72)


# ============================================================
# PLOTS
# ============================================================

def plot_distribution(results, output):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(
        results["lesion_area_percent"],
        bins=30,
    )

    ax.set_xlabel(
        "Lesion area (% of image)"
    )
    ax.set_ylabel(
        "Number of test images"
    )
    ax.set_title(
        "ISIC Test Set — Lesion Size Distribution"
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_by_size(
    summary,
    metric,
    ylabel,
    output,
):
    groups = ["Small", "Medium", "Large"]
    x = np.arange(len(groups))

    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS.values():
        subset = (
            summary[
                summary["model"] == model["name"]
            ]
            .set_index("size_group")
            .reindex(groups)
        )

        means = subset[
            f"{metric}_mean"
        ].to_numpy(dtype=float)

        stds = (
            subset[f"{metric}_std"]
            .fillna(0)
            .to_numpy(dtype=float)
        )

        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            linewidth=2,
            capsize=4,
            label=model["name"],
        )

    ax.set_xticks(x, groups)
    ax.set_xlabel("Lesion Size Group")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{ylabel} by Lesion Size"
    )
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_metric_vs_size(
    results,
    metric_key,
    ylabel,
    output,
):
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in MODELS.values():
        ax.scatter(
            results["lesion_area_percent"],
            results[metric_key(model)],
            s=18,
            alpha=0.45,
            label=model["name"],
        )

    ax.set_xlabel(
        "Lesion area (% of image)"
    )
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{ylabel} vs. Lesion Size"
    )
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# JSON
# ============================================================

def make_json_report(
    results,
    summary,
    correlations,
    small_threshold,
    large_threshold,
):
    report = {
        "test_images_analyzed": int(len(results)),

        "lesion_size_definition": {
            "small": f"< {small_threshold}%",
            "medium": (
                f"{small_threshold}% - "
                f"{large_threshold}%"
            ),
            "large": f"> {large_threshold}%",
        },

        "lesion_area_percent": {
            "mean": float(
                results["lesion_area_percent"].mean()
            ),
            "median": float(
                results["lesion_area_percent"].median()
            ),
            "min": float(
                results["lesion_area_percent"].min()
            ),
            "max": float(
                results["lesion_area_percent"].max()
            ),
        },

        "group_counts": {
            group: int(
                (
                    results["size_group"] == group
                ).sum()
            )
            for group in ["Small", "Medium", "Large"]
        },

        "correlations": (
            correlations.to_dict(
                orient="records"
            )
        ),

        "performance_by_size": (
            summary.to_dict(
                orient="records"
            )
        ),
    }

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    if args.small_threshold <= 0:
        raise ValueError(
            "--small-threshold must be > 0."
        )

    if args.large_threshold <= args.small_threshold:
        raise ValueError(
            "--large-threshold must be greater than "
            "--small-threshold."
        )

    print("=" * 72)
    print("ISIC LESION SIZE ANALYSIS")
    print("=" * 72)
    print(f"Test CSV       : {args.test_csv}")
    print(f"Predictions    : {args.predictions_csv}")
    print(f"Ground truth   : {args.mask_dir}")
    print(f"Output         : {args.output_dir}")
    print(
        f"Thresholds     : "
        f"Small < {args.small_threshold:g}% | "
        f"Medium <= {args.large_threshold:g}% | "
        f"Large > {args.large_threshold:g}%"
    )
    print("=" * 72)

    # --------------------------------------------------------
    # Check paths
    # --------------------------------------------------------

    if not args.test_csv.exists():
        raise FileNotFoundError(
            f"Test CSV not found:\n{args.test_csv}"
        )

    if not args.predictions_csv.exists():
        raise FileNotFoundError(
            f"Predictions CSV not found:\n"
            f"{args.predictions_csv}\n\n"
            "Run predict.py first."
        )

    if not args.mask_dir.exists():
        raise FileNotFoundError(
            f"Ground-truth mask directory not found:\n"
            f"{args.mask_dir}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    test_df, image_column, test_ids = load_test_ids(
        args.test_csv
    )

    predictions = load_predictions(
        args.predictions_csv
    )

    print()
    print(f"Test image column : {image_column}")
    print(f"Test images       : {len(test_ids)}")
    print(
        f"Prediction rows   : {len(predictions)}"
    )

    # --------------------------------------------------------
    # Analyze
    # --------------------------------------------------------

    results = create_per_image_results(
        test_ids=test_ids,
        predictions=predictions,
        mask_dir=args.mask_dir,
        small_threshold=args.small_threshold,
        large_threshold=args.large_threshold,
    )

    summary = make_summary(results)

    correlations = calculate_correlations(
        results
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_distribution(results)
    print_performance(summary)
    print_best_models(summary)
    print_correlations(correlations)

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    per_image_path = (
        args.output_dir
        / "lesion_size_results.csv"
    )

    summary_path = (
        args.output_dir
        / "metrics_by_size.csv"
    )

    correlation_path = (
        args.output_dir
        / "lesion_size_correlations.csv"
    )

    results.to_csv(
        per_image_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    correlations.to_csv(
        correlation_path,
        index=False,
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    if not args.no_plots:
        plot_distribution(
            results,
            args.output_dir
            / "lesion_size_distribution.png",
        )

        plot_metric_by_size(
            summary,
            "dice",
            "Dice Score",
            args.output_dir
            / "dice_by_lesion_size.png",
        )

        plot_metric_by_size(
            summary,
            "iou",
            "IoU",
            args.output_dir
            / "iou_by_lesion_size.png",
        )

        # Explicit metric columns for plotting.
        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for model in MODELS.values():
            ax.scatter(
                results["lesion_area_percent"],
                results[model["dice"]],
                s=18,
                alpha=0.45,
                label=model["name"],
            )

        ax.set_xlabel(
            "Lesion area (% of image)"
        )
        ax.set_ylabel("Dice Score")
        ax.set_title(
            "Dice Score vs. Lesion Size"
        )
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            args.output_dir
            / "dice_vs_lesion_size.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(10, 6)
        )

        for model in MODELS.values():
            ax.scatter(
                results["lesion_area_percent"],
                results[model["iou"]],
                s=18,
                alpha=0.45,
                label=model["name"],
            )

        ax.set_xlabel(
            "Lesion area (% of image)"
        )
        ax.set_ylabel("IoU")
        ax.set_title(
            "IoU vs. Lesion Size"
        )
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            args.output_dir
            / "iou_vs_lesion_size.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    report = make_json_report(
        results=results,
        summary=summary,
        correlations=correlations,
        small_threshold=args.small_threshold,
        large_threshold=args.large_threshold,
    )

    json_path = (
        args.output_dir
        / "lesion_size_summary.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Per-image results : {per_image_path}")
    print(f"Grouped results   : {summary_path}")
    print(f"Correlations      : {correlation_path}")
    print(f"JSON report       : {json_path}")

    if not args.no_plots:
        print(
            f"Plots             : "
            f"{args.output_dir}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()
