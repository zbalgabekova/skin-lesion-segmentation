"""
Compare segmentation models using saved evaluation results.

Expected files:

    evaluation/
        unet_results.json
        attention_unet_results.json
        segformer_results.json

Outputs:

    evaluation/
        comparison.csv
        comparison_summary.txt
        dice_comparison.png
        iou_comparison.png
        precision_comparison.png
        recall_comparison.png
        f1_comparison.png
        parameters_comparison.png
        inference_speed_comparison.png
        gpu_memory_comparison.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

EVALUATION_DIR = Path("evaluation")

MODEL_FILES = {
    "U-Net": "unet_results.json",
    "Attention U-Net": "attention_unet_results.json",
    "SegFormer-B0": "segformer_results.json",
}


# Current parameter counts from your trained models.
#
# These are used because the current evaluate.py JSON files
# do not store the parameter count.
PARAMETERS = {
    "U-Net": 24_436_369,
    "Attention U-Net": 31_783_633,
    "SegFormer-B0": 3_714_401,
}


# Metrics where higher is better
HIGHER_IS_BETTER = [
    "dice",
    "iou",
    "precision",
    "recall",
    "f1",
    "pixel_accuracy",
    "inference_fps",
]


# ==========================================================
# Load Results
# ==========================================================

def load_results():
    """
    Load evaluation JSON files.
    """

    results = []

    for model_name, filename in MODEL_FILES.items():

        filepath = EVALUATION_DIR / filename

        if not filepath.exists():

            raise FileNotFoundError(
                f"Could not find evaluation file:\n"
                f"{filepath}"
            )

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        data["model"] = model_name

        data["parameters"] = PARAMETERS[
            model_name
        ]

        results.append(data)

    return results


# ==========================================================
# Create DataFrame
# ==========================================================

def create_dataframe(results):
    """
    Convert results into a pandas DataFrame.
    """

    rows = []

    for result in results:

        rows.append(
            {
                "Model": result["model"],
                "Parameters": result["parameters"],
                "Parameters (M)": (
                    result["parameters"] / 1_000_000
                ),
                "Dice": result["dice"],
                "IoU": result["iou"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "Pixel Accuracy": result[
                    "pixel_accuracy"
                ],
                "Inference Time (sec)": result[
                    "total_inference_time_sec"
                ],
                "Time / Image (sec)": result[
                    "inference_time_per_image_sec"
                ],
                "FPS": result[
                    "inference_fps"
                ],
                "Peak GPU Memory (MB)": result[
                    "peak_gpu_memory_mb"
                ],
                "Test Images": result[
                    "num_images"
                ],
            }
        )

    return pd.DataFrame(rows)


# ==========================================================
# Print Comparison
# ==========================================================

def print_comparison(df):
    """
    Print main comparison table.
    """

    print()
    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    display_columns = [
        "Model",
        "Parameters (M)",
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "F1",
        "FPS",
        "Peak GPU Memory (MB)",
    ]

    display_df = df[display_columns].copy()

    display_df["Parameters (M)"] = (
        display_df["Parameters (M)"].map(
            lambda x: f"{x:.2f}"
        )
    )

    for column in [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "F1",
    ]:

        display_df[column] = (
            display_df[column].map(
                lambda x: f"{x:.4f}"
            )
        )

    display_df["FPS"] = (
        display_df["FPS"].map(
            lambda x: f"{x:.2f}"
        )
    )

    display_df["Peak GPU Memory (MB)"] = (
        display_df[
            "Peak GPU Memory (MB)"
        ].map(
            lambda x: f"{x:.1f}"
        )
    )

    print(
        display_df.to_string(
            index=False
        )
    )

    print("=" * 100)


# ==========================================================
# Find Best Models
# ==========================================================

def find_best_models(df):
    """
    Find the best model for each metric.
    """

    metric_columns = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "F1",
        "Pixel Accuracy",
        "FPS",
    ]

    best_models = {}

    for metric in metric_columns:

        index = df[metric].idxmax()

        best_models[metric] = (
            df.loc[index, "Model"]
        )

    return best_models


# ==========================================================
# Print Best Models
# ==========================================================

def print_best_models(df):
    """
    Print the best model for each metric.
    """

    best_models = find_best_models(df)

    print()
    print("=" * 100)
    print("BEST MODEL BY METRIC")
    print("=" * 100)

    for metric, model in best_models.items():

        value = df.loc[
            df["Model"] == model,
            metric,
        ].iloc[0]

        print(
            f"{metric:<20}: "
            f"{model:<20} "
            f"({value:.4f})"
        )

    print("=" * 100)


# ==========================================================
# Accuracy Improvements
# ==========================================================

def calculate_improvements(df):
    """
    Calculate improvements relative to U-Net.
    """

    baseline = df[
        df["Model"] == "U-Net"
    ].iloc[0]

    print()
    print("=" * 100)
    print("IMPROVEMENT RELATIVE TO U-NET")
    print("=" * 100)

    for _, row in df.iterrows():

        if row["Model"] == "U-Net":
            continue

        dice_difference = (
            row["Dice"]
            - baseline["Dice"]
        )

        iou_difference = (
            row["IoU"]
            - baseline["IoU"]
        )

        fps_difference = (
            row["FPS"]
            - baseline["FPS"]
        )

        parameter_ratio = (
            baseline["Parameters"]
            / row["Parameters"]
        )

        print()
        print(row["Model"])

        print(
            f"  Dice change : "
            f"{dice_difference:+.4f}"
        )

        print(
            f"  IoU change  : "
            f"{iou_difference:+.4f}"
        )

        print(
            f"  FPS change  : "
            f"{fps_difference:+.2f}"
        )

        print(
            f"  Parameter reduction/ratio: "
            f"{parameter_ratio:.2f}x"
        )

    print()
    print("=" * 100)


# ==========================================================
# Save CSV
# ==========================================================

def save_csv(df):
    """
    Save complete comparison to CSV.
    """

    output_file = (
        EVALUATION_DIR
        / "comparison.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nCSV saved to: {output_file}"
    )


# ==========================================================
# Save Summary
# ==========================================================

def save_summary(df):
    """
    Save human-readable summary.
    """

    output_file = (
        EVALUATION_DIR
        / "comparison_summary.txt"
    )

    best_models = find_best_models(df)

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "MODEL COMPARISON\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "Test set size: "
            f"{int(df['Test Images'].iloc[0])}\n\n"
        )

        # --------------------------------------------------
        # Main results
        # --------------------------------------------------

        file.write(
            "MAIN RESULTS\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for _, row in df.iterrows():

            file.write(
                f"\n{row['Model']}\n"
            )

            file.write(
                f"  Parameters: "
                f"{row['Parameters']:,}\n"
            )

            file.write(
                f"  Dice: "
                f"{row['Dice']:.4f}\n"
            )

            file.write(
                f"  IoU: "
                f"{row['IoU']:.4f}\n"
            )

            file.write(
                f"  Precision: "
                f"{row['Precision']:.4f}\n"
            )

            file.write(
                f"  Recall: "
                f"{row['Recall']:.4f}\n"
            )

            file.write(
                f"  F1: "
                f"{row['F1']:.4f}\n"
            )

            file.write(
                f"  Pixel Accuracy: "
                f"{row['Pixel Accuracy']:.4f}\n"
            )

            file.write(
                f"  FPS: "
                f"{row['FPS']:.2f}\n"
            )

            file.write(
                f"  Time/Image: "
                f"{row['Time / Image (sec)']:.4f} sec\n"
            )

            file.write(
                f"  Peak GPU Memory: "
                f"{row['Peak GPU Memory (MB)']:.1f} MB\n"
            )

        # --------------------------------------------------
        # Best models
        # --------------------------------------------------

        file.write(
            "\n\nBEST MODEL BY METRIC\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for metric, model in best_models.items():

            value = df.loc[
                df["Model"] == model,
                metric,
            ].iloc[0]

            file.write(
                f"{metric}: "
                f"{model} "
                f"({value:.4f})\n"
            )

    print(
        f"Summary saved to: {output_file}"
    )


# ==========================================================
# Plot Helper
# ==========================================================

def create_bar_plot(
    df,
    column,
    title,
    ylabel,
    filename,
    format_as_percentage=False,
):
    """
    Create and save a bar plot.
    """

    plt.figure(
        figsize=(8, 5)
    )

    bars = plt.bar(
        df["Model"],
        df[column],
    )

    plt.title(title)

    plt.ylabel(ylabel)

    plt.xticks(
        rotation=15
    )

    # Add values above bars
    for bar, value in zip(
        bars,
        df[column],
    ):

        if format_as_percentage:

            text = (
                f"{value * 100:.2f}%"
            )

        else:

            text = f"{value:.3f}"

        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            text,
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_file = (
        EVALUATION_DIR
        / filename
    )

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Plot saved to: {output_file}"
    )


# ==========================================================
# Create All Plots
# ==========================================================

def create_plots(df):
    """
    Create comparison plots.
    """

    # ------------------------------------------------------
    # Accuracy
    # ------------------------------------------------------

    create_bar_plot(
        df,
        "Dice",
        "Dice Score Comparison",
        "Dice Score",
        "dice_comparison.png",
    )

    create_bar_plot(
        df,
        "IoU",
        "IoU Comparison",
        "IoU",
        "iou_comparison.png",
    )

    create_bar_plot(
        df,
        "Precision",
        "Precision Comparison",
        "Precision",
        "precision_comparison.png",
    )

    create_bar_plot(
        df,
        "Recall",
        "Recall Comparison",
        "Recall",
        "recall_comparison.png",
    )

    create_bar_plot(
        df,
        "F1",
        "F1 Score Comparison",
        "F1 Score",
        "f1_comparison.png",
    )

    # ------------------------------------------------------
    # Efficiency
    # ------------------------------------------------------

    create_bar_plot(
        df,
        "Parameters (M)",
        "Model Parameter Comparison",
        "Parameters (Millions)",
        "parameters_comparison.png",
    )

    create_bar_plot(
        df,
        "FPS",
        "Inference Speed Comparison",
        "Frames per Second",
        "inference_speed_comparison.png",
    )

    create_bar_plot(
        df,
        "Peak GPU Memory (MB)",
        "Peak GPU Memory Comparison",
        "Memory (MB)",
        "gpu_memory_comparison.png",
    )


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 100)
    print("MODEL COMPARISON")
    print("=" * 100)

    # ------------------------------------------------------
    # Check directory
    # ------------------------------------------------------

    if not EVALUATION_DIR.exists():

        raise FileNotFoundError(
            f"Evaluation directory not found: "
            f"{EVALUATION_DIR}"
        )

    # ------------------------------------------------------
    # Load results
    # ------------------------------------------------------

    results = load_results()

    # ------------------------------------------------------
    # Create DataFrame
    # ------------------------------------------------------

    df = create_dataframe(
        results
    )

    # ------------------------------------------------------
    # Print results
    # ------------------------------------------------------

    print_comparison(df)

    print_best_models(df)

    calculate_improvements(df)

    # ------------------------------------------------------
    # Save results
    # ------------------------------------------------------

    save_csv(df)

    save_summary(df)

    # ------------------------------------------------------
    # Create plots
    # ------------------------------------------------------

    print()
    print("Creating plots...")

    create_plots(df)

    print()
    print("=" * 100)
    print("COMPARISON COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()