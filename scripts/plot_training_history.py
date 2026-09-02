"""
Plot training histories for U-Net, Attention U-Net, and SegFormer-B0.

Expected project structure:

Project2/
├── checkpoints/
│   ├── unet/
│   │   └── training_history.csv
│   ├── attention_unet/
│   │   └── training_history.csv
│   └── segformer/
│       └── training_history.csv
└── plot_training_history.py

The script creates comparable plots for:
    1. Loss
    2. Dice
    3. IoU
    4. Precision
    5. Recall
    6. Learning rate

It also prints the best validation epoch for each model.

Usage:
    python plot_training_history.py

Optional:
    python plot_training_history.py --output-dir plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ==========================================================
# Configuration
# ==========================================================

MODEL_DIRS = {
    "U-Net": Path("checkpoints/unet"),
    "Attention U-Net": Path("checkpoints/attention_unet"),
    "SegFormer-B0": Path("checkpoints/segformer"),
}

HISTORY_FILENAME = "training_history.csv"


# ==========================================================
# Arguments
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot training histories for all segmentation models."
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where plots will be saved.",
    )

    return parser.parse_args()


# ==========================================================
# Column Detection
# ==========================================================

def normalize_column_name(name):
    """
    Normalize a CSV column name so that small naming differences
    such as 'val_dice', 'validation_dice', or 'Val Dice' can be
    recognized.
    """
    return (
        str(name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


def find_column(df, candidates):
    """
    Find a column using a list of possible names.

    Returns None if no candidate exists.
    """

    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate = normalize_column_name(candidate)

        if candidate in normalized:
            return normalized[candidate]

    return None


def detect_columns(df):
    """
    Detect epoch and metric columns.

    Supports common names such as:

        train_loss
        val_loss
        validation_loss

        train_dice
        val_dice
        validation_dice

        train_iou
        val_iou

        train_precision
        val_precision

        train_recall
        val_recall

        learning_rate
        lr
    """

    columns = {}

    columns["epoch"] = find_column(
        df,
        [
            "epoch",
            "epochs",
        ],
    )

    columns["train_loss"] = find_column(
        df,
        [
            "train_loss",
            "training_loss",
            "loss_train",
        ],
    )

    columns["val_loss"] = find_column(
        df,
        [
            "val_loss",
            "validation_loss",
            "valid_loss",
            "loss_val",
            "loss_validation",
        ],
    )

    columns["train_dice"] = find_column(
        df,
        [
            "train_dice",
            "training_dice",
            "dice_train",
        ],
    )

    columns["val_dice"] = find_column(
        df,
        [
            "val_dice",
            "validation_dice",
            "valid_dice",
            "dice_val",
            "dice_validation",
        ],
    )

    columns["train_iou"] = find_column(
        df,
        [
            "train_iou",
            "training_iou",
            "train_iou_score",
            "iou_train",
        ],
    )

    columns["val_iou"] = find_column(
        df,
        [
            "val_iou",
            "validation_iou",
            "valid_iou",
            "val_iou_score",
            "iou_val",
            "iou_validation",
        ],
    )

    columns["train_precision"] = find_column(
        df,
        [
            "train_precision",
            "training_precision",
            "precision_train",
        ],
    )

    columns["val_precision"] = find_column(
        df,
        [
            "val_precision",
            "validation_precision",
            "valid_precision",
            "precision_val",
            "precision_validation",
        ],
    )

    columns["train_recall"] = find_column(
        df,
        [
            "train_recall",
            "training_recall",
            "recall_train",
        ],
    )

    columns["val_recall"] = find_column(
        df,
        [
            "val_recall",
            "validation_recall",
            "valid_recall",
            "recall_val",
            "recall_validation",
        ],
    )

    columns["learning_rate"] = find_column(
        df,
        [
            "learning_rate",
            "lr",
        ],
    )

    return columns


# ==========================================================
# Load History
# ==========================================================

def load_history(model_name, model_dir):
    """
    Load one model's training history.
    """

    history_path = model_dir / HISTORY_FILENAME

    if not history_path.exists():
        raise FileNotFoundError(
            f"\nTraining history not found for {model_name}:\n"
            f"  {history_path}\n"
        )

    df = pd.read_csv(history_path)

    if df.empty:
        raise ValueError(
            f"Training history is empty:\n{history_path}"
        )

    columns = detect_columns(df)

    if columns["epoch"] is None:
        # Fall back to row number if epoch is not stored.
        df["_epoch"] = range(1, len(df) + 1)
        columns["epoch"] = "_epoch"

    return df, columns


# ==========================================================
# Plot Helper
# ==========================================================

def plot_metric(
    histories,
    metric_name,
    train_key,
    val_key,
    output_file,
    ylabel,
    title,
    y_min=None,
    y_max=None,
):
    """
    Plot one metric for all models.

    Each model gets:
        solid line   = training
        dashed line  = validation
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    plotted_any = False

    for model_name, data in histories.items():

        df = data["df"]
        columns = data["columns"]

        epoch_col = columns["epoch"]

        train_col = columns.get(train_key)
        val_col = columns.get(val_key)

        if train_col is not None:

            ax.plot(
                df[epoch_col],
                df[train_col],
                label=f"{model_name} - Train",
            )

            plotted_any = True

        if val_col is not None:

            ax.plot(
                df[epoch_col],
                df[val_col],
                linestyle="--",
                label=f"{model_name} - Validation",
            )

            plotted_any = True

    if not plotted_any:
        plt.close(fig)
        print(
            f"Skipping {metric_name}: "
            "no matching columns were found."
        )
        return

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)

    if y_min is not None or y_max is not None:
        ax.set_ylim(
            bottom=y_min,
            top=y_max,
        )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best",
        fontsize=9,
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )


def plot_learning_rate(
    histories,
    output_file,
):
    """
    Plot learning rate for all models.
    """

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    plotted_any = False

    for model_name, data in histories.items():

        df = data["df"]
        columns = data["columns"]

        epoch_col = columns["epoch"]
        lr_col = columns.get(
            "learning_rate"
        )

        if lr_col is None:
            continue

        ax.plot(
            df[epoch_col],
            df[lr_col],
            label=model_name,
        )

        plotted_any = True

    if not plotted_any:
        plt.close(fig)

        print(
            "Skipping learning rate plot: "
            "no learning_rate/lr column found."
        )

        return

    ax.set_title(
        "Learning Rate vs Epoch"
    )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Learning Rate"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_file}"
    )


# ==========================================================
# Best Validation Metrics
# ==========================================================

def print_best_results(histories):
    """
    Print best validation Dice and IoU for every model.
    """

    print()
    print("=" * 80)
    print("BEST VALIDATION RESULTS")
    print("=" * 80)

    for model_name, data in histories.items():

        df = data["df"]
        columns = data["columns"]

        epoch_col = columns["epoch"]

        print()
        print(model_name)
        print("-" * 40)

        # --------------------------------------------------
        # Dice
        # --------------------------------------------------

        val_dice_col = columns.get(
            "val_dice"
        )

        if val_dice_col is not None:

            valid = df[
                [epoch_col, val_dice_col]
            ].dropna()

            if not valid.empty:

                best_idx = valid[
                    val_dice_col
                ].idxmax()

                best_epoch = valid.loc[
                    best_idx,
                    epoch_col,
                ]

                best_dice = valid.loc[
                    best_idx,
                    val_dice_col,
                ]

                print(
                    f"Best Val Dice : "
                    f"{best_dice:.4f} "
                    f"(Epoch {int(best_epoch)})"
                )

        # --------------------------------------------------
        # IoU
        # --------------------------------------------------

        val_iou_col = columns.get(
            "val_iou"
        )

        if val_iou_col is not None:

            valid = df[
                [epoch_col, val_iou_col]
            ].dropna()

            if not valid.empty:

                best_idx = valid[
                    val_iou_col
                ].idxmax()

                best_epoch = valid.loc[
                    best_idx,
                    epoch_col,
                ]

                best_iou = valid.loc[
                    best_idx,
                    val_iou_col,
                ]

                print(
                    f"Best Val IoU  : "
                    f"{best_iou:.4f} "
                    f"(Epoch {int(best_epoch)})"
                )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        val_loss_col = columns.get(
            "val_loss"
        )

        if val_loss_col is not None:

            valid = df[
                [epoch_col, val_loss_col]
            ].dropna()

            if not valid.empty:

                best_idx = valid[
                    val_loss_col
                ].idxmin()

                best_epoch = valid.loc[
                    best_idx,
                    epoch_col,
                ]

                best_loss = valid.loc[
                    best_idx,
                    val_loss_col,
                ]

                print(
                    f"Lowest Val Loss: "
                    f"{best_loss:.4f} "
                    f"(Epoch {int(best_epoch)})"
                )


# ==========================================================
# Training Summary
# ==========================================================

def print_history_summary(histories):
    """
    Print basic training-history information.
    """

    print()
    print("=" * 80)
    print("TRAINING HISTORY SUMMARY")
    print("=" * 80)

    for model_name, data in histories.items():

        df = data["df"]
        columns = data["columns"]

        epoch_col = columns["epoch"]

        first_epoch = df[
            epoch_col
        ].iloc[0]

        last_epoch = df[
            epoch_col
        ].iloc[-1]

        print(
            f"{model_name:<20} "
            f"Epochs recorded: "
            f"{int(first_epoch)}-{int(last_epoch)}"
        )


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

    print("=" * 80)
    print("TRAINING HISTORY ANALYSIS")
    print("=" * 80)

    histories = {}

    # ------------------------------------------------------
    # Load all histories
    # ------------------------------------------------------

    for model_name, model_dir in (
        MODEL_DIRS.items()
    ):

        try:

            df, columns = load_history(
                model_name,
                model_dir,
            )

            histories[
                model_name
            ] = {
                "df": df,
                "columns": columns,
            }

            print(
                f"\nLoaded {model_name}:"
            )

            print(
                f"  File: "
                f"{model_dir / HISTORY_FILENAME}"
            )

            print(
                f"  Rows: "
                f"{len(df)}"
            )

            print(
                f"  Columns: "
                f"{list(df.columns)}"
            )

        except Exception as error:

            print(
                f"\nWARNING: Could not load "
                f"{model_name}:"
            )

            print(
                f"  {error}"
            )

    if not histories:

        raise RuntimeError(
            "No training histories could be loaded."
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    print_history_summary(
        histories
    )

    # ------------------------------------------------------
    # Loss
    # ------------------------------------------------------

    plot_metric(
        histories=histories,
        metric_name="loss",
        train_key="train_loss",
        val_key="val_loss",
        output_file=(
            output_dir
            / "loss_comparison.png"
        ),
        ylabel="Loss",
        title=(
            "Training and Validation Loss"
        ),
    )

    # ------------------------------------------------------
    # Dice
    # ------------------------------------------------------

    plot_metric(
        histories=histories,
        metric_name="dice",
        train_key="train_dice",
        val_key="val_dice",
        output_file=(
            output_dir
            / "dice_comparison.png"
        ),
        ylabel="Dice Score",
        title=(
            "Training and Validation Dice"
        ),
        y_min=0.0,
        y_max=1.0,
    )

    # ------------------------------------------------------
    # IoU
    # ------------------------------------------------------

    plot_metric(
        histories=histories,
        metric_name="iou",
        train_key="train_iou",
        val_key="val_iou",
        output_file=(
            output_dir
            / "iou_comparison.png"
        ),
        ylabel="IoU",
        title=(
            "Training and Validation IoU"
        ),
        y_min=0.0,
        y_max=1.0,
    )

    # ------------------------------------------------------
    # Precision
    # ------------------------------------------------------

    plot_metric(
        histories=histories,
        metric_name="precision",
        train_key="train_precision",
        val_key="val_precision",
        output_file=(
            output_dir
            / "precision_comparison.png"
        ),
        ylabel="Precision",
        title=(
            "Training and Validation Precision"
        ),
        y_min=0.0,
        y_max=1.0,
    )

    # ------------------------------------------------------
    # Recall
    # ------------------------------------------------------

    plot_metric(
        histories=histories,
        metric_name="recall",
        train_key="train_recall",
        val_key="val_recall",
        output_file=(
            output_dir
            / "recall_comparison.png"
        ),
        ylabel="Recall",
        title=(
            "Training and Validation Recall"
        ),
        y_min=0.0,
        y_max=1.0,
    )

    # ------------------------------------------------------
    # Learning rate
    # ------------------------------------------------------

    plot_learning_rate(
        histories=histories,
        output_file=(
            output_dir
            / "learning_rate_comparison.png"
        ),
    )

    # ------------------------------------------------------
    # Best validation results
    # ------------------------------------------------------

    print_best_results(
        histories
    )

    # ------------------------------------------------------
    # Finish
    # ------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING HISTORY ANALYSIS COMPLETE")
    print("=" * 80)

    print(
        f"Plots saved to: "
        f"{output_dir.resolve()}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
