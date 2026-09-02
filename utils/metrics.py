"""
Evaluation metrics for binary medical image segmentation.
"""

import torch

EPS = 1e-7


# ==========================================================
# Helper functions
# ==========================================================

def _prepare_predictions(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
):
    """
    Convert model logits to binary predictions.

    Args:
        logits: Model outputs of shape (B, 1, H, W).
        targets: Ground truth masks of shape (B, 1, H, W).
        threshold: Probability threshold.

    Returns:
        Tuple (predictions, targets).
    """

    preds = torch.sigmoid(logits)
    preds = (preds > threshold).float()

    return preds, targets.float()


def _statistics(
    preds: torch.Tensor,
    targets: torch.Tensor,
):
    """
    Compute TP, FP, FN and TN for every image.

    Returns:
        Tuple of tensors with shape (batch_size,)
    """

    dims = (1, 2, 3)

    tp = (preds * targets).sum(dim=dims)

    fp = (preds * (1.0 - targets)).sum(dim=dims)

    fn = ((1.0 - preds) * targets).sum(dim=dims)

    tn = ((1.0 - preds) * (1.0 - targets)).sum(dim=dims)

    return tp, fp, fn, tn


# ==========================================================
# Private metric implementations
# ==========================================================

def _dice(tp, fp, fn):

    return (2 * tp + EPS) / (2 * tp + fp + fn + EPS)


def _iou(tp, fp, fn):

    return (tp + EPS) / (tp + fp + fn + EPS)


def _precision(tp, fp):

    return (tp + EPS) / (tp + fp + EPS)


def _recall(tp, fn):

    return (tp + EPS) / (tp + fn + EPS)


def _f1(precision, recall):

    return (
        2 * precision * recall
    ) / (
        precision + recall + EPS
    )


def _pixel_accuracy(preds, targets):

    accuracy = (preds == targets).float()

    return accuracy.mean(dim=(1, 2, 3))


# ==========================================================
# Public metric functions
# ==========================================================

def dice_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean Dice coefficient.
    """

    preds, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    tp, fp, fn, _ = _statistics(
        preds,
        targets,
    )

    return _dice(tp, fp, fn).mean().item()


def iou_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean Intersection-over-Union.
    """

    preds, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    tp, fp, fn, _ = _statistics(
        preds,
        targets,
    )

    return _iou(tp, fp, fn).mean().item()


def precision_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean precision.
    """

    preds, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    tp, fp, _, _ = _statistics(
        preds,
        targets,
    )

    return _precision(tp, fp).mean().item()


def recall_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean recall.
    """

    preds, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    tp, _, fn, _ = _statistics(
        preds,
        targets,
    )

    return _recall(tp, fn).mean().item()


def f1_score(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean F1-score.
    """

    preds, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    tp, fp, fn, _ = _statistics(
        preds,
        targets,
    )

    precision = _precision(tp, fp)

    recall = _recall(tp, fn)

    return _f1(
        precision,
        recall,
    ).mean().item()


def pixel_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Mean pixel accuracy.
    """

    preds, targets = _prepare_predictions(
        logits,
        targets,
        threshold,
    )

    return _pixel_accuracy(
        preds,
        targets,
    ).mean().item()

# ==========================================================
# Metric Tracker
# ==========================================================

class MetricTracker:
    """
    Accumulates loss and evaluation metrics over an epoch.

    Example:
        tracker = MetricTracker()

        tracker.update(loss, outputs, masks)

        results = tracker.compute()

        print(results["dice"])
    """

    def __init__(self):

        self.reset()

    def reset(self):
        """
        Reset all stored statistics.
        """

        self.history = {
            "loss": [],
            "dice": [],
            "iou": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "pixel_accuracy": [],
        }

    def update(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        targets: torch.Tensor,
        threshold: float = 0.5,
    ):
        """
        Update metrics for one batch.
        """

        self.history["loss"].append(loss.item())

        preds, targets = _prepare_predictions(
            logits,
            targets,
            threshold,
        )

        tp, fp, fn, _ = _statistics(
            preds,
            targets,
        )

        dice = _dice(tp, fp, fn)

        iou = _iou(tp, fp, fn)

        precision = _precision(tp, fp)

        recall = _recall(tp, fn)

        f1 = _f1(
            precision,
            recall,
        )

        pixel_accuracy = _pixel_accuracy(
            preds,
            targets,
        )

        self.history["dice"].append(
            dice.mean().item()
        )

        self.history["iou"].append(
            iou.mean().item()
        )

        self.history["precision"].append(
            precision.mean().item()
        )

        self.history["recall"].append(
            recall.mean().item()
        )

        self.history["f1"].append(
            f1.mean().item()
        )

        self.history["pixel_accuracy"].append(
            pixel_accuracy.mean().item()
        )

    def compute(self):
        """
        Compute average metrics over the epoch.

        Returns:
            Dictionary containing average metrics.
        """

        if len(self.history["loss"]) == 0:
            raise RuntimeError(
                "MetricTracker contains no data."
            )

        results = {}

        for key, values in self.history.items():

            results[key] = (
                sum(values) / len(values)
            )

        return results

    def __getitem__(self, metric):
        """
        Allow access like:

            tracker["dice"]
        """

        return self.compute()[metric]

    def __str__(self):

        results = self.compute()

        return (
            f"Loss: {results['loss']:.4f} | "
            f"Dice: {results['dice']:.4f} | "
            f"IoU: {results['iou']:.4f} | "
            f"Precision: {results['precision']:.4f} | "
            f"Recall: {results['recall']:.4f} | "
            f"F1: {results['f1']:.4f} | "
            f"Pixel Acc: {results['pixel_accuracy']:.4f}"
        )