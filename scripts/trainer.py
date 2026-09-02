"""
Trainer for medical image segmentation.
"""
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast

from utils.metrics import MetricTracker
from utils.schedulers import step_scheduler

from tqdm import tqdm


class EarlyStopping:
    """
    Early stopping based on a monitored metric.
    """

    def __init__(
        self,
        patience=10,
        mode="max",
        min_delta=0.0,
    ):
        """
        Args:
            patience:
                Number of epochs without improvement.

            mode:
                "max" or "min".

            min_delta:
                Minimum improvement required.
        """

        self.patience = patience

        self.mode = mode

        self.min_delta = min_delta

        self.counter = 0

        if mode == "max":
            self.best_score = float("-inf")
        else:
            self.best_score = float("inf")

        self.should_stop = False

    def step(self, value):
        """
        Update early stopping state.

        Returns
        -------
        bool
            True if training should stop.
        """

        if self.mode == "max":

            improved = value > (
                self.best_score + self.min_delta
            )

        else:

            improved = value < (
                self.best_score - self.min_delta
            )

        if improved:

            self.best_score = value

            self.counter = 0

        else:

            self.counter += 1

            if self.counter >= self.patience:

                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Generic trainer for semantic segmentation.
    """

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        device,
        scheduler=None,
        epochs=50,
        monitor="dice",
        mode="max",
        use_amp=False,
        gradient_clip=1.0,
        patience=10,
        checkpoint_dir="checkpoints",
    ):

        self.device = device

        self.model = model.to(device)

        self.criterion = criterion

        self.optimizer = optimizer

        self.scheduler = scheduler

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.epochs = epochs

        self.monitor = monitor

        self.mode = mode

        self.use_amp = (
            use_amp
            and device.type == "cuda"
        )

        self.gradient_clip = gradient_clip

        self.scaler = GradScaler(
            enabled=self.use_amp
        )

        self.early_stopping = EarlyStopping(
            patience=patience,
            mode=mode,
        )

        self.best_metric = (
            float("-inf")
            if mode == "max"
            else float("inf")
        )

        self.current_epoch = 0

        self.history = {
            "train_loss": [],
            "train_dice": [],
            "train_iou": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "train_pixel_accuracy": [],

            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "val_pixel_accuracy": [],

            "learning_rate": [],
        }

        self.checkpoint_dir = Path(
            checkpoint_dir
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # =====================================================
    # Helper methods
    # =====================================================

    def _move_to_device(
        self,
        images,
        masks,
    ):
        """
        Move tensors to device.
        """

        images = images.to(
            self.device,
            non_blocking=True,
        )

        masks = masks.to(
            self.device,
            non_blocking=True,
        )

        return images, masks

    def _forward_pass(
        self,
        images,
        masks,
    ):
        """
        Forward pass with AMP.
        """

        with autocast(
            enabled=self.use_amp,
        ):

            outputs = self.model(images)

            loss = self.criterion(
                outputs,
                masks,
            )

        return outputs, loss

    def _backward_pass(
        self,
        loss,
    ):
        """
        Backward pass with gradient scaling.
        """

        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()

        if self.gradient_clip is not None:

            self.scaler.unscale_(
                self.optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.gradient_clip,
            )

        self.scaler.step(
            self.optimizer
        )

        self.scaler.update()

    def _is_best(
        self,
        metric,
    ):
        """
        Check whether current metric is the best.
        """

        if self.mode == "max":

            return metric > self.best_metric

        return metric < self.best_metric
    
    
    # =====================================================
    # Training
    # =====================================================

    def train_epoch(self):
        """
        Train the model for one epoch.

        Returns
        -------
        dict
            Dictionary containing average training metrics.
        """

        self.model.train()

        tracker = MetricTracker()

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Train {self.current_epoch + 1}/{self.epochs}",
            leave=False,
        )

        for images, masks in progress_bar:

            # ---------------------------------------------
            # Move batch to device
            # ---------------------------------------------

            images, masks = self._move_to_device(
                images,
                masks,
            )

            # ---------------------------------------------
            # Forward pass
            # ---------------------------------------------

            outputs, loss = self._forward_pass(
                images,
                masks,
            )

            # ---------------------------------------------
            # Backward pass
            # ---------------------------------------------

            self._backward_pass(loss)

            # ---------------------------------------------
            # Update metrics
            # ---------------------------------------------

            tracker.update(
                loss,
                outputs.detach(),
                masks.detach(),
            )

            metrics = tracker.compute()

            current_lr = (
                self.optimizer.param_groups[0]["lr"]
            )

            progress_bar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                dice=f"{metrics['dice']:.4f}",
                lr=f"{current_lr:.2e}",
            )

        progress_bar.close()

        results = tracker.compute()

        results["lr"] = (
            self.optimizer.param_groups[0]["lr"]
        )

        return results
    
    # =====================================================
    # Validation
    # =====================================================

    @torch.inference_mode()
    def validate(self):
        """
        Validate the model for one epoch.

        Returns
        -------
        dict
            Dictionary containing average validation metrics.
        """

        self.model.eval()

        tracker = MetricTracker()

        progress_bar = tqdm(
            self.val_loader,
            desc=f"Val {self.current_epoch + 1}/{self.epochs}",
            leave=False,
        )

        for images, masks in progress_bar:

            # ---------------------------------------------
            # Move batch to device
            # ---------------------------------------------

            images, masks = self._move_to_device(
                images,
                masks,
            )

            # ---------------------------------------------
            # Forward pass
            # ---------------------------------------------

            outputs, loss = self._forward_pass(
                images,
                masks,
            )

            # ---------------------------------------------
            # Update metrics
            # ---------------------------------------------

            tracker.update(
                loss,
                outputs,
                masks,
            )

            metrics = tracker.compute()

            progress_bar.set_postfix(
                loss=f"{metrics['loss']:.4f}",
                dice=f"{metrics['dice']:.4f}",
            )

        progress_bar.close()

        return tracker.compute()
    
    # =====================================================
    # Checkpoints
    # =====================================================

    def save_checkpoint(
        self,
        filename,
    ):
        """
        Save training checkpoint.
        """

        checkpoint = {

            "epoch": self.current_epoch,

            "model_state_dict":
                self.model.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "scheduler_state_dict":
                (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),

            "best_metric":
                self.best_metric,

            "history":
                self.history,
        }

        torch.save(
            checkpoint,
            self.checkpoint_dir / filename,
        )

    def load_checkpoint(
        self,
        filename="latest_model.pth",
    ):
        """
        Load checkpoint.
        """

        checkpoint = torch.load(
            self.checkpoint_dir / filename,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if (
            self.scheduler is not None
            and checkpoint["scheduler_state_dict"] is not None
        ):

            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.current_epoch = checkpoint["epoch"]

        self.best_metric = checkpoint["best_metric"]

        self.history = checkpoint["history"]

        print(
            f"Checkpoint loaded "
            f"(epoch {self.current_epoch})."
        )

    # =====================================================
    # Scheduler
    # =====================================================

    def _step_scheduler(
        self,
        val_results,
    ):
        """
        Update learning-rate scheduler.
        """

        if self.scheduler is None:
            return

        step_scheduler(
            self.scheduler,
            metric=val_results[self.monitor],
        )

    # =====================================================
    # Best model
    # =====================================================

    def _save_best_model(
        self,
        metric,
    ):
        """
        Save best-performing model.
        """

        if self._is_best(metric):

            self.best_metric = metric

            self.save_checkpoint(
                "best_model.pth"
            )

            print(
                f"✓ New best "
                f"{self.monitor}: "
                f"{metric:.4f}"
            )

    def _save_latest_model(self):
        """
        Save latest checkpoint.
        """

        self.save_checkpoint(
            "latest_model.pth"
        )
        
    # =====================================================
    # Main training loop
    # =====================================================

    import time
    
    def fit(self):
        
        overall_start = time.time()
        """
        Train the model.
        """

        print("=" * 70)
        print("Training started")
        print("=" * 70)

        for epoch in range(
            self.current_epoch,
            self.epochs,
        ):

            self.current_epoch = epoch

            print()
            print(
                f"Epoch {epoch + 1}/{self.epochs}"
            )
            
            # ------------------------------------------
            # Start timer
            # ------------------------------------------

            start_time = time.time()

            # ------------------------------------------
            # Train
            # ------------------------------------------

            train_results = self.train_epoch()

            # ------------------------------------------
            # Validate
            # ------------------------------------------

            val_results = self.validate()

            # ------------------------------------------
            # Scheduler
            # ------------------------------------------

            self._step_scheduler(
                val_results
            )

            # ------------------------------------------
            # History
            # ------------------------------------------

            for key, value in train_results.items():

                if key == "lr":
                    continue

                self.history[
                    f"train_{key}"
                ].append(value)

            for key, value in val_results.items():

                self.history[
                    f"val_{key}"
                ].append(value)

            self.history[
                "learning_rate"
            ].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # ------------------------------------------
            # Console output
            # ------------------------------------------

            print(
                "\nTraining"
            )

            print(
                f"Loss          : {train_results['loss']:.4f}"
            )

            print(
                f"Dice          : {train_results['dice']:.4f}"
            )

            print(
                f"IoU           : {train_results['iou']:.4f}"
            )

            print(
                "\nValidation"
            )

            print(
                f"Loss          : {val_results['loss']:.4f}"
            )

            print(
                f"Dice          : {val_results['dice']:.4f}"
            )

            print(
                f"IoU           : {val_results['iou']:.4f}"
            )

            print(
                f"\nLearning Rate : "
                f"{self.optimizer.param_groups[0]['lr']:.2e}"
            )

            # ------------------------------------------
            # Best model
            # ------------------------------------------

            self._save_best_model(
                val_results[
                    self.monitor
                ]
            )

            # ------------------------------------------
            # Latest checkpoint
            # ------------------------------------------

            self._save_latest_model()
            
            # ------------------------------------------
            # Epoch time
            # ------------------------------------------

            elapsed = time.time() - start_time

            print(f"Epoch Time    : {elapsed:.1f}s")
            
            # ------------------------------------------
            # GPU Memory
            # ------------------------------------------

            if self.device.type == "cuda":

                current_memory = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                peak_memory = (
                    torch.cuda.max_memory_allocated()
                    / 1024**3
                )

                print(
                    f"GPU Memory    : "
                    f"{current_memory:.2f} GB "
                    f"(Peak: {peak_memory:.2f} GB)"
                )

                torch.cuda.reset_peak_memory_stats()

            # ------------------------------------------
            # Early stopping
            # ------------------------------------------

            stop = self.early_stopping.step(
                val_results[
                    self.monitor
                ]
            )

            if stop:

                print()

                print(
                    "Early stopping triggered."
                )

                break

        print()

        print("=" * 70)
        print("Training completed")
        print("=" * 70)

        print(
            f"Best {self.monitor}: "
            f"{self.best_metric:.4f}"
        )

        print(
            f"Best checkpoint:"
        )

        print(
            self.checkpoint_dir /
            "best_model.pth"
        )
        
        
        import pandas as pd

        pd.DataFrame(
            self.history
        ).to_csv(
            self.checkpoint_dir /
            "training_history.csv",
            index=False,
        )
        
        total_time = time.time() - overall_start

        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        print(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        
        
      
        
        
        