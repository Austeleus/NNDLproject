import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import Config
from ..data import DataLoaders
from ..data.dataset import load_class_mappings
from ..data.transforms import mixup_data
from ..losses import HierarchicalLoss
from ..models import HierarchicalClassifier
from ..utils.logging import WandbLogger
from .metrics import MetricsCalculator, EpochMetrics
from .scheduler import (
    ProgressiveUnfreezeScheduler,
    create_optimizer,
    create_lr_scheduler,
)


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_val_acc: float = 0.0
    best_epoch: int = 0
    patience_counter: int = 0


class Trainer:
    def __init__(
        self,
        model: HierarchicalClassifier,
        config: Config,
        dataloaders: DataLoaders,
        logger: Optional[WandbLogger] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.dataloaders = dataloaders
        self.logger = logger

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.loss_fn = HierarchicalLoss(
            config.loss,
            use_margin_loss=getattr(config.loss, 'use_margin_loss', False),
            margin=getattr(config.loss, 'margin', 2.0),
            lambda_margin=getattr(config.loss, 'lambda_margin', 0.5),
            num_subclasses=config.model.num_subclasses,
            num_superclasses=config.model.num_superclasses,
        )
        self.optimizer = create_optimizer(model, config.training)

        steps_per_epoch = len(dataloaders.train)
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            config.training,
            steps_per_epoch,
        )

        self.unfreeze_scheduler = ProgressiveUnfreezeScheduler(model, config.training)

        self.train_metrics = MetricsCalculator(
            num_superclasses=config.model.num_superclasses,
            num_subclasses=config.model.num_subclasses,
        )
        self.val_metrics = MetricsCalculator(
            num_superclasses=config.model.num_superclasses,
            num_subclasses=config.model.num_subclasses,
        )

        self.scaler = GradScaler() if self.device.type == "cuda" else None

        self.state = TrainerState()

        self.save_dir = Path(config.logging.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.oe_iter = None

        self.use_mixup = getattr(config.augmentation, 'mix_prob', 0) > 0
        self.mixup_alpha = getattr(config.augmentation, 'mixup_alpha', 0.2)
        self.mix_prob = getattr(config.augmentation, 'mix_prob', 0.5)

    def train(self) -> Dict[str, float]:
        for epoch in range(self.config.training.epochs):
            self.state.epoch = epoch

            new_lr = self.unfreeze_scheduler.apply_phase(epoch)
            if new_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = new_lr
                trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                print(f"Epoch {epoch}: Unfreezing phase, LR={new_lr:.1e}, trainable params={trainable:,}")

            train_metrics = self._train_epoch()

            val_seen_metrics = self._validate(self.dataloaders.val_seen, "val_seen")
            val_unseen_metrics = self._validate(self.dataloaders.val_unseen, "val_unseen", is_unseen=True)

            combined_acc = (
                val_seen_metrics.super_acc + val_unseen_metrics.super_acc +
                val_seen_metrics.sub_acc + val_unseen_metrics.sub_acc
            ) / 4

            if self.logger:
                self.logger.log_epoch(
                    epoch=epoch,
                    train_metrics=train_metrics.to_dict(),
                    val_metrics={
                        **{f"seen_{k}": v for k, v in val_seen_metrics.to_dict().items()},
                        **{f"unseen_{k}": v for k, v in val_unseen_metrics.to_dict().items()},
                    },
                    lr=self.optimizer.param_groups[0]["lr"],
                )

            print(
                f"Epoch {epoch}: "
                f"Train Super={train_metrics.super_acc:.1f}%, Sub={train_metrics.sub_acc:.1f}% | "
                f"Val Seen={val_seen_metrics.super_acc:.1f}%/{val_seen_metrics.sub_acc:.1f}%, "
                f"Unseen={val_unseen_metrics.super_acc:.1f}%/{val_unseen_metrics.sub_acc:.1f}% | "
                f"Combined={combined_acc:.1f}%"
            )

            if combined_acc > self.state.best_val_acc:
                self.state.best_val_acc = combined_acc
                self.state.best_epoch = epoch
                self.state.patience_counter = 0
                self._save_checkpoint("best_model.pt")
            else:
                self.state.patience_counter += 1

            if self.state.patience_counter >= self.config.training.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self._save_checkpoint("final_model.pt")

        return {
            "best_val_acc": self.state.best_val_acc,
            "best_epoch": self.state.best_epoch,
        }

    def _train_epoch(self) -> EpochMetrics:
        self.model.train()
        self.train_metrics.reset()

        oe_loader = self.dataloaders.oe
        self.oe_iter = iter(oe_loader)

        pbar = tqdm(self.dataloaders.train, desc=f"Epoch {self.state.epoch}")

        for batch_idx, batch in enumerate(pbar):
            loss = self._train_step(batch, batch_idx)

            pbar.set_postfix({"loss": f"{loss:.4f}"})

            self.state.global_step += 1

        return self.train_metrics.compute()

    def _train_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> float:
        images = batch["image"].to(self.device)
        super_targets = batch["superclass"].to(self.device)
        sub_targets = batch["subclass"].to(self.device)
        is_oe = batch["is_oe"].to(self.device)

        if batch_idx % 3 == 0 and self.dataloaders.oe is not None:
            if self.oe_iter is None:
                self.oe_iter = iter(self.dataloaders.oe)
            try:
                oe_batch = next(self.oe_iter)
            except StopIteration:
                self.oe_iter = iter(self.dataloaders.oe)
                oe_batch = next(self.oe_iter)

            oe_images = oe_batch["image"].to(self.device)
            oe_super = oe_batch["superclass"].to(self.device)
            oe_sub = oe_batch["subclass"].to(self.device)
            oe_flags = torch.ones(oe_images.size(0), dtype=torch.bool, device=self.device)

            images = torch.cat([images, oe_images], dim=0)
            super_targets = torch.cat([super_targets, oe_super], dim=0)
            sub_targets = torch.cat([sub_targets, oe_sub], dim=0)
            is_oe = torch.cat([is_oe, oe_flags], dim=0)

        use_mixup_this_step = self.use_mixup and np.random.random() < self.mix_prob
        lam = 1.0
        super_targets_b = super_targets
        sub_targets_b = sub_targets
        is_oe_b = is_oe

        if use_mixup_this_step:
            images, super_targets, super_targets_b, sub_targets, sub_targets_b, lam, mix_index = mixup_data(
                images, super_targets, sub_targets, self.mixup_alpha
            )
            is_oe_b = is_oe[mix_index]

        self.optimizer.zero_grad()

        if self.scaler is not None:
            with autocast():
                output = self.model(images)
                if use_mixup_this_step:
                    loss_out_a = self.loss_fn(output, super_targets, sub_targets, is_oe)
                    loss_out_b = self.loss_fn(output, super_targets_b, sub_targets_b, is_oe_b)
                    total_loss = lam * loss_out_a.total + (1 - lam) * loss_out_b.total
                else:
                    loss_out = self.loss_fn(output, super_targets, sub_targets, is_oe)
                    total_loss = loss_out.total

            self.scaler.scale(total_loss).backward()

            if self.config.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(images)
            if use_mixup_this_step:
                loss_out_a = self.loss_fn(output, super_targets, sub_targets, is_oe)
                loss_out_b = self.loss_fn(output, super_targets_b, sub_targets_b, is_oe_b)
                total_loss = lam * loss_out_a.total + (1 - lam) * loss_out_b.total
            else:
                loss_out = self.loss_fn(output, super_targets, sub_targets, is_oe)
                total_loss = loss_out.total

            total_loss.backward()

            if self.config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )

            self.optimizer.step()

        self.lr_scheduler.step()

        if not use_mixup_this_step:
            seen_mask = ~is_oe
            if seen_mask.any():
                self.train_metrics.update(
                    output.super_logits[seen_mask],
                    output.sub_logits[seen_mask],
                    super_targets[seen_mask],
                    sub_targets[seen_mask],
                )

        return total_loss.item()

    @torch.no_grad()
    def _validate(
        self,
        dataloader: DataLoader,
        name: str,
        is_unseen: bool = False,
    ) -> EpochMetrics:
        self.model.eval()
        self.val_metrics.reset()

        for batch in dataloader:
            images = batch["image"].to(self.device)
            super_targets = batch["superclass"].to(self.device)
            sub_targets = batch["subclass"].to(self.device)

            if is_unseen:
                super_targets = torch.full_like(super_targets, self.config.model.num_superclasses)
                sub_targets = torch.full_like(sub_targets, self.config.model.num_subclasses)

            output = self.model(images)

            self.val_metrics.update(
                output.super_logits,
                output.sub_logits,
                super_targets,
                sub_targets,
            )

        return self.val_metrics.compute()

    def _save_checkpoint(self, filename: str) -> None:
        path = self.save_dir / filename

        checkpoint = {
            "epoch": self.state.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.state.best_val_acc,
            "config": {
                "model": self.config.model.__dict__,
                "training": self.config.training.__dict__,
            },
        }

        torch.save(checkpoint, path)

        if self.logger:
            self.logger.save_model(path)

    def load_checkpoint(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state.epoch = checkpoint["epoch"]
        self.state.best_val_acc = checkpoint["best_val_acc"]
