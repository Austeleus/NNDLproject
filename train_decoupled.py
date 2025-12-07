#!/usr/bin/env python3
"""
Decoupled training script for hierarchical classification with novel detection.

Phase A: Pure classification training (no novel class)
Phase B: Novelty detection training with small LR fine-tuning
"""
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import Config
from src.data import create_splits, create_dataloaders, load_class_mappings
from src.data.splits import save_splits_info
from src.data.dataset import get_subclasses_by_superclass
from src.data.transforms import mixup_data
from src.models import create_decoupled_model, DecoupledClassifier
from src.losses import PhaseALoss, PerHeadPhaseBLoss
from src.inference.calibration import calibrate_decoupled_model
from src.utils.logging import setup_logging


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_next_version(base_dir: Path) -> int:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not existing:
        return 1
    versions = []
    for d in existing:
        try:
            versions.append(int(d.name[1:]))
        except ValueError:
            pass
    return max(versions, default=0) + 1


class PhaseATrainer:
    """Phase A: Pure classification training."""

    def __init__(
        self,
        model: DecoupledClassifier,
        config: Config,
        dataloaders,
        device: torch.device,
        save_dir: Path,
    ):
        self.model = model
        self.config = config
        self.dataloaders = dataloaders
        self.device = device
        self.save_dir = save_dir

        self.loss_fn = PhaseALoss(
            label_smoothing=config.loss.label_smoothing,
            lambda_super=config.loss.lambda_super,
            lambda_sub=config.loss.lambda_sub,
        )

        self.scaler = GradScaler() if device.type == "cuda" else None

    def train(self, epochs: int, lr: float) -> dict:
        self.model.train()
        optimizer = AdamW(
            self.model.get_phase_a_params(),
            lr=lr,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs * len(self.dataloaders.train),
            eta_min=self.config.training.min_lr,
        )

        best_acc = 0.0
        best_epoch = 0

        for epoch in range(epochs):
            train_metrics = self._train_epoch(optimizer, scheduler, epoch)
            val_metrics = self._validate()

            combined_acc = (
                val_metrics["super_acc"] + val_metrics["sub_acc"]
            ) / 2

            print(
                f"[Phase A] Epoch {epoch}: "
                f"Train Super={train_metrics['super_acc']:.1f}%, Sub={train_metrics['sub_acc']:.1f}% | "
                f"Val Super={val_metrics['super_acc']:.1f}%, Sub={val_metrics['sub_acc']:.1f}%"
            )

            if combined_acc > best_acc:
                best_acc = combined_acc
                best_epoch = epoch
                self._save_checkpoint("phase_a_best.pt", epoch)

        self._save_checkpoint("phase_a_final.pt", epochs - 1)

        return {"best_acc": best_acc, "best_epoch": best_epoch}

    def _train_epoch(self, optimizer, scheduler, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        super_correct = 0
        sub_correct = 0
        total = 0

        pbar = tqdm(self.dataloaders.train, desc=f"Phase A Epoch {epoch}")

        for batch in pbar:
            images = batch["image"].to(self.device)
            super_targets = batch["superclass"].to(self.device)
            sub_targets = batch["subclass"].to(self.device)

            optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    output = self.model(images, super_targets=super_targets)
                    loss_out = self.loss_fn(
                        output.super_logits,
                        output.sub_logits,
                        super_targets,
                        sub_targets,
                    )

                self.scaler.scale(loss_out.total).backward()

                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip,
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = self.model(images, super_targets=super_targets)
                loss_out = self.loss_fn(
                    output.super_logits,
                    output.sub_logits,
                    super_targets,
                    sub_targets,
                )

                loss_out.total.backward()

                if self.config.training.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip,
                    )

                optimizer.step()

            scheduler.step()

            total_loss += loss_out.total.item()
            super_preds = output.super_logits.argmax(dim=1)
            sub_preds = output.sub_logits.argmax(dim=1)
            super_correct += (super_preds == super_targets).sum().item()
            sub_correct += (sub_preds == sub_targets).sum().item()
            total += images.size(0)

            pbar.set_postfix({"loss": f"{loss_out.total.item():.4f}"})

        return {
            "loss": total_loss / len(self.dataloaders.train),
            "super_acc": 100.0 * super_correct / total,
            "sub_acc": 100.0 * sub_correct / total,
        }

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()
        super_correct = 0
        sub_correct = 0
        total = 0

        for batch in self.dataloaders.val_seen:
            images = batch["image"].to(self.device)
            super_targets = batch["superclass"].to(self.device)
            sub_targets = batch["subclass"].to(self.device)

            output = self.model(images, super_targets=super_targets)

            super_preds = output.super_logits.argmax(dim=1)
            sub_preds = output.sub_logits.argmax(dim=1)

            super_correct += (super_preds == super_targets).sum().item()
            sub_correct += (sub_preds == sub_targets).sum().item()
            total += images.size(0)

        return {
            "super_acc": 100.0 * super_correct / total,
            "sub_acc": 100.0 * sub_correct / total,
        }

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        path = self.save_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "phase": "A",
        }, path)


class PhaseBTrainer:
    """Phase B: Novelty detection training with fine-tuning."""

    def __init__(
        self,
        model: DecoupledClassifier,
        config: Config,
        dataloaders,
        device: torch.device,
        save_dir: Path,
        backbone_lr_mult: float = 0.01,
    ):
        self.model = model
        self.config = config
        self.dataloaders = dataloaders
        self.device = device
        self.save_dir = save_dir
        self.backbone_lr_mult = backbone_lr_mult

        self.loss_fn = PerHeadPhaseBLoss(
            num_superclasses=config.model.num_superclasses,
            energy_margin=getattr(config.loss, 'energy_margin', 5.0),
            lambda_energy_margin=getattr(config.loss, 'lambda_energy_margin', 0.5),
            lambda_classification=0.1,
            label_smoothing=config.loss.label_smoothing,
        )

        self.scaler = GradScaler() if device.type == "cuda" else None
        self.oe_iter = None

    def train(
        self,
        epochs: int,
        lr: float,
        novelty_hidden_dim: int = 256,
        energy_weight: float = 0.5,
    ) -> dict:
        self.model.init_phase_b(
            hidden_dim=novelty_hidden_dim,
            temperature=1.0,
            energy_weight=energy_weight,
        )
        self.model.to(self.device)

        param_groups = self.model.get_phase_b_params(
            include_backbone=True,
            backbone_lr_mult=self.backbone_lr_mult,
        )

        optimizer_params = []
        for group in param_groups:
            optimizer_params.append({
                "params": group["params"],
                "lr": lr * group["lr_mult"],
            })

        optimizer = AdamW(
            optimizer_params,
            weight_decay=self.config.training.weight_decay,
        )
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs * len(self.dataloaders.train),
            eta_min=self.config.training.min_lr,
        )

        best_f1 = -1.0
        best_epoch = 0

        for epoch in range(epochs):
            train_metrics = self._train_epoch(optimizer, scheduler, epoch)
            val_metrics = self._validate()

            combined_f1 = (
                val_metrics["super_novel_f1"] + val_metrics["sub_novel_f1"]
            ) / 2

            print(
                f"[Phase B] Epoch {epoch}: "
                f"Loss={train_metrics['loss']:.4f} | "
                f"Super Novel F1={val_metrics['super_novel_f1']:.3f}, "
                f"Sub Novel F1={val_metrics['sub_novel_f1']:.3f} | "
                f"Seen Acc: Super={val_metrics['seen_super_acc']:.1f}%, Sub={val_metrics['seen_sub_acc']:.1f}%"
            )

            if combined_f1 > best_f1:
                best_f1 = combined_f1
                best_epoch = epoch
                self._save_checkpoint("phase_b_best.pt", epoch)

        self._save_checkpoint("phase_b_final.pt", epochs - 1)

        return {"best_f1": best_f1, "best_epoch": best_epoch}

    def _train_epoch(self, optimizer, scheduler, epoch: int) -> dict:
        self.model.train()
        total_loss = 0.0
        self.oe_iter = iter(self.dataloaders.oe)

        pbar = tqdm(self.dataloaders.train, desc=f"Phase B Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            super_targets = batch["superclass"].to(self.device)
            sub_targets = batch["subclass"].to(self.device)
            batch_size = images.size(0)

            # For seen samples: neither super nor sub is novel
            is_super_novel = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
            is_sub_novel = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

            if batch_idx % 2 == 0:
                try:
                    oe_batch = next(self.oe_iter)
                except StopIteration:
                    self.oe_iter = iter(self.dataloaders.oe)
                    oe_batch = next(self.oe_iter)

                oe_images = oe_batch["image"].to(self.device)
                oe_super = oe_batch["superclass"].to(self.device)
                oe_sub = oe_batch["subclass"].to(self.device)
                oe_size = oe_images.size(0)

                images = torch.cat([images, oe_images], dim=0)
                super_targets = torch.cat([super_targets, oe_super], dim=0)
                sub_targets = torch.cat([sub_targets, oe_sub], dim=0)

                # OE samples: superclass is KNOWN (is_super_novel=False)
                # but subclass is NOVEL (is_sub_novel=True)
                is_super_novel = torch.cat([
                    is_super_novel,
                    torch.zeros(oe_size, dtype=torch.bool, device=self.device)
                ], dim=0)
                is_sub_novel = torch.cat([
                    is_sub_novel,
                    torch.ones(oe_size, dtype=torch.bool, device=self.device)
                ], dim=0)

            # Build local subclass targets for classification anchor
            local_sub_targets = {}
            for super_idx in range(self.config.model.num_superclasses):
                mask = super_targets == super_idx
                if mask.any():
                    global_subs = sub_targets[mask]
                    local_targets = torch.zeros_like(global_subs)
                    for i, gs in enumerate(global_subs):
                        local_idx = self.model.sub_head.global_to_local.get(super_idx, {}).get(gs.item(), 0)
                        local_targets[i] = local_idx
                    local_sub_targets[super_idx] = local_targets

            optimizer.zero_grad()

            if self.scaler is not None:
                with autocast():
                    output = self.model(images, super_targets=super_targets)

                    loss_out = self.loss_fn(
                        output.super_novelty_score,
                        output.sub_novelty_scores,
                        is_super_novel,
                        is_sub_novel,
                        output.super_logits,
                        output.per_head_sub_logits,
                        super_targets=super_targets,
                        sub_targets=sub_targets,
                        local_sub_targets=local_sub_targets,
                    )

                self.scaler.scale(loss_out.total).backward()

                if self.config.training.grad_clip > 0:
                    self.scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip,
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                output = self.model(images, super_targets=super_targets)

                loss_out = self.loss_fn(
                    output.super_novelty_score,
                    output.sub_novelty_scores,
                    is_super_novel,
                    is_sub_novel,
                    output.super_logits,
                    output.per_head_sub_logits,
                    super_targets=super_targets,
                    sub_targets=sub_targets,
                    local_sub_targets=local_sub_targets,
                )

                loss_out.total.backward()

                if self.config.training.grad_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.grad_clip,
                    )

                optimizer.step()

            scheduler.step()

            total_loss += loss_out.total.item()
            pbar.set_postfix({"loss": f"{loss_out.total.item():.4f}"})

        return {"loss": total_loss / len(self.dataloaders.train)}

    @torch.no_grad()
    def _validate(self) -> dict:
        self.model.eval()

        seen_super_correct = 0
        seen_sub_correct = 0
        seen_total = 0

        super_novel_tp = 0
        super_novel_fp = 0
        super_novel_fn = 0
        sub_novel_tp = 0
        sub_novel_fp = 0
        sub_novel_fn = 0

        for batch in self.dataloaders.val_seen:
            images = batch["image"].to(self.device)
            super_targets = batch["superclass"].to(self.device)
            sub_targets = batch["subclass"].to(self.device)

            output = self.model(images, super_targets=super_targets)

            super_preds = output.super_logits.argmax(dim=1)
            sub_preds = output.sub_logits.argmax(dim=1)

            seen_super_correct += (super_preds == super_targets).sum().item()
            seen_sub_correct += (sub_preds == sub_targets).sum().item()
            seen_total += images.size(0)

            # Super novelty false positives (seen samples predicted as novel)
            if output.super_novelty_score is not None:
                super_novel_pred = output.super_novelty_score > 0
                super_novel_fp += super_novel_pred.sum().item()

            # Sub novelty false positives (seen samples predicted as novel)
            if output.sub_novelty_scores is not None:
                for super_idx, (mask, scores) in output.sub_novelty_scores.items():
                    if scores.numel() > 0:
                        sub_novel_pred = scores > 0
                        sub_novel_fp += sub_novel_pred.sum().item()

        for batch in self.dataloaders.val_unseen:
            images = batch["image"].to(self.device)

            output = self.model(images)

            # Super novelty: unseen samples should all be detected as novel
            if output.super_novelty_score is not None:
                super_novel_pred = output.super_novelty_score > 0
                super_novel_tp += super_novel_pred.sum().item()
                super_novel_fn += (~super_novel_pred).sum().item()

            # Sub novelty: unseen samples should all be detected as novel
            if output.sub_novelty_scores is not None:
                for super_idx, (mask, scores) in output.sub_novelty_scores.items():
                    if scores.numel() > 0:
                        sub_novel_pred = scores > 0
                        sub_novel_tp += sub_novel_pred.sum().item()
                        sub_novel_fn += (~sub_novel_pred).sum().item()

        super_precision = super_novel_tp / max(super_novel_tp + super_novel_fp, 1)
        super_recall = super_novel_tp / max(super_novel_tp + super_novel_fn, 1)
        super_f1 = 2 * super_precision * super_recall / max(super_precision + super_recall, 1e-8)

        sub_precision = sub_novel_tp / max(sub_novel_tp + sub_novel_fp, 1)
        sub_recall = sub_novel_tp / max(sub_novel_tp + sub_novel_fn, 1)
        sub_f1 = 2 * sub_precision * sub_recall / max(sub_precision + sub_recall, 1e-8)

        return {
            "seen_super_acc": 100.0 * seen_super_correct / max(seen_total, 1),
            "seen_sub_acc": 100.0 * seen_sub_correct / max(seen_total, 1),
            "super_novel_f1": super_f1,
            "sub_novel_f1": sub_f1,
        }

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        path = self.save_dir / filename
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "phase": "B",
        }, path)


def main():
    parser = argparse.ArgumentParser(description="Decoupled training for hierarchical classifier")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--version", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--phase-a-epochs", type=int, default=25)
    parser.add_argument("--phase-b-epochs", type=int, default=15)
    parser.add_argument("--phase-a-lr", type=float, default=1e-3)
    parser.add_argument("--phase-b-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.01)
    parser.add_argument("--novelty-hidden-dim", type=int, default=256)
    parser.add_argument("--energy-weight", type=float, default=0.5)
    parser.add_argument("--skip-phase-a", type=Path, default=None,
                        help="Skip Phase A and load from checkpoint")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    base_save_dir = config.logging.save_dir
    if args.version:
        version_name = args.version if args.version.startswith("v") else f"v{args.version}"
    else:
        version_num = get_next_version(base_save_dir)
        version_name = f"v{version_num}"

    save_dir = base_save_dir / version_name
    save_dir.mkdir(parents=True, exist_ok=True)

    config.logging.save_dir = save_dir
    config.to_yaml(save_dir / "config.yaml")

    set_seed(config.training.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Decoupled Hierarchical Classification Training")
    print("=" * 60)
    print(f"Version: {version_name}")
    print(f"Save dir: {save_dir}")
    print(f"Device: {device}")
    print(f"Phase A epochs: {args.phase_a_epochs}, LR: {args.phase_a_lr}")
    print(f"Phase B epochs: {args.phase_b_epochs}, LR: {args.phase_b_lr}")
    print(f"Backbone LR multiplier: {args.backbone_lr_mult}")
    print("=" * 60)

    print("\nCreating data splits...")
    splits = create_splits(
        config.data.data_dir,
        oe_per_super=config.data.oe_subclasses_per_super,
        unseen_per_super=config.data.unseen_val_subclasses_per_super,
        train_ratio=config.data.train_seen_ratio,
        seed=config.training.seed,
    )

    save_splits_info(splits, save_dir / "splits_info.json")

    print(f"  Train seen: {len(splits.train_seen)} images")
    print(f"  Val seen: {len(splits.val_seen)} images")
    print(f"  OE: {len(splits.oe)} images")
    print(f"  Unseen val: {len(splits.unseen_val)} images")

    print("\nCreating dataloaders...")
    loaders = create_dataloaders(config, splits)

    print("\nCreating model...")
    subclasses_per_super = get_subclasses_by_superclass(config.data.data_dir)
    model = create_decoupled_model(config.model, subclasses_per_super)
    model.to(device)

    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")

    if args.skip_phase_a:
        print(f"\nLoading Phase A checkpoint from {args.skip_phase_a}...")
        checkpoint = torch.load(args.skip_phase_a, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        phase_a_results = {"skipped": True}
    else:
        print("\n" + "=" * 60)
        print("PHASE A: Classification Training")
        print("=" * 60)

        phase_a_trainer = PhaseATrainer(
            model=model,
            config=config,
            dataloaders=loaders,
            device=device,
            save_dir=save_dir,
        )

        phase_a_results = phase_a_trainer.train(
            epochs=args.phase_a_epochs,
            lr=args.phase_a_lr,
        )

        print(f"\nPhase A complete: Best acc={phase_a_results['best_acc']:.2f}%")

        best_checkpoint = torch.load(save_dir / "phase_a_best.pt", map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

    print("\n" + "=" * 60)
    print("PHASE B: Novelty Detection Training")
    print("=" * 60)

    phase_b_trainer = PhaseBTrainer(
        model=model,
        config=config,
        dataloaders=loaders,
        device=device,
        save_dir=save_dir,
        backbone_lr_mult=args.backbone_lr_mult,
    )

    phase_b_results = phase_b_trainer.train(
        epochs=args.phase_b_epochs,
        lr=args.phase_b_lr,
        novelty_hidden_dim=args.novelty_hidden_dim,
        energy_weight=args.energy_weight,
    )

    print(f"\nPhase B complete: Best F1={phase_b_results['best_f1']:.3f}")

    print("\n" + "=" * 60)
    print("Calibration")
    print("=" * 60)

    best_checkpoint = torch.load(save_dir / "phase_b_best.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    calibration = calibrate_decoupled_model(
        model=model,
        seen_loader=loaders.val_seen,
        unseen_loader=loaders.val_unseen,
        config=config.inference,
        device=device,
        num_superclasses=config.model.num_superclasses,
    )

    print(f"  Super temperature: {calibration.temperature_super:.3f}")
    print(f"  MSP threshold super: {calibration.msp_threshold_super:.3f}")
    for i, t in calibration.temperatures_sub.items():
        print(f"  Sub[{i}] temperature: {t:.3f}, MSP threshold: {calibration.msp_thresholds_sub.get(i, 0.5):.3f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "calibration": {
            "temperature_super": calibration.temperature_super,
            "temperatures_sub": calibration.temperatures_sub,
            "msp_threshold_super": calibration.msp_threshold_super,
            "msp_thresholds_sub": calibration.msp_thresholds_sub,
            "threshold_novelty_super": calibration.threshold_novelty_super,
            "thresholds_novelty_sub": calibration.thresholds_novelty_sub,
        },
        "phase_a_results": phase_a_results,
        "phase_b_results": phase_b_results,
    }, save_dir / "final_model.pt")

    results = {
        "phase_a": phase_a_results,
        "phase_b": phase_b_results,
        "calibration": {
            "temperature_super": calibration.temperature_super,
            "msp_threshold_super": calibration.msp_threshold_super,
        },
    }

    with open(save_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Model saved to: {save_dir / 'final_model.pt'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
