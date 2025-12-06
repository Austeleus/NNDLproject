#!/usr/bin/env python3
import argparse
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.data import create_splits, create_dataloaders, load_class_mappings
from src.data.splits import save_splits_info
from src.data.dataset import get_subclasses_by_superclass
from src.models import create_model
from src.training import Trainer
from src.inference import calibrate_model, HierarchicalPredictor
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
    """Find the next available version number."""
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


def main():
    parser = argparse.ArgumentParser(description="Train hierarchical classifier")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to config file",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (used in WandB)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version name (e.g., 'v5'). If not provided, auto-increments.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        choices=["resnet50", "resnet34", "efficientnet_b0"],
        help="Override backbone from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    args = parser.parse_args()

    config = Config.from_yaml(args.config)

    if args.experiment_name:
        config.logging.experiment_name = args.experiment_name
    if args.backbone:
        config.model.backbone = args.backbone
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
        config.training.phase1_lr = args.lr
        config.training.phase2_lr = args.lr / 10
        config.training.phase3_lr = args.lr / 100

    # Set up versioned experiment directory
    base_save_dir = config.logging.save_dir
    if args.version:
        version_name = args.version if args.version.startswith("v") else f"v{args.version}"
    else:
        version_num = get_next_version(base_save_dir)
        version_name = f"v{version_num}"

    config.logging.save_dir = base_save_dir / version_name
    config.logging.save_dir.mkdir(parents=True, exist_ok=True)

    # Save the config used for this run
    config.to_yaml(config.logging.save_dir / "config.yaml")

    set_seed(config.training.seed)

    print("=" * 60)
    print("Hierarchical Open-Set Image Classification")
    print("=" * 60)
    print(f"Version: {version_name}")
    print(f"Save dir: {config.logging.save_dir}")
    print(f"Backbone: {config.model.backbone}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.lr}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)

    print("\nCreating data splits...")
    splits = create_splits(
        config.data.data_dir,
        oe_per_super=config.data.oe_subclasses_per_super,
        unseen_per_super=config.data.unseen_val_subclasses_per_super,
        train_ratio=config.data.train_seen_ratio,
        seed=config.training.seed,
    )

    splits_path = config.logging.save_dir / "splits_info.json"
    config.logging.save_dir.mkdir(parents=True, exist_ok=True)
    save_splits_info(splits, splits_path)

    print(f"  Train seen: {len(splits.train_seen)} images ({len(splits.seen_subclasses)} subclasses)")
    print(f"  Val seen: {len(splits.val_seen)} images")
    print(f"  OE: {len(splits.oe)} images ({len(splits.oe_subclasses)} subclasses)")
    print(f"  Unseen val: {len(splits.unseen_val)} images ({len(splits.unseen_subclasses)} subclasses)")

    print("\nCreating dataloaders...")
    loaders = create_dataloaders(config, splits)
    print(f"  Train batches: {len(loaders.train)}")
    print(f"  Val seen batches: {len(loaders.val_seen)}")
    print(f"  Val unseen batches: {len(loaders.val_unseen)}")
    print(f"  OE batches: {len(loaders.oe)}")

    print("\nCreating model...")
    model = create_model(config.model)
    params = model.count_parameters()
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Backbone: {params['backbone']:,}")
    print(f"  Super head: {params['super_head']:,}")
    print(f"  Sub head: {params['sub_head']:,}")

    print("\nSetting up logging...")
    logger = setup_logging(config, enabled=not args.no_wandb)

    print("\nStarting training...")
    trainer = Trainer(
        model=model,
        config=config,
        dataloaders=loaders,
        logger=logger,
    )

    results = trainer.train()

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"  Best epoch: {results['best_epoch']}")
    print("=" * 60)

    print("\nRunning calibration on best model...")
    best_model_path = config.logging.save_dir / "best_model.pt"
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    device = next(model.parameters()).device
    calibration = calibrate_model(
        model=model,
        seen_loader=loaders.val_seen,
        unseen_loader=loaders.val_unseen,
        config=config.inference,
        device=device,
    )

    calibration_path = config.logging.save_dir / "calibration.pt"
    torch.save({
        "temperature": calibration.temperature,
        "threshold_super": calibration.threshold_super,
        "threshold_sub": calibration.threshold_sub,
    }, calibration_path)

    print(f"\nCalibration saved to {calibration_path}")
    print(f"  Temperature: {calibration.temperature:.3f}")
    print(f"  Threshold super: {calibration.threshold_super:.3f}")
    print(f"  Threshold sub: {calibration.threshold_sub:.3f}")
    print(f"  Val CE super: {calibration.val_ce_super:.3f}")
    print(f"  Val CE sub: {calibration.val_ce_sub:.3f}")

    logger.finish()

    print("\nDone!")


if __name__ == "__main__":
    main()
