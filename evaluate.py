#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import Config
from src.data import create_dataloaders, load_class_mappings
from src.data.splits import load_splits_from_info
from src.data.dataset import get_subclasses_by_superclass
from src.models import create_model
from src.inference import calibrate_model, HierarchicalPredictor, CalibrationResult
from src.training.metrics import MetricsCalculator


def evaluate_loader(
    predictor: HierarchicalPredictor,
    dataloader,
    name: str,
    is_unseen: bool = False,
) -> dict:
    model = predictor.model
    device = predictor.device
    config = predictor.config

    metrics = MetricsCalculator(
        num_superclasses=config.model.num_superclasses,
        num_subclasses=config.model.num_subclasses,
    )

    model.eval()
    novel_super_idx = config.model.num_superclasses
    novel_sub_idx = config.model.num_subclasses

    all_super_preds = []
    all_sub_preds = []
    all_super_targets = []
    all_sub_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            images = batch["image"].to(device)
            super_targets = batch["superclass"].to(device)
            sub_targets = batch["subclass"].to(device)

            if is_unseen:
                super_targets = torch.full_like(super_targets, novel_super_idx)
                sub_targets = torch.full_like(sub_targets, novel_sub_idx)

            super_preds, sub_preds = predictor.predict_batch(images)

            all_super_preds.append(super_preds.cpu())
            all_sub_preds.append(sub_preds.cpu())
            all_super_targets.append(super_targets.cpu())
            all_sub_targets.append(sub_targets.cpu())

            output = model(images)
            metrics.update(
                output.super_logits,
                output.sub_logits,
                super_targets,
                sub_targets,
            )

    super_preds = torch.cat(all_super_preds)
    sub_preds = torch.cat(all_sub_preds)
    super_targets = torch.cat(all_super_targets)
    sub_targets = torch.cat(all_sub_targets)

    super_acc = (super_preds == super_targets).float().mean().item() * 100
    sub_acc = (sub_preds == sub_targets).float().mean().item() * 100

    epoch_metrics = metrics.compute()

    return {
        "super_acc": super_acc,
        "sub_acc": sub_acc,
        "super_ce": epoch_metrics.super_ce,
        "sub_ce": epoch_metrics.sub_ce,
        "novel_super_f1": epoch_metrics.novel_super_f1,
        "novel_sub_f1": epoch_metrics.novel_sub_f1,
        "total_samples": len(super_preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Version to evaluate (e.g., 'v5'). Will look in experiments/v5/",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to model checkpoint (overrides --version)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (auto-detected if using --version)",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Path to calibration file (if not provided, will recalibrate)",
    )
    parser.add_argument(
        "--splits-info",
        type=Path,
        default=None,
        help="Path to splits_info.json (for reproducible splits)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save evaluation results JSON",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Force recalibration even if calibration file exists",
    )
    args = parser.parse_args()

    # Handle versioned experiments
    if args.version:
        version_name = args.version if args.version.startswith("v") else f"v{args.version}"
        exp_dir = Path("experiments") / version_name
        if not exp_dir.exists():
            print(f"Error: Experiment directory {exp_dir} does not exist")
            return
        args.checkpoint = args.checkpoint or exp_dir / "best_model.pt"
        args.config = args.config or exp_dir / "config.yaml"
        args.calibration = args.calibration or exp_dir / "calibration.pt"
        args.splits_info = args.splits_info or exp_dir / "splits_info.json"
        args.output = args.output or Path("reports") / f"eval_{version_name}.json"

    if not args.checkpoint:
        print("Error: Must provide either --version or --checkpoint")
        return

    if not args.config:
        args.config = Path("configs/default.yaml")

    config = Config.from_yaml(args.config)
    config.data.num_workers = 4

    print("=" * 60)
    print("Model Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading model...")
    model = create_model(config.model)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best val acc: {checkpoint.get('best_val_acc', 'unknown')}")

    print("\nLoading data...")
    if args.splits_info and args.splits_info.exists():
        splits = load_splits_from_info(config.data.data_dir, args.splits_info)
        print(f"  Loaded splits from {args.splits_info}")
    else:
        from src.data import create_splits
        splits = create_splits(
            config.data.data_dir,
            seed=config.training.seed,
        )
        print("  Created new splits")

    loaders = create_dataloaders(config, splits)

    _, _, sub_to_super = load_class_mappings(config.data.data_dir)

    calibration = None
    if args.calibration and args.calibration.exists() and not args.recalibrate:
        print(f"\nLoading calibration from {args.calibration}...")
        cal_data = torch.load(args.calibration)
        calibration = CalibrationResult(
            temperature=cal_data["temperature"],
            threshold_super=cal_data["threshold_super"],
            threshold_sub=cal_data["threshold_sub"],
            val_ce_super=0.0,
            val_ce_sub=0.0,
            novel_f1_super=0.0,
            novel_f1_sub=0.0,
        )
        print(f"  Temperature: {calibration.temperature:.3f}")
        print(f"  Threshold super: {calibration.threshold_super:.3f}")
        print(f"  Threshold sub: {calibration.threshold_sub:.3f}")
    else:
        print("\nRunning calibration...")
        calibration = calibrate_model(
            model=model,
            seen_loader=loaders.val_seen,
            unseen_loader=loaders.val_unseen,
            config=config.inference,
            device=device,
        )

    predictor = HierarchicalPredictor(
        model=model,
        config=config,
        calibration=calibration,
        subclass_to_superclass=sub_to_super,
    )

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    results = {}

    print("\nValidation (Seen classes):")
    val_seen_results = evaluate_loader(predictor, loaders.val_seen, "val_seen", is_unseen=False)
    results["val_seen"] = val_seen_results
    print(f"  Super accuracy: {val_seen_results['super_acc']:.2f}%")
    print(f"  Sub accuracy: {val_seen_results['sub_acc']:.2f}%")
    print(f"  Super CE: {val_seen_results['super_ce']:.4f}")
    print(f"  Sub CE: {val_seen_results['sub_ce']:.4f}")

    print("\nValidation (Unseen classes - should predict 'novel'):")
    val_unseen_results = evaluate_loader(predictor, loaders.val_unseen, "val_unseen", is_unseen=True)
    results["val_unseen"] = val_unseen_results
    print(f"  Super accuracy: {val_unseen_results['super_acc']:.2f}%")
    print(f"  Sub accuracy: {val_unseen_results['sub_acc']:.2f}%")
    print(f"  Novel super F1: {val_unseen_results['novel_super_f1']:.4f}")
    print(f"  Novel sub F1: {val_unseen_results['novel_sub_f1']:.4f}")

    overall_super = (
        val_seen_results["super_acc"] * val_seen_results["total_samples"] +
        val_unseen_results["super_acc"] * val_unseen_results["total_samples"]
    ) / (val_seen_results["total_samples"] + val_unseen_results["total_samples"])

    overall_sub = (
        val_seen_results["sub_acc"] * val_seen_results["total_samples"] +
        val_unseen_results["sub_acc"] * val_unseen_results["total_samples"]
    ) / (val_seen_results["total_samples"] + val_unseen_results["total_samples"])

    print("\nOverall:")
    print(f"  Super accuracy: {overall_super:.2f}%")
    print(f"  Sub accuracy: {overall_sub:.2f}%")

    results["overall"] = {
        "super_acc": overall_super,
        "sub_acc": overall_sub,
    }

    results["calibration"] = {
        "temperature": calibration.temperature,
        "threshold_super": calibration.threshold_super,
        "threshold_sub": calibration.threshold_sub,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
