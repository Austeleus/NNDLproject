#!/usr/bin/env python3
"""
Evaluate Mahalanobis-based novel detection on a trained model.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import Config
from src.data import create_dataloaders, load_class_mappings
from src.data.splits import load_splits_from_info
from src.models import create_model
from src.inference import MahalanobisPredictor
from src.training.metrics import MetricsCalculator


def evaluate_with_mahalanobis(
    predictor: MahalanobisPredictor,
    dataloader,
    name: str,
    is_unseen: bool = False,
) -> dict:
    model = predictor.model
    device = predictor.device
    config = predictor.config

    novel_super_idx = config.model.num_superclasses
    novel_sub_idx = config.model.num_subclasses

    all_super_preds = []
    all_sub_preds = []
    all_super_targets = []
    all_sub_targets = []

    model.eval()
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

    super_preds = torch.cat(all_super_preds)
    sub_preds = torch.cat(all_sub_preds)
    super_targets = torch.cat(all_super_targets)
    sub_targets = torch.cat(all_sub_targets)

    super_acc = (super_preds == super_targets).float().mean().item() * 100
    sub_acc = (sub_preds == sub_targets).float().mean().item() * 100

    if is_unseen:
        super_novel_correct = (super_preds == novel_super_idx).sum().item()
        sub_novel_correct = (sub_preds == novel_sub_idx).sum().item()
        total = len(super_preds)
        novel_super_recall = super_novel_correct / total
        novel_sub_recall = sub_novel_correct / total
    else:
        novel_super_recall = 0.0
        novel_sub_recall = 0.0
        super_novel_fp = (super_preds == novel_super_idx).sum().item()
        sub_novel_fp = (sub_preds == novel_sub_idx).sum().item()

    return {
        "super_acc": super_acc,
        "sub_acc": sub_acc,
        "novel_super_recall": novel_super_recall if is_unseen else None,
        "novel_sub_recall": novel_sub_recall if is_unseen else None,
        "super_novel_fp": super_novel_fp if not is_unseen else None,
        "sub_novel_fp": sub_novel_fp if not is_unseen else None,
        "total_samples": len(super_preds),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate with Mahalanobis detector")
    parser.add_argument(
        "--version",
        type=str,
        default="v5",
        help="Version to evaluate (e.g., 'v5')",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save results",
    )
    args = parser.parse_args()

    version_name = args.version if args.version.startswith("v") else f"v{args.version}"
    exp_dir = Path("experiments") / version_name

    if not exp_dir.exists():
        print(f"Error: {exp_dir} does not exist")
        return

    checkpoint_path = exp_dir / "best_model.pt"
    config_path = exp_dir / "config.yaml"
    splits_path = exp_dir / "splits_info.json"

    config = Config.from_yaml(config_path)
    config.data.num_workers = 4

    print("=" * 60)
    print("Mahalanobis-based Novel Detection Evaluation")
    print("=" * 60)
    print(f"Version: {version_name}")
    print(f"Checkpoint: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading model...")
    model = create_model(config.model)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    print("\nLoading data splits...")
    if splits_path.exists():
        splits = load_splits_from_info(config.data.data_dir, splits_path)
    else:
        from src.data import create_splits
        splits = create_splits(config.data.data_dir, seed=config.training.seed)

    loaders = create_dataloaders(config, splits)

    _, _, sub_to_super = load_class_mappings(config.data.data_dir)

    print("\nCreating Mahalanobis predictor...")
    predictor = MahalanobisPredictor(
        model=model,
        config=config,
        subclass_to_superclass=sub_to_super,
        temperature=1.0,
    )

    print("\nFitting Mahalanobis on training data...")
    predictor.fit(loaders.train)

    print("\nTuning thresholds on validation data...")
    predictor.tune_thresholds(loaders.val_seen, loaders.val_unseen)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    results = {}

    print("\nValidation (Seen classes):")
    val_seen_results = evaluate_with_mahalanobis(
        predictor, loaders.val_seen, "val_seen", is_unseen=False
    )
    results["val_seen"] = val_seen_results
    print(f"  Super accuracy: {val_seen_results['super_acc']:.2f}%")
    print(f"  Sub accuracy: {val_seen_results['sub_acc']:.2f}%")
    print(f"  Super false positives (novel): {val_seen_results['super_novel_fp']}")
    print(f"  Sub false positives (novel): {val_seen_results['sub_novel_fp']}")

    print("\nValidation (Unseen classes - should predict 'novel'):")
    val_unseen_results = evaluate_with_mahalanobis(
        predictor, loaders.val_unseen, "val_unseen", is_unseen=True
    )
    results["val_unseen"] = val_unseen_results
    print(f"  Super accuracy (novel detection): {val_unseen_results['super_acc']:.2f}%")
    print(f"  Sub accuracy (novel detection): {val_unseen_results['sub_acc']:.2f}%")
    print(f"  Novel super recall: {val_unseen_results['novel_super_recall']:.3f}")
    print(f"  Novel sub recall: {val_unseen_results['novel_sub_recall']:.3f}")

    n_seen = val_seen_results["total_samples"]
    n_unseen = val_unseen_results["total_samples"]

    overall_super = (
        val_seen_results["super_acc"] * n_seen +
        val_unseen_results["super_acc"] * n_unseen
    ) / (n_seen + n_unseen)

    overall_sub = (
        val_seen_results["sub_acc"] * n_seen +
        val_unseen_results["sub_acc"] * n_unseen
    ) / (n_seen + n_unseen)

    print("\nOverall:")
    print(f"  Super accuracy: {overall_super:.2f}%")
    print(f"  Sub accuracy: {overall_sub:.2f}%")

    results["overall"] = {
        "super_acc": overall_super,
        "sub_acc": overall_sub,
    }

    results["mahalanobis"] = {
        "threshold_super": predictor.threshold_super,
        "threshold_sub": predictor.threshold_sub,
    }

    mahal_path = exp_dir / "mahalanobis.pt"
    predictor.save(mahal_path)

    output_path = args.output or Path("reports") / f"eval_mahal_{version_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
