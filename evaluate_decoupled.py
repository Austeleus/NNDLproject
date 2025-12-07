#!/usr/bin/env python3
"""Evaluate decoupled model with per-head calibration."""
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
from src.models import create_decoupled_model


def evaluate_model(
    model,
    dataloader,
    device,
    name: str,
    is_unseen: bool = False,
    msp_threshold_super: float = 0.5,
    msp_thresholds_sub: dict = None,
    temperature_super: float = 1.0,
    temperatures_sub: dict = None,
    novelty_threshold_super: float = -2.0,
    novelty_thresholds_sub: dict = None,
    use_novelty_scores: bool = True,
    num_superclasses: int = 3,
    num_subclasses: int = 87,
) -> dict:
    model.eval()

    if msp_thresholds_sub is None:
        msp_thresholds_sub = {i: 0.5 for i in range(num_superclasses)}
    if temperatures_sub is None:
        temperatures_sub = {i: 1.0 for i in range(num_superclasses)}
    if novelty_thresholds_sub is None:
        novelty_thresholds_sub = {i: -2.0 for i in range(num_superclasses)}

    novel_super_idx = num_superclasses
    novel_sub_idx = num_subclasses

    all_super_preds = []
    all_sub_preds = []
    all_super_targets = []
    all_sub_targets = []

    all_super_msp = []
    all_sub_msp = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
            images = batch["image"].to(device)
            super_targets = batch["superclass"]
            sub_targets = batch["subclass"]

            if is_unseen:
                super_targets = torch.full_like(super_targets, novel_super_idx)
                sub_targets = torch.full_like(sub_targets, novel_sub_idx)

            output = model(images)

            super_probs = F.softmax(output.super_logits / temperature_super, dim=1)
            super_msp, super_class_preds = super_probs.max(dim=1)

            super_preds = super_class_preds.clone()

            # Use learned novelty scores if available, otherwise fall back to MSP
            if use_novelty_scores and output.super_novelty_score is not None:
                super_novel_mask = output.super_novelty_score > novelty_threshold_super
                super_preds[super_novel_mask] = novel_super_idx
            else:
                super_preds[super_msp < msp_threshold_super] = novel_super_idx

            sub_preds = torch.full((images.size(0),), novel_sub_idx, dtype=torch.long, device=device)
            sub_msp_values = torch.zeros(images.size(0), device=device)

            for super_idx in range(num_superclasses):
                mask = super_class_preds == super_idx
                if not mask.any():
                    continue

                if output.per_head_sub_logits is not None:
                    local_logits = output.per_head_sub_logits.get(super_idx)
                    if local_logits is not None:
                        temp = temperatures_sub.get(super_idx, 1.0)
                        local_probs = F.softmax(local_logits / temp, dim=1)
                        local_msp, local_preds = local_probs.max(dim=1)

                        mask_indices = mask.nonzero(as_tuple=True)[0]
                        sub_msp_values[mask_indices] = local_msp

                        # Get novelty scores if available
                        sub_novelty_scores = None
                        if use_novelty_scores and output.sub_novelty_scores is not None:
                            sub_novelty_data = output.sub_novelty_scores.get(super_idx)
                            if sub_novelty_data is not None:
                                _, sub_novelty_scores = sub_novelty_data

                        novelty_thresh = novelty_thresholds_sub.get(super_idx, -2.0)
                        msp_thresh = msp_thresholds_sub.get(super_idx, 0.5)

                        for i, idx in enumerate(mask_indices):
                            # Check if sample is novel using learned scores or MSP
                            is_novel = False
                            if sub_novelty_scores is not None and i < len(sub_novelty_scores):
                                is_novel = sub_novelty_scores[i].item() > novelty_thresh
                            else:
                                is_novel = local_msp[i].item() < msp_thresh

                            if not is_novel:
                                local_pred = local_preds[i].item()
                                global_idx = model.sub_head.local_to_global[super_idx].get(local_pred)
                                if global_idx is not None:
                                    sub_preds[idx] = global_idx

            all_super_preds.append(super_preds.cpu())
            all_sub_preds.append(sub_preds.cpu())
            all_super_targets.append(super_targets)
            all_sub_targets.append(sub_targets)
            all_super_msp.append(super_msp.cpu())
            all_sub_msp.append(sub_msp_values.cpu())

    super_preds = torch.cat(all_super_preds)
    sub_preds = torch.cat(all_sub_preds)
    super_targets = torch.cat(all_super_targets)
    sub_targets = torch.cat(all_sub_targets)
    super_msp = torch.cat(all_super_msp)
    sub_msp = torch.cat(all_sub_msp)

    super_acc = (super_preds == super_targets).float().mean().item() * 100
    sub_acc = (sub_preds == sub_targets).float().mean().item() * 100

    super_novel_tp = ((super_preds == novel_super_idx) & (super_targets == novel_super_idx)).sum().item()
    super_novel_fp = ((super_preds == novel_super_idx) & (super_targets != novel_super_idx)).sum().item()
    super_novel_fn = ((super_preds != novel_super_idx) & (super_targets == novel_super_idx)).sum().item()

    super_precision = super_novel_tp / max(super_novel_tp + super_novel_fp, 1)
    super_recall = super_novel_tp / max(super_novel_tp + super_novel_fn, 1)
    super_novel_f1 = 2 * super_precision * super_recall / max(super_precision + super_recall, 1e-8)

    sub_novel_tp = ((sub_preds == novel_sub_idx) & (sub_targets == novel_sub_idx)).sum().item()
    sub_novel_fp = ((sub_preds == novel_sub_idx) & (sub_targets != novel_sub_idx)).sum().item()
    sub_novel_fn = ((sub_preds != novel_sub_idx) & (sub_targets == novel_sub_idx)).sum().item()

    sub_precision = sub_novel_tp / max(sub_novel_tp + sub_novel_fp, 1)
    sub_recall = sub_novel_tp / max(sub_novel_tp + sub_novel_fn, 1)
    sub_novel_f1 = 2 * sub_precision * sub_recall / max(sub_precision + sub_recall, 1e-8)

    return {
        "super_acc": super_acc,
        "sub_acc": sub_acc,
        "super_novel_f1": super_novel_f1,
        "sub_novel_f1": sub_novel_f1,
        "super_novel_precision": super_precision,
        "super_novel_recall": super_recall,
        "sub_novel_precision": sub_precision,
        "sub_novel_recall": sub_recall,
        "total_samples": len(super_preds),
        "avg_super_msp": super_msp.mean().item(),
        "avg_sub_msp": sub_msp.mean().item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate decoupled model")
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="final_model.pt")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    version_name = args.version if args.version.startswith("v") else f"v{args.version}"
    exp_dir = Path("experiments") / version_name

    config_path = exp_dir / "config.yaml"
    checkpoint_path = exp_dir / args.checkpoint
    splits_path = exp_dir / "splits_info.json"
    output_path = args.output or Path("reports") / f"eval_decoupled_{version_name}.json"

    config = Config.from_yaml(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Decoupled Model Evaluation")
    print("=" * 60)
    print(f"Version: {version_name}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")

    print("\nLoading model...")
    subclasses_per_super = get_subclasses_by_superclass(config.data.data_dir)
    model = create_decoupled_model(config.model, subclasses_per_super)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    has_novelty_heads = any("novelty_head" in k for k in checkpoint["model_state_dict"].keys())
    if has_novelty_heads:
        model.init_phase_b()

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    calibration = checkpoint.get("calibration", {})
    temperature_super = calibration.get("temperature_super", 1.0)
    temperatures_sub = calibration.get("temperatures_sub", {})
    msp_threshold_super = calibration.get("msp_threshold_super", 0.5)
    msp_thresholds_sub = calibration.get("msp_thresholds_sub", {})
    novelty_threshold_super = calibration.get("threshold_novelty_super", -2.0)
    novelty_thresholds_sub = calibration.get("thresholds_novelty_sub", {})

    # Use novelty scores by default if model has novelty heads
    use_novelty_scores = has_novelty_heads

    print(f"\nCalibration:")
    print(f"  Temperature super: {temperature_super}")
    print(f"  MSP threshold super: {msp_threshold_super}")
    print(f"  Novelty threshold super: {novelty_threshold_super}")
    print(f"  Using novelty scores: {use_novelty_scores}")
    for i in range(config.model.num_superclasses):
        t = temperatures_sub.get(i, temperatures_sub.get(str(i), 1.0))
        msp_thresh = msp_thresholds_sub.get(i, msp_thresholds_sub.get(str(i), 0.5))
        nov_thresh = novelty_thresholds_sub.get(i, novelty_thresholds_sub.get(str(i), -2.0))
        print(f"  Sub[{i}]: temp={t}, msp_thresh={msp_thresh}, novelty_thresh={nov_thresh}")

    temps_sub_int = {int(k): v for k, v in temperatures_sub.items()}
    threshs_sub_int = {int(k): v for k, v in msp_thresholds_sub.items()}
    novelty_threshs_sub_int = {int(k): v for k, v in novelty_thresholds_sub.items()}

    print("\nLoading data...")
    splits = load_splits_from_info(config.data.data_dir, splits_path)
    loaders = create_dataloaders(config, splits)

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    results = {}

    print("\nValidation (Seen classes):")
    val_seen_results = evaluate_model(
        model, loaders.val_seen, device, "val_seen",
        is_unseen=False,
        msp_threshold_super=msp_threshold_super,
        msp_thresholds_sub=threshs_sub_int,
        temperature_super=temperature_super,
        temperatures_sub=temps_sub_int,
        novelty_threshold_super=novelty_threshold_super,
        novelty_thresholds_sub=novelty_threshs_sub_int,
        use_novelty_scores=use_novelty_scores,
        num_superclasses=config.model.num_superclasses,
        num_subclasses=config.model.num_subclasses,
    )
    results["val_seen"] = val_seen_results
    print(f"  Super accuracy: {val_seen_results['super_acc']:.2f}%")
    print(f"  Sub accuracy: {val_seen_results['sub_acc']:.2f}%")
    print(f"  Avg super MSP: {val_seen_results['avg_super_msp']:.3f}")
    print(f"  Avg sub MSP: {val_seen_results['avg_sub_msp']:.3f}")

    print("\nValidation (Unseen classes - should predict 'novel'):")
    val_unseen_results = evaluate_model(
        model, loaders.val_unseen, device, "val_unseen",
        is_unseen=True,
        msp_threshold_super=msp_threshold_super,
        msp_thresholds_sub=threshs_sub_int,
        temperature_super=temperature_super,
        temperatures_sub=temps_sub_int,
        novelty_threshold_super=novelty_threshold_super,
        novelty_thresholds_sub=novelty_threshs_sub_int,
        use_novelty_scores=use_novelty_scores,
        num_superclasses=config.model.num_superclasses,
        num_subclasses=config.model.num_subclasses,
    )
    results["val_unseen"] = val_unseen_results
    print(f"  Super accuracy (novel detection): {val_unseen_results['super_acc']:.2f}%")
    print(f"  Sub accuracy (novel detection): {val_unseen_results['sub_acc']:.2f}%")
    print(f"  Super novel F1: {val_unseen_results['super_novel_f1']:.4f}")
    print(f"  Sub novel F1: {val_unseen_results['sub_novel_f1']:.4f}")
    print(f"  Avg super MSP: {val_unseen_results['avg_super_msp']:.3f}")
    print(f"  Avg sub MSP: {val_unseen_results['avg_sub_msp']:.3f}")

    total_samples = val_seen_results["total_samples"] + val_unseen_results["total_samples"]
    overall_super = (
        val_seen_results["super_acc"] * val_seen_results["total_samples"] +
        val_unseen_results["super_acc"] * val_unseen_results["total_samples"]
    ) / total_samples

    overall_sub = (
        val_seen_results["sub_acc"] * val_seen_results["total_samples"] +
        val_unseen_results["sub_acc"] * val_unseen_results["total_samples"]
    ) / total_samples

    print("\nOverall:")
    print(f"  Super accuracy: {overall_super:.2f}%")
    print(f"  Sub accuracy: {overall_sub:.2f}%")

    results["overall"] = {
        "super_acc": overall_super,
        "sub_acc": overall_sub,
    }

    results["calibration"] = calibration

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 60)
    print("Comparison with baseline (CLIP zero-shot):")
    print("  Baseline super: 80% overall (99% seen, 33% unseen)")
    print("  Baseline sub: 13% overall (61% seen, 0.24% unseen)")
    print("=" * 60)


if __name__ == "__main__":
    main()
