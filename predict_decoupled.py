#!/usr/bin/env python3
"""Generate predictions using decoupled model with learned novelty scores."""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.config import Config
from src.data import create_test_dataloader, load_class_mappings
from src.data.dataset import get_subclasses_by_superclass
from src.models import create_decoupled_model


def main():
    parser = argparse.ArgumentParser(description="Generate predictions with decoupled model")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--novelty-threshold-super", type=float, default=-2.0)
    parser.add_argument("--novelty-threshold-sub", type=float, default=-2.0)
    parser.add_argument("--use-msp-fallback", action="store_true",
                        help="Fall back to MSP if novelty scores unavailable")
    parser.add_argument("--msp-threshold", type=float, default=0.5)
    args = parser.parse_args()

    # Try to find config from checkpoint directory
    if args.config is None:
        checkpoint_dir = args.checkpoint.parent
        config_path = checkpoint_dir / "config.yaml"
        if config_path.exists():
            args.config = config_path
        else:
            args.config = Path("configs/default.yaml")

    config = Config.from_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Decoupled Model Prediction")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Novelty threshold super: {args.novelty_threshold_super}")
    print(f"Novelty threshold sub: {args.novelty_threshold_sub}")

    # Load model
    print("\nLoading model...")
    subclasses_per_super = get_subclasses_by_superclass(config.data.data_dir)
    model = create_decoupled_model(config.model, subclasses_per_super)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Check if model has novelty heads
    has_novelty_heads = any("novelty_head" in k for k in checkpoint["model_state_dict"].keys())
    if has_novelty_heads:
        model.init_phase_b()
        print("  Model has novelty heads - will use learned novelty scores")
    else:
        print("  Model does NOT have novelty heads - using MSP fallback")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load calibration if available
    calibration = checkpoint.get("calibration", {})
    novelty_thresh_super = calibration.get("threshold_novelty_super", args.novelty_threshold_super)
    novelty_threshs_sub = calibration.get("thresholds_novelty_sub", {})
    temperature_super = calibration.get("temperature_super", 1.0)
    temperatures_sub = calibration.get("temperatures_sub", {})
    msp_threshold_super = calibration.get("msp_threshold_super", args.msp_threshold)
    msp_thresholds_sub = calibration.get("msp_thresholds_sub", {})

    print(f"\nCalibration:")
    print(f"  Novelty threshold super: {novelty_thresh_super}")
    print(f"  Temperature super: {temperature_super}")

    # Load class mappings
    print("\nLoading class mappings...")
    superclass_names, subclass_names, sub_to_super = load_class_mappings(config.data.data_dir)
    superclass_names[config.model.num_superclasses] = "novel"
    subclass_names[config.model.num_subclasses] = "novel"

    novel_super_idx = config.model.num_superclasses
    novel_sub_idx = config.model.num_subclasses

    # Create test dataloader
    print("\nCreating test dataloader...")
    test_loader = create_test_dataloader(config)
    print(f"  Test images: {len(test_loader.dataset)}")

    # Generate predictions
    print("\nGenerating predictions...")
    all_filenames = []
    all_super_preds = []
    all_sub_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            images = batch["image"].to(device)
            filenames = batch["filename"]
            batch_size = images.size(0)

            output = model(images)

            # Superclass prediction
            super_probs = F.softmax(output.super_logits / temperature_super, dim=1)
            super_msp, super_class_preds = super_probs.max(dim=1)
            super_preds = super_class_preds.clone()

            # Use novelty scores if available
            if has_novelty_heads and output.super_novelty_score is not None:
                super_novel_mask = output.super_novelty_score > novelty_thresh_super
                super_preds[super_novel_mask] = novel_super_idx
            elif args.use_msp_fallback:
                super_preds[super_msp < msp_threshold_super] = novel_super_idx

            # Subclass prediction
            sub_preds = torch.full((batch_size,), novel_sub_idx, dtype=torch.long, device=device)

            for super_idx in range(config.model.num_superclasses):
                mask = super_class_preds == super_idx
                if not mask.any():
                    continue

                if output.per_head_sub_logits is not None:
                    local_logits = output.per_head_sub_logits.get(super_idx)
                    if local_logits is not None:
                        temp = temperatures_sub.get(super_idx, temperatures_sub.get(str(super_idx), 1.0))
                        local_probs = F.softmax(local_logits / temp, dim=1)
                        local_msp, local_preds = local_probs.max(dim=1)

                        mask_indices = mask.nonzero(as_tuple=True)[0]

                        # Get novelty scores for this head
                        sub_novelty_scores = None
                        if has_novelty_heads and output.sub_novelty_scores is not None:
                            sub_data = output.sub_novelty_scores.get(super_idx)
                            if sub_data is not None:
                                _, sub_novelty_scores = sub_data

                        novelty_thresh = novelty_threshs_sub.get(
                            super_idx, novelty_threshs_sub.get(str(super_idx), args.novelty_threshold_sub)
                        )
                        msp_thresh = msp_thresholds_sub.get(
                            super_idx, msp_thresholds_sub.get(str(super_idx), args.msp_threshold)
                        )

                        for i, idx in enumerate(mask_indices):
                            is_novel = False
                            if sub_novelty_scores is not None and i < len(sub_novelty_scores):
                                is_novel = sub_novelty_scores[i].item() > novelty_thresh
                            elif args.use_msp_fallback:
                                is_novel = local_msp[i].item() < msp_thresh

                            if not is_novel:
                                local_pred = local_preds[i].item()
                                global_idx = model.sub_head.local_to_global[super_idx].get(local_pred)
                                if global_idx is not None:
                                    sub_preds[idx] = global_idx

            all_filenames.extend(filenames)
            all_super_preds.extend(super_preds.cpu().tolist())
            all_sub_preds.extend(sub_preds.cpu().tolist())

    # Create submission
    print("\nCreating submission file...")
    rows = []
    for fname, sp, subp in zip(all_filenames, all_super_preds, all_sub_preds):
        super_name = superclass_names.get(sp, "novel")
        sub_name = subclass_names.get(subp, "novel")
        img_id = fname.replace(".jpg", "").replace(".png", "")

        rows.append({
            "id": img_id,
            "superclass_index": sp,
            "subclass_index": subp,
            "superclass": super_name,
            "subclass": sub_name,
        })

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"\nSubmission saved to {args.output}")
    print(f"  Total predictions: {len(df)}")

    print("\nPrediction distribution:")
    print("  Superclass:")
    for name, count in df["superclass"].value_counts().items():
        print(f"    {name}: {count} ({count/len(df)*100:.1f}%)")

    print("  Subclass (top 10):")
    for name, count in df["subclass"].value_counts().head(10).items():
        print(f"    {name}: {count} ({count/len(df)*100:.1f}%)")

    novel_super_pct = (df["superclass"] == "novel").mean() * 100
    novel_sub_pct = (df["subclass"] == "novel").mean() * 100
    print(f"\n  Novel predictions: super={novel_super_pct:.1f}%, sub={novel_sub_pct:.1f}%")


if __name__ == "__main__":
    main()
