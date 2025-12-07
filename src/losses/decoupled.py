from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cross_entropy import LabelSmoothingCrossEntropy


@dataclass
class PhaseALossOutput:
    total: torch.Tensor
    super_ce: torch.Tensor
    sub_ce: torch.Tensor

    def to_dict(self) -> Dict[str, float]:
        return {
            "total": self.total.item(),
            "super_ce": self.super_ce.item(),
            "sub_ce": self.sub_ce.item(),
        }


@dataclass
class PhaseBLossOutput:
    total: torch.Tensor
    novelty_bce: torch.Tensor
    energy_margin: torch.Tensor
    classification_ce: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "total": self.total.item(),
            "novelty_bce": self.novelty_bce.item(),
            "energy_margin": self.energy_margin.item(),
        }
        if self.classification_ce is not None:
            d["classification_ce"] = self.classification_ce.item()
        return d


class PhaseALoss(nn.Module):
    """
    Pure classification loss for Phase A training.
    No novel class, no OE - just classification.
    """

    def __init__(
        self,
        label_smoothing: float = 0.1,
        lambda_super: float = 1.0,
        lambda_sub: float = 1.0,
    ):
        super().__init__()
        self.super_ce = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.sub_ce = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.lambda_super = lambda_super
        self.lambda_sub = lambda_sub

    def forward(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        super_targets: torch.Tensor,
        sub_targets: torch.Tensor,
    ) -> PhaseALossOutput:
        super_loss = self.super_ce(super_logits, super_targets)
        sub_loss = self.sub_ce(sub_logits, sub_targets)

        total = self.lambda_super * super_loss + self.lambda_sub * sub_loss

        return PhaseALossOutput(
            total=total,
            super_ce=super_loss,
            sub_ce=sub_loss,
        )


class PhaseBLoss(nn.Module):
    """
    Novelty detection loss for Phase B training.
    Binary classification (seen vs novel) with energy margin regularization.
    """

    def __init__(
        self,
        energy_margin: float = 5.0,
        lambda_energy_margin: float = 0.5,
        lambda_classification: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.energy_margin = energy_margin
        self.lambda_energy_margin = lambda_energy_margin
        self.lambda_classification = lambda_classification

        self.bce = nn.BCEWithLogitsLoss()
        self.classification_ce = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    def compute_energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return -temperature * torch.logsumexp(logits / temperature, dim=1)

    def forward(
        self,
        novelty_scores: torch.Tensor,
        is_novel: torch.Tensor,
        classifier_logits: Optional[torch.Tensor] = None,
        classifier_targets: Optional[torch.Tensor] = None,
        seen_mask: Optional[torch.Tensor] = None,
    ) -> PhaseBLossOutput:
        """
        Args:
            novelty_scores: [B] raw novelty scores (higher = more novel)
            is_novel: [B] binary labels (1 = novel/OE, 0 = seen)
            classifier_logits: [B, C] for energy margin computation
            classifier_targets: [B] for classification loss on seen samples
            seen_mask: [B] mask for seen samples
        """
        novelty_targets = is_novel.float()
        bce_loss = self.bce(novelty_scores, novelty_targets)

        energy_margin_loss = torch.tensor(0.0, device=novelty_scores.device)
        if classifier_logits is not None:
            energy = self.compute_energy(classifier_logits)

            seen_energy = energy[~is_novel] if (~is_novel).any() else None
            novel_energy = energy[is_novel] if is_novel.any() else None

            if seen_energy is not None and novel_energy is not None:
                # Novel samples should have higher energy (more uncertain)
                # seen_energy.mean() + margin < novel_energy.mean()
                margin_violation = F.relu(
                    self.energy_margin - (novel_energy.mean() - seen_energy.mean())
                )
                energy_margin_loss = margin_violation

        total = bce_loss + self.lambda_energy_margin * energy_margin_loss

        classification_loss = None
        if (
            self.lambda_classification > 0 and
            classifier_logits is not None and
            classifier_targets is not None and
            seen_mask is not None and
            seen_mask.any()
        ):
            classification_loss = self.classification_ce(
                classifier_logits[seen_mask],
                classifier_targets[seen_mask],
            )
            total = total + self.lambda_classification * classification_loss

        return PhaseBLossOutput(
            total=total,
            novelty_bce=bce_loss,
            energy_margin=energy_margin_loss,
            classification_ce=classification_loss,
        )


class PerHeadPhaseBLoss(nn.Module):
    """
    Per-head novelty detection loss for hierarchical Phase B training.
    Separate losses for superclass novelty and per-superclass subclass novelty.

    Key insight: OE samples belong to SEEN superclasses but UNSEEN subclasses.
    So is_super_novel should be False for OE, but is_sub_novel should be True.
    """

    def __init__(
        self,
        num_superclasses: int = 3,
        energy_margin: float = 5.0,
        lambda_energy_margin: float = 0.5,
        lambda_super: float = 1.0,
        lambda_sub: float = 1.0,
        lambda_classification: float = 0.1,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.num_superclasses = num_superclasses
        self.lambda_super = lambda_super
        self.lambda_sub = lambda_sub
        self.lambda_classification = lambda_classification
        self.energy_margin = energy_margin
        self.lambda_energy_margin = lambda_energy_margin

        self.bce = nn.BCEWithLogitsLoss()
        self.classification_ce = LabelSmoothingCrossEntropy(smoothing=label_smoothing)

    def compute_energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return -temperature * torch.logsumexp(logits / temperature, dim=1)

    def forward(
        self,
        super_novelty_score: torch.Tensor,
        sub_novelty_scores: Dict[int, tuple],
        is_super_novel: torch.Tensor,
        is_sub_novel: torch.Tensor,
        super_logits: torch.Tensor,
        per_head_sub_logits: Dict[int, torch.Tensor],
        super_targets: Optional[torch.Tensor] = None,
        sub_targets: Optional[torch.Tensor] = None,
        local_sub_targets: Optional[Dict[int, torch.Tensor]] = None,
    ) -> PhaseBLossOutput:
        """
        Args:
            super_novelty_score: [B] raw novelty scores for superclass head
            sub_novelty_scores: Dict[super_idx -> (mask, scores)] for subclass heads
            is_super_novel: [B] binary labels for superclass novelty (False for OE samples)
            is_sub_novel: [B] binary labels for subclass novelty (True for OE samples)
            super_logits: [B, num_super] superclass classification logits
            per_head_sub_logits: Dict[super_idx -> [N, local_num_sub]] per-head subclass logits
            super_targets: [B] superclass targets for classification anchor
            sub_targets: [B] global subclass targets (not used directly)
            local_sub_targets: Dict[super_idx -> [N]] local subclass targets per head
        """
        # Superclass novelty loss
        # Key insight: Use is_sub_novel for super novelty training (cascading uncertainty)
        # If subclass is novel, we should also flag superclass as "uncertain/novel"
        # This aligns with evaluation which expects both levels to detect novelty for OE samples
        super_bce = self.bce(super_novelty_score, is_sub_novel.float())

        # Energy margin for superclass (also based on is_sub_novel)
        super_energy_margin = torch.tensor(0.0, device=super_novelty_score.device)
        energy = self.compute_energy(super_logits)
        seen_mask = ~is_sub_novel
        novel_mask = is_sub_novel
        if seen_mask.any() and novel_mask.any():
            seen_energy = energy[seen_mask]
            novel_energy = energy[novel_mask]
            margin_violation = F.relu(
                self.energy_margin - (novel_energy.mean() - seen_energy.mean())
            )
            super_energy_margin = margin_violation

        super_loss = super_bce + self.lambda_energy_margin * super_energy_margin

        # Subclass novelty loss (per-head)
        sub_loss_total = torch.tensor(0.0, device=super_novelty_score.device)
        sub_count = 0

        for super_idx, (mask, scores) in sub_novelty_scores.items():
            if scores.numel() == 0:
                continue

            # Use is_sub_novel for subclass heads
            local_is_novel = is_sub_novel[mask]
            local_logits = per_head_sub_logits.get(super_idx)

            if local_logits is not None:
                # BCE loss for subclass novelty
                sub_bce = self.bce(scores, local_is_novel.float())

                # Energy margin for this subclass head
                sub_energy_margin = torch.tensor(0.0, device=scores.device)
                sub_energy = self.compute_energy(local_logits)
                local_seen = ~local_is_novel
                local_novel = local_is_novel
                if local_seen.any() and local_novel.any():
                    margin_violation = F.relu(
                        self.energy_margin - (sub_energy[local_novel].mean() - sub_energy[local_seen].mean())
                    )
                    sub_energy_margin = margin_violation

                sub_loss = sub_bce + self.lambda_energy_margin * sub_energy_margin
                sub_loss_total = sub_loss_total + sub_loss
                sub_count += 1

        if sub_count > 0:
            sub_loss_total = sub_loss_total / sub_count

        total = self.lambda_super * super_loss + self.lambda_sub * sub_loss_total

        # Classification anchor loss to prevent feature drift
        classification_loss = None
        if self.lambda_classification > 0:
            # Superclass classification on seen samples
            seen_super_mask = ~is_super_novel
            if seen_super_mask.any() and super_targets is not None:
                super_ce = self.classification_ce(
                    super_logits[seen_super_mask],
                    super_targets[seen_super_mask],
                )
                total = total + self.lambda_classification * super_ce
                classification_loss = super_ce

            # Subclass classification on seen samples (per-head)
            if local_sub_targets is not None:
                for super_idx, (mask, _) in sub_novelty_scores.items():
                    local_logits = per_head_sub_logits.get(super_idx)
                    local_targets = local_sub_targets.get(super_idx)
                    if local_logits is None or local_targets is None:
                        continue

                    local_seen = ~is_sub_novel[mask]
                    if local_seen.any():
                        sub_ce = self.classification_ce(
                            local_logits[local_seen],
                            local_targets[local_seen],
                        )
                        total = total + self.lambda_classification * sub_ce

        return PhaseBLossOutput(
            total=total,
            novelty_bce=super_bce,
            energy_margin=super_energy_margin,
            classification_ce=classification_loss,
        )
