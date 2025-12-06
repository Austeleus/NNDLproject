from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LossConfig
from ..models.classifier import ModelOutput
from .cross_entropy import LabelSmoothingCrossEntropy, OutlierExposureLoss


class MarginSeparationLoss(nn.Module):
    """
    Margin-based loss for novel class separation.
    For seen samples: max_known > novel_logit + margin
    For OE samples: novel_logit > max_known + margin
    """

    def __init__(self, margin: float = 2.0, num_known_classes: int = 87):
        super().__init__()
        self.margin = margin
        self.num_known = num_known_classes

    def forward(
        self,
        logits: torch.Tensor,
        is_oe: torch.Tensor,
    ) -> torch.Tensor:
        known_logits = logits[:, :self.num_known]
        novel_logit = logits[:, -1]
        max_known = known_logits.max(dim=1).values

        seen_mask = ~is_oe
        oe_mask = is_oe

        loss = torch.tensor(0.0, device=logits.device)

        if seen_mask.any():
            # Seen: max_known should be > novel + margin
            # Loss = max(0, novel + margin - max_known)
            seen_loss = F.relu(novel_logit[seen_mask] + self.margin - max_known[seen_mask])
            loss = loss + seen_loss.mean()

        if oe_mask.any():
            # OE: novel should be > max_known + margin
            # Loss = max(0, max_known + margin - novel)
            oe_loss = F.relu(max_known[oe_mask] + self.margin - novel_logit[oe_mask])
            loss = loss + oe_loss.mean()

        return loss


class EnergyMarginLoss(nn.Module):
    """
    Direct energy margin loss to ensure OE samples have higher energy than seen samples.
    Energy = -logsumexp(logits), higher energy = more uncertain.
    """

    def __init__(self, margin: float = 3.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        logits: torch.Tensor,
        is_oe: torch.Tensor,
    ) -> torch.Tensor:
        energy = -torch.logsumexp(logits, dim=1)

        seen_mask = ~is_oe
        oe_mask = is_oe

        if not seen_mask.any() or not oe_mask.any():
            return torch.tensor(0.0, device=logits.device)

        seen_energy = energy[seen_mask].mean()
        oe_energy = energy[oe_mask].mean()

        # OE energy should be higher than seen energy by at least margin
        # Loss = max(0, margin - (oe_energy - seen_energy))
        loss = F.relu(self.margin - (oe_energy - seen_energy))
        return loss


class CenterLoss(nn.Module):
    """
    Center loss to pull features of the same class toward their center.
    Creates tighter clusters, leaving empty space for novel samples.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        centers_batch = self.centers[labels]
        loss = ((features - centers_batch) ** 2).sum(dim=1).mean()
        return loss


class OERepulsionLoss(nn.Module):
    """
    Push OE sample features away from all class centers.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        features: torch.Tensor,
        centers: torch.Tensor,
    ) -> torch.Tensor:
        # Distance from each OE sample to all centers
        distances = torch.cdist(features, centers)  # [B, num_classes]
        min_dist = distances.min(dim=1).values
        # Push away: penalize if min distance < margin
        loss = F.relu(self.margin - min_dist).mean()
        return loss


@dataclass
class LossOutput:
    total: torch.Tensor
    super_ce: torch.Tensor
    sub_ce: torch.Tensor
    oe: Optional[torch.Tensor] = None
    margin: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, float]:
        d = {
            "total": self.total.item(),
            "super_ce": self.super_ce.item(),
            "sub_ce": self.sub_ce.item(),
        }
        if self.oe is not None:
            d["oe"] = self.oe.item()
        if self.margin is not None:
            d["margin"] = self.margin.item()
        return d


class HierarchicalLoss(nn.Module):
    def __init__(
        self,
        config: LossConfig,
        use_margin_loss: bool = False,
        margin: float = 2.0,
        lambda_margin: float = 0.5,
        num_subclasses: int = 87,
        num_superclasses: int = 3,
    ):
        super().__init__()
        self.config = config

        self.super_ce = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
        self.sub_ce = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
        self.oe_loss = OutlierExposureLoss(smoothing=config.label_smoothing)

        self.use_margin_loss = use_margin_loss
        if use_margin_loss:
            self.margin_loss_sub = MarginSeparationLoss(margin=margin, num_known_classes=num_subclasses)
            self.margin_loss_super = MarginSeparationLoss(margin=margin, num_known_classes=num_superclasses)
        self.lambda_margin = lambda_margin

        self.lambda_super = config.lambda_super
        self.lambda_sub = config.lambda_sub
        self.lambda_oe = config.lambda_oe

    def forward(
        self,
        model_output: ModelOutput,
        super_targets: torch.Tensor,
        sub_targets: torch.Tensor,
        is_oe: Optional[torch.Tensor] = None,
    ) -> LossOutput:
        if is_oe is not None and is_oe.any():
            seen_mask = ~is_oe
            oe_mask = is_oe

            if seen_mask.any():
                super_loss = self.super_ce(
                    model_output.super_logits[seen_mask],
                    super_targets[seen_mask],
                )
                sub_loss = self.sub_ce(
                    model_output.sub_logits[seen_mask],
                    sub_targets[seen_mask],
                )
            else:
                super_loss = torch.tensor(0.0, device=model_output.super_logits.device)
                sub_loss = torch.tensor(0.0, device=model_output.sub_logits.device)

            if oe_mask.any():
                oe_loss = self.oe_loss(
                    model_output.super_logits[oe_mask],
                    model_output.sub_logits[oe_mask],
                    super_targets[oe_mask],
                )
            else:
                oe_loss = torch.tensor(0.0, device=model_output.super_logits.device)

            total = (
                self.lambda_super * super_loss +
                self.lambda_sub * sub_loss +
                self.lambda_oe * oe_loss
            )

            margin_loss = None
            if self.use_margin_loss:
                margin_loss_sub = self.margin_loss_sub(model_output.sub_logits, is_oe)
                margin_loss_super = self.margin_loss_super(model_output.super_logits, is_oe)
                margin_loss = margin_loss_sub + margin_loss_super
                total = total + self.lambda_margin * margin_loss

            return LossOutput(
                total=total,
                super_ce=super_loss,
                sub_ce=sub_loss,
                oe=oe_loss,
                margin=margin_loss,
            )
        else:
            super_loss = self.super_ce(model_output.super_logits, super_targets)
            sub_loss = self.sub_ce(model_output.sub_logits, sub_targets)

            total = self.lambda_super * super_loss + self.lambda_sub * sub_loss

            return LossOutput(
                total=total,
                super_ce=super_loss,
                sub_ce=sub_loss,
                oe=None,
            )

    def forward_oe_only(
        self,
        model_output: ModelOutput,
        super_targets: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.oe_loss(model_output.super_logits, model_output.sub_logits, super_targets)
