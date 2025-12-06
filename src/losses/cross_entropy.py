import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)

        # Handle -inf in logits (from hierarchical masking)
        valid_mask = logits != float("-inf")
        num_valid = valid_mask.sum(dim=-1, keepdim=True).float()

        # Replace -inf with very negative number for softmax stability
        safe_logits = logits.clone()
        safe_logits[~valid_mask] = -1e9

        log_probs = F.log_softmax(safe_logits, dim=-1)

        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            # Spread smoothing only over valid classes
            smooth_per_class = self.smoothing / (num_valid - 1).clamp(min=1)
            smooth_targets = smooth_per_class * valid_mask.float()
            # Set target class probability
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
            # Zero out invalid positions
            smooth_targets = smooth_targets * valid_mask.float()

        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class OutlierExposureLoss(nn.Module):
    def __init__(
        self,
        novel_super_idx: int = 3,
        novel_sub_idx: int = 87,
        smoothing: float = 0.1,
    ):
        super().__init__()
        self.novel_super_idx = novel_super_idx
        self.novel_sub_idx = novel_sub_idx
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)

    def forward(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        super_targets: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size = super_logits.size(0)
        device = super_logits.device

        # Both heads should predict "novel" for OE samples
        super_novel_targets = torch.full(
            (batch_size,), self.novel_super_idx, dtype=torch.long, device=device
        )
        sub_novel_targets = torch.full(
            (batch_size,), self.novel_sub_idx, dtype=torch.long, device=device
        )

        super_loss = self.ce_loss(super_logits, super_novel_targets)
        sub_loss = self.ce_loss(sub_logits, sub_novel_targets)

        return super_loss + sub_loss
