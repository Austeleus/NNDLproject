from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn.functional as F


@dataclass
class MetricsAccumulator:
    super_correct: int = 0
    super_total: int = 0
    sub_correct: int = 0
    sub_total: int = 0

    super_ce_sum: float = 0.0
    sub_ce_sum: float = 0.0
    ce_count: int = 0

    seen_super_correct: int = 0
    seen_super_total: int = 0
    unseen_super_correct: int = 0
    unseen_super_total: int = 0

    seen_sub_correct: int = 0
    seen_sub_total: int = 0
    unseen_sub_correct: int = 0
    unseen_sub_total: int = 0

    novel_super_tp: int = 0
    novel_super_fp: int = 0
    novel_super_fn: int = 0
    novel_sub_tp: int = 0
    novel_sub_fp: int = 0
    novel_sub_fn: int = 0


@dataclass
class EpochMetrics:
    super_acc: float
    sub_acc: float
    super_ce: float
    sub_ce: float

    seen_super_acc: float = 0.0
    unseen_super_acc: float = 0.0
    seen_sub_acc: float = 0.0
    unseen_sub_acc: float = 0.0

    novel_super_f1: float = 0.0
    novel_sub_f1: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "super_acc": self.super_acc,
            "sub_acc": self.sub_acc,
            "super_ce": self.super_ce,
            "sub_ce": self.sub_ce,
            "seen_super_acc": self.seen_super_acc,
            "unseen_super_acc": self.unseen_super_acc,
            "seen_sub_acc": self.seen_sub_acc,
            "unseen_sub_acc": self.unseen_sub_acc,
            "novel_super_f1": self.novel_super_f1,
            "novel_sub_f1": self.novel_sub_f1,
        }


class MetricsCalculator:
    def __init__(
        self,
        num_superclasses: int = 3,
        num_subclasses: int = 87,
        seen_subclasses: Optional[Set[int]] = None,
    ):
        self.num_superclasses = num_superclasses
        self.num_subclasses = num_subclasses
        self.novel_super_idx = num_superclasses
        self.novel_sub_idx = num_subclasses
        self.seen_subclasses = seen_subclasses or set()

        self.reset()

    def reset(self) -> None:
        self.acc = MetricsAccumulator()

    def update(
        self,
        super_logits: torch.Tensor,
        sub_logits: torch.Tensor,
        super_targets: torch.Tensor,
        sub_targets: torch.Tensor,
        is_unseen: Optional[torch.Tensor] = None,
    ) -> None:
        super_preds = super_logits.argmax(dim=1)
        sub_preds = sub_logits.argmax(dim=1)

        batch_size = super_logits.size(0)

        super_correct = (super_preds == super_targets).sum().item()
        sub_correct = (sub_preds == sub_targets).sum().item()

        self.acc.super_correct += super_correct
        self.acc.super_total += batch_size
        self.acc.sub_correct += sub_correct
        self.acc.sub_total += batch_size

        super_probs = F.softmax(super_logits, dim=1)
        sub_probs = F.softmax(sub_logits, dim=1)

        super_ce = F.cross_entropy(super_logits, super_targets, reduction="sum").item()
        sub_ce = F.cross_entropy(sub_logits, sub_targets, reduction="sum").item()

        self.acc.super_ce_sum += super_ce
        self.acc.sub_ce_sum += sub_ce
        self.acc.ce_count += batch_size

        if is_unseen is not None:
            seen_mask = ~is_unseen
            unseen_mask = is_unseen

            if seen_mask.any():
                self.acc.seen_super_correct += (super_preds[seen_mask] == super_targets[seen_mask]).sum().item()
                self.acc.seen_super_total += seen_mask.sum().item()
                self.acc.seen_sub_correct += (sub_preds[seen_mask] == sub_targets[seen_mask]).sum().item()
                self.acc.seen_sub_total += seen_mask.sum().item()

            if unseen_mask.any():
                self.acc.unseen_super_correct += (super_preds[unseen_mask] == super_targets[unseen_mask]).sum().item()
                self.acc.unseen_super_total += unseen_mask.sum().item()
                self.acc.unseen_sub_correct += (sub_preds[unseen_mask] == sub_targets[unseen_mask]).sum().item()
                self.acc.unseen_sub_total += unseen_mask.sum().item()

        novel_super_mask = super_targets == self.novel_super_idx
        if novel_super_mask.any():
            self.acc.novel_super_tp += ((super_preds == self.novel_super_idx) & novel_super_mask).sum().item()
            self.acc.novel_super_fn += ((super_preds != self.novel_super_idx) & novel_super_mask).sum().item()
        non_novel_super_mask = super_targets != self.novel_super_idx
        if non_novel_super_mask.any():
            self.acc.novel_super_fp += ((super_preds == self.novel_super_idx) & non_novel_super_mask).sum().item()

        novel_sub_mask = sub_targets == self.novel_sub_idx
        if novel_sub_mask.any():
            self.acc.novel_sub_tp += ((sub_preds == self.novel_sub_idx) & novel_sub_mask).sum().item()
            self.acc.novel_sub_fn += ((sub_preds != self.novel_sub_idx) & novel_sub_mask).sum().item()
        non_novel_sub_mask = sub_targets != self.novel_sub_idx
        if non_novel_sub_mask.any():
            self.acc.novel_sub_fp += ((sub_preds == self.novel_sub_idx) & non_novel_sub_mask).sum().item()

    def compute(self) -> EpochMetrics:
        super_acc = self.acc.super_correct / max(self.acc.super_total, 1) * 100
        sub_acc = self.acc.sub_correct / max(self.acc.sub_total, 1) * 100

        super_ce = self.acc.super_ce_sum / max(self.acc.ce_count, 1)
        sub_ce = self.acc.sub_ce_sum / max(self.acc.ce_count, 1)

        seen_super_acc = self.acc.seen_super_correct / max(self.acc.seen_super_total, 1) * 100
        unseen_super_acc = self.acc.unseen_super_correct / max(self.acc.unseen_super_total, 1) * 100
        seen_sub_acc = self.acc.seen_sub_correct / max(self.acc.seen_sub_total, 1) * 100
        unseen_sub_acc = self.acc.unseen_sub_correct / max(self.acc.unseen_sub_total, 1) * 100

        novel_super_f1 = self._compute_f1(
            self.acc.novel_super_tp,
            self.acc.novel_super_fp,
            self.acc.novel_super_fn,
        )
        novel_sub_f1 = self._compute_f1(
            self.acc.novel_sub_tp,
            self.acc.novel_sub_fp,
            self.acc.novel_sub_fn,
        )

        return EpochMetrics(
            super_acc=super_acc,
            sub_acc=sub_acc,
            super_ce=super_ce,
            sub_ce=sub_ce,
            seen_super_acc=seen_super_acc,
            unseen_super_acc=unseen_super_acc,
            seen_sub_acc=seen_sub_acc,
            unseen_sub_acc=unseen_sub_acc,
            novel_super_f1=novel_super_f1,
            novel_sub_f1=novel_sub_f1,
        )

    def _compute_f1(self, tp: int, fp: int, fn: int) -> float:
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
