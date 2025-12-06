import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class MahalanobisNoveltyDetector(nn.Module):
    """
    Mahalanobis distance-based novelty detection.

    Computes class centroids and covariance from training features,
    then uses distance to nearest centroid for novel detection.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        regularization: float = 1e-4,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.regularization = regularization

        # Class centroids (computed from training data)
        self.register_buffer(
            "centroids",
            torch.zeros(num_classes, feature_dim)
        )

        # Shared precision matrix (inverse covariance)
        self.register_buffer(
            "precision",
            torch.eye(feature_dim)
        )

        # Track if fitted
        self.register_buffer(
            "is_fitted",
            torch.tensor(False)
        )

        # Learnable threshold parameters
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """
        Compute centroids and precision matrix from training features.

        Args:
            features: (N, feature_dim) tensor of training features
            labels: (N,) tensor of class labels
        """
        device = features.device

        # Compute per-class centroids
        centroids = torch.zeros(self.num_classes, self.feature_dim, device=device)
        counts = torch.zeros(self.num_classes, device=device)

        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centroids[c] = features[mask].mean(dim=0)
                counts[c] = mask.sum()

        self.centroids.copy_(centroids)

        # Compute shared covariance matrix
        centered_features = []
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                centered = features[mask] - centroids[c]
                centered_features.append(centered)

        all_centered = torch.cat(centered_features, dim=0)
        covariance = (all_centered.T @ all_centered) / (all_centered.size(0) - 1)

        # Add regularization and compute precision (inverse)
        covariance += self.regularization * torch.eye(self.feature_dim, device=device)
        precision = torch.linalg.inv(covariance)

        self.precision.copy_(precision)
        self.is_fitted.fill_(True)

    def mahalanobis_distance(
        self,
        features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Mahalanobis distance to each class centroid.

        Returns:
            distances: (batch, num_classes) distances to each centroid
            min_distances: (batch,) minimum distance (to nearest class)
        """
        batch_size = features.size(0)

        # (batch, num_classes, feature_dim)
        diff = features.unsqueeze(1) - self.centroids.unsqueeze(0)

        # Mahalanobis: sqrt((x - mu)^T @ Sigma^-1 @ (x - mu))
        # (batch, num_classes, feature_dim) @ (feature_dim, feature_dim)
        left = torch.einsum('bcd,dd->bcd', diff, self.precision)
        # (batch, num_classes)
        distances = torch.sqrt((left * diff).sum(dim=-1) + 1e-8)

        min_distances = distances.min(dim=1).values

        return distances, min_distances

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute novel logit based on Mahalanobis distance.

        High distance = high novel probability.
        """
        _, min_distances = self.mahalanobis_distance(features)

        # Convert distance to logit
        # Higher distance → higher novel logit
        novel_logit = self.alpha + self.beta * min_distances

        return novel_logit


class HybridNoveltyHead(nn.Module):
    """
    Combines energy-based virtual logit with Mahalanobis distance.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        energy_weight: float = 0.5,
        mahal_weight: float = 0.5,
    ):
        super().__init__()
        self.mahalanobis = MahalanobisNoveltyDetector(feature_dim, num_classes)
        self.energy_weight = energy_weight
        self.mahal_weight = mahal_weight

        # Energy-based parameters
        self.alpha_energy = nn.Parameter(torch.tensor(0.0))
        self.beta_energy = nn.Parameter(torch.tensor(1.0))

    def compute_energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        return -temperature * torch.logsumexp(logits / temperature, dim=1)

    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute hybrid novel logit from both energy and Mahalanobis.
        """
        # Energy-based
        energy = self.compute_energy(logits)
        energy_novel = self.alpha_energy + self.beta_energy * energy

        # Mahalanobis-based (if fitted)
        if self.mahalanobis.is_fitted:
            mahal_novel = self.mahalanobis(features)
            novel_logit = (
                self.energy_weight * energy_novel +
                self.mahal_weight * mahal_novel
            )
        else:
            novel_logit = energy_novel

        return novel_logit
