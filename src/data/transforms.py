from typing import Tuple
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import autoaugment

from ..config import AugmentationConfig


def mixup_data(
    x: torch.Tensor,
    y_super: torch.Tensor,
    y_sub: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float, torch.Tensor]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_super_a, y_super_b = y_super, y_super[index]
    y_sub_a, y_sub_b = y_sub, y_sub[index]

    return mixed_x, y_super_a, y_super_b, y_sub_a, y_sub_b, lam, index


def mixup_criterion(
    criterion,
    pred_super,
    pred_sub,
    y_super_a,
    y_super_b,
    y_sub_a,
    y_sub_b,
    lam: float,
):
    loss_super = lam * criterion(pred_super, y_super_a) + (1 - lam) * criterion(pred_super, y_super_b)
    loss_sub = lam * criterion(pred_sub, y_sub_a) + (1 - lam) * criterion(pred_sub, y_sub_b)
    return loss_super, loss_sub


def get_train_transforms(config: AugmentationConfig, image_size: int = 224) -> T.Compose:
    transforms_list = [
        T.RandomResizedCrop(
            image_size,
            scale=config.random_crop_scale,
            ratio=config.random_crop_ratio,
            interpolation=T.InterpolationMode.BILINEAR,
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue,
        ),
        autoaugment.RandAugment(
            num_ops=config.randaugment_n,
            magnitude=config.randaugment_m,
        ),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ]

    if hasattr(config, 'random_erasing_prob') and config.random_erasing_prob > 0:
        transforms_list.append(
            T.RandomErasing(
                p=config.random_erasing_prob,
                scale=config.random_erasing_scale,
                ratio=(0.3, 3.3),
            )
        )

    return T.Compose(transforms_list)


def get_val_transforms(config: AugmentationConfig, image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def get_test_transforms(config: AugmentationConfig, image_size: int = 64) -> T.Compose:
    return get_val_transforms(config, image_size)


def denormalize(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    if tensor.device != mean_t.device:
        mean_t = mean_t.to(tensor.device)
        std_t = std_t.to(tensor.device)

    return tensor * std_t + mean_t
