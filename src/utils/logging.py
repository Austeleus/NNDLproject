from pathlib import Path
from typing import Any, Dict, Optional

import wandb

from ..config import Config


class WandbLogger:
    def __init__(
        self,
        config: Config,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.config = config

        if self.enabled:
            wandb.init(
                project=config.logging.project_name,
                name=config.logging.experiment_name,
                config=self._config_to_dict(config),
            )

    def _config_to_dict(self, config: Config) -> Dict[str, Any]:
        from dataclasses import asdict

        def convert(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        result = {}
        for section in ["data", "augmentation", "model", "training", "loss", "inference"]:
            section_dict = asdict(getattr(config, section))
            for k, v in section_dict.items():
                result[f"{section}/{k}"] = convert(v)

        return result

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        if self.enabled:
            wandb.log(metrics, step=step)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
    ) -> None:
        if not self.enabled:
            return

        log_dict = {"epoch": epoch}

        for k, v in train_metrics.items():
            log_dict[f"train/{k}"] = v

        if val_metrics:
            for k, v in val_metrics.items():
                log_dict[f"val/{k}"] = v

        if lr is not None:
            log_dict["lr"] = lr

        wandb.log(log_dict)

    def log_batch(
        self,
        step: int,
        loss: float,
        lr: float,
    ) -> None:
        if self.enabled:
            wandb.log({
                "batch/loss": loss,
                "batch/lr": lr,
            }, step=step)

    def save_model(self, model_path: Path) -> None:
        if self.enabled:
            wandb.save(str(model_path))

    def finish(self) -> None:
        if self.enabled:
            wandb.finish()


def setup_logging(config: Config, enabled: bool = True) -> WandbLogger:
    return WandbLogger(config, enabled=enabled)
