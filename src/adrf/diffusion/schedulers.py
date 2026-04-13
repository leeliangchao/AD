"""Scheduler adapters built on top of diffusers."""

from __future__ import annotations

from typing import Any

from diffusers import DDPMScheduler
import torch


class DiffusersSchedulerAdapter:
    """Small wrapper around a diffusers DDPM scheduler."""

    def __init__(self, scheduler: DDPMScheduler) -> None:
        self.scheduler = scheduler

    def add_noise(self, sample: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to a clean sample using the wrapped scheduler."""

        return self.scheduler.add_noise(sample, noise, timesteps)

    def step(self, model_output: torch.Tensor, timestep: int | torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        """Take one scheduler step and return the previous sample estimate."""

        timestep_value = int(timestep.item()) if isinstance(timestep, torch.Tensor) else int(timestep)
        return self.scheduler.step(model_output, timestep_value, sample).prev_sample


def make_scheduler(name: str, **kwargs: Any) -> DiffusersSchedulerAdapter:
    """Construct a supported diffusers scheduler adapter."""

    normalized = name.lower()
    if normalized != "ddpm":
        raise ValueError(f"Unsupported diffusers scheduler: {name}")
    scheduler = DDPMScheduler(
        num_train_timesteps=int(kwargs.pop("num_train_timesteps", 1000)),
        beta_schedule=str(kwargs.pop("beta_schedule", "linear")),
        clip_sample=bool(kwargs.pop("clip_sample", False)),
        **kwargs,
    )
    return DiffusersSchedulerAdapter(scheduler)

