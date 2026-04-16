"""High-level diffusers adapter for noise prediction in AD normality models."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import nn

from adrf.diffusion.models import DiffusersUNetAdapter, make_unet_model
from adrf.diffusion.schedulers import DiffusersSchedulerAdapter, make_scheduler
from adrf.normality.state import install_normality_runtime_state, make_default_normality_runtime_state


class DiffusersNoisePredictorAdapter(nn.Module):
    """Train/infer helper around a diffusers UNet and scheduler."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int | None,
        hidden_channels: int,
        learning_rate: float,
        noise_level: float,
        sample_size: int,
        num_train_timesteps: int,
        num_classes: int | None = None,
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.noise_level = noise_level
        self.scheduler: DiffusersSchedulerAdapter = make_scheduler("ddpm", num_train_timesteps=num_train_timesteps)
        self.model: DiffusersUNetAdapter = make_unet_model(
            {
                "sample_size": sample_size,
                "in_channels": input_channels,
                "out_channels": int(output_channels if output_channels is not None else input_channels),
                "layers_per_block": 1,
                "block_out_channels": (hidden_channels, hidden_channels * 2),
                "norm_num_groups": 4,
                "num_class_embeds": num_classes,
            }
        )
        install_normality_runtime_state(self, make_default_normality_runtime_state())

    def forward_train_step(
        self,
        clean_batch: torch.Tensor,
        *,
        conditioning: torch.Tensor | None = None,
        class_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one forward noise-prediction step for training."""

        noisy_batch, target_noise, timesteps = self._sample_noisy_inputs(clean_batch)
        model_inputs = torch.cat([noisy_batch, conditioning], dim=1) if conditioning is not None else noisy_batch
        predicted_noise = self.model(model_inputs, timesteps, class_ids=class_ids)
        loss = functional.mse_loss(predicted_noise, target_noise)
        return loss, target_noise

    def forward_infer_step(
        self,
        clean_image: torch.Tensor,
        *,
        target_noise: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
        class_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward inference step and return predicted/target noise."""

        max_timestep = self.scheduler.scheduler.config.num_train_timesteps - 1
        timesteps = torch.full(
            (clean_image.shape[0],),
            fill_value=max_timestep,
            device=clean_image.device,
            dtype=torch.long,
        )
        noisy_image, target_noise, timesteps = self._sample_noisy_inputs(
            clean_image,
            timesteps=timesteps,
            target_noise=target_noise,
        )
        model_inputs = torch.cat([noisy_image, conditioning], dim=1) if conditioning is not None else noisy_image
        predicted_noise = self.model(model_inputs, timesteps, class_ids=class_ids)
        return predicted_noise, target_noise, noisy_image, timesteps

    def reconstruct_clean(
        self,
        noisy_batch: torch.Tensor,
        predicted_noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate x0 from xt and predicted noise under the wrapped scheduler."""

        alphas_cumprod = self.scheduler.scheduler.alphas_cumprod.to(device=noisy_batch.device, dtype=noisy_batch.dtype)
        alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sigma_t = torch.sqrt((1.0 - alpha_t).clamp_min(0.0))
        return (noisy_batch - sigma_t * self.noise_level * predicted_noise) / torch.sqrt(alpha_t.clamp_min(1e-12))

    def _sample_noisy_inputs(
        self,
        clean_batch: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        target_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample Gaussian noise and DDPM timesteps for the input batch."""

        batch_size = clean_batch.shape[0]
        if target_noise is None:
            target_noise = torch.randn_like(clean_batch)
        raw_timesteps = timesteps
        if raw_timesteps is None:
            raw_timesteps = torch.randint(
                low=0,
                high=self.scheduler.scheduler.config.num_train_timesteps,
                size=(batch_size,),
                device=clean_batch.device,
                dtype=torch.long,
            )
        noisy_batch = self.scheduler.add_noise(clean_batch, self.noise_level * target_noise, raw_timesteps)
        return noisy_batch, target_noise, raw_timesteps
