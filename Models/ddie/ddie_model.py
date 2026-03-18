"""
DDI-E: Conditional Diffusion Model for Probabilistic Reconstruction
of Electrical Time Series under Random Missing Data
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial

from Models.interpretable_diffusion.transformer import Transformer
from Models.interpretable_diffusion.model_utils import default, identity, extract


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as in Improved DDPM."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDI_E(nn.Module):
    """
    DDI-E diffusion model with:
    - Mask as condition (2nd channel)
    - Cosine beta schedule
    - Frequency domain L1 loss
    """

    def __init__(
        self,
        seq_length=128,
        feature_size=1,
        n_layer_enc=4,
        n_layer_dec=2,
        d_model=128,
        timesteps=1000,
        sampling_timesteps=200,
        loss_type='l1',
        beta_schedule='cosine',
        n_heads=8,
        mlp_hidden_times=4,
        attn_pd=0.0,
        resid_pd=0.0,
        kernel_size=1,
        padding_size=0,
        use_mask_condition=True,
        freq_loss_weight=0.1,
        use_trend_cycle=False,
        **kwargs
    ):
        super().__init__()
        self.use_mask_condition = use_mask_condition
        self.freq_loss_weight = freq_loss_weight
        self.feature_size = feature_size
        self.seq_length = seq_length

        # Input channels: 2 if mask conditioning, else 1
        in_feat = 2 if use_mask_condition else 1
        self.model = Transformer(
            n_feat=in_feat,
            n_channel=seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length,
            conv_params=[kernel_size, padding_size],
        )
        # Project output back to single channel (data only)
        self.out_proj = nn.Conv1d(in_feat, 1, 1)

        betas = cosine_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.fast_sampling = self.sampling_timesteps < timesteps

        register_buffer = lambda n, v: self.register_buffer(n, v.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

        register_buffer('loss_weight', torch.sqrt(alphas) * torch.sqrt(1.0 - alphas_cumprod) / betas / 100)

    def _prepare_input(self, x, mask=None):
        """Concatenate mask as 2nd channel if conditioning."""
        if self.use_mask_condition and mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            return torch.cat([x, mask], dim=-1)
        if self.use_mask_condition and mask is None:
            ones = torch.ones_like(x)
            return torch.cat([x, ones], dim=-1)
        return x

    def output(self, x, t, mask=None, padding_masks=None):
        """Model output: predict x_start (B, L, 1)."""
        inp = self._prepare_input(x, mask)
        trend, season = self.model(inp, t, padding_masks=padding_masks)
        out = trend + season
        out = self.out_proj(out.transpose(1, 2)).transpose(1, 2)
        return out

    def predict_noise_from_start(self, x_t, t, x0):
        return (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        return F.l1_loss if self.loss_type == 'l1' else F.mse_loss

    def _train_loss(self, x_start, t, mask=None, noise=None, padding_masks=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Replace observed positions in x_t with noised x_start (for conditioning)
        if mask is not None:
            mask_exp = mask if mask.dim() == 3 else mask.unsqueeze(-1)
            x_t_obs = self.q_sample(x_start, t, noise)
            x_t = torch.where(mask_exp > 0.5, x_t_obs, x_t)

        model_out = self.output(x_t, t, mask=mask, padding_masks=padding_masks)
        train_loss = self.loss_fn(model_out, x_start, reduction='none')

        fourier_loss = torch.tensor(0.0, device=x_start.device)
        if self.freq_loss_weight > 0:
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(x_start.transpose(1, 2), norm='forward')
            amp1 = torch.abs(fft1)
            amp2 = torch.abs(fft2)
            fourier_loss = F.l1_loss(amp1, amp2, reduction='none')
            train_loss = train_loss + self.freq_loss_weight * fourier_loss.transpose(1, 2)

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        train_loss = train_loss * extract(self.loss_weight, t, train_loss.shape)
        return train_loss.mean()

    def forward(self, x, mask=None, **kwargs):
        b, device = x.shape[0], x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self._train_loss(x_start=x, t=t, mask=mask, **kwargs)

    def model_predictions(self, x, t, mask=None, clip_x_start=True, padding_masks=None):
        maybe_clip = partial(torch.clamp, min=-1.0, max=1.0) if clip_x_start else identity
        x_start = self.output(x, t, mask=mask, padding_masks=padding_masks)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(self, x, t, mask=None, clip_denoised=True, padding_masks=None):
        _, x_start = self.model_predictions(x, t, mask=mask, clip_x_start=clip_denoised, padding_masks=padding_masks)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    def p_sample(self, x, t, mask=None, clip_denoised=True, model_kwargs=None):
        b, *_, device = *x.shape, self.betas.device
        batched_times = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x, batched_times, mask=mask, clip_denoised=clip_denoised, **model_kwargs or {}
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def sample_infill(self, shape, target, mask, clip_denoised=True):
        """Conditional imputation: replace observed with target, denoise missing."""
        batch, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        partial_mask = (mask > 0.5)
        if mask.dim() == 2:
            partial_mask = partial_mask.unsqueeze(-1)

        for t in reversed(range(0, self.num_timesteps)):
            img, _ = self.p_sample(img, t, mask=mask, clip_denoised=clip_denoised)
            target_t = self.q_sample(target, t=torch.full((batch,), t, device=device, dtype=torch.long))
            img = torch.where(partial_mask, target_t, img)

        img = torch.where(partial_mask, target, img)
        return img

    @torch.no_grad()
    def fast_sample_infill(self, shape, target, mask, clip_denoised=True):
        """Faster imputation using reduced sampling steps."""
        batch, device = shape[0], self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        partial_mask = (mask > 0.5)
        if mask.dim() == 2:
            partial_mask = partial_mask.unsqueeze(-1)

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start = self.model_predictions(img, time_cond, mask=mask, clip_x_start=clip_denoised)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)
            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

            target_t = self.q_sample(target, t=time_cond)
            img = torch.where(partial_mask, target_t, img)

        img = torch.where(partial_mask, target, img)
        return img
