"""
src/autoencoder/model.py
========================
Raman spectra autoencoders used as frozen feature extractors for the
downstream PI-LSTM / Neural ODE soft-sensor benchmark.

Three classes are exposed:

* ``CDAE_Raman``    — Convolutional Denoising Autoencoder
* ``CVAE_Raman``    — Convolutional Variational Autoencoder  (β = 1)
* ``BetaVAE_Raman`` — β-VAE with modular Encoder1D / Decoder1D submodules
                       and LeakyReLU activations; designed for disentangled
                       latent representations of the biochemical state

CDAE / CVAE share the same 1D convolutional encoder / decoder backbone:

    Encoder  (input length L, e.g. 2001)
        Conv1d(  1 ->  32, k=7) + BatchNorm + ReLU + MaxPool(2)
        Conv1d( 32 ->  64, k=5) + BatchNorm + ReLU + MaxPool(2)
        Conv1d( 64 -> 128, k=3) + BatchNorm + ReLU + MaxPool(2)
        Flatten ( = 128 * L//8 )
        Linear -> latent_dim                         (CDAE)
        Linear -> mu, log_var (heads)                (CVAE)

    Decoder
        Linear -> 128 * L//8 + Reshape
        ConvTranspose1d(128 ->  64, k=3, stride=2) + BatchNorm + ReLU
        ConvTranspose1d( 64 ->  32, k=5, stride=2) + BatchNorm + ReLU
        ConvTranspose1d( 32 ->   1, k=7, stride=2)
        Crop / pad to input length L

BetaVAE_Raman uses a **deeper 4-block backbone** with LeakyReLU:

    Encoder1D  (input length L, e.g. 2001)
        Conv1d(  1 ->  32, k=7) + BN + LeakyReLU + MaxPool(2)   -> L//2
        Conv1d( 32 ->  64, k=5) + BN + LeakyReLU + MaxPool(2)   -> L//4
        Conv1d( 64 -> 128, k=3) + BN + LeakyReLU + MaxPool(2)   -> L//8
        Conv1d(128 -> 256, k=3) + BN + LeakyReLU + MaxPool(2)   -> L//16
        Flatten ( = 256 * L//16 )     # 256 * 125 = 32_000 for L=2001

    Decoder1D
        Linear(latent_dim -> 256 * L//16) + Reshape
        ConvTranspose1d(256 -> 128, k=3, stride=2) + BN + LeakyReLU
        ConvTranspose1d(128 ->  64, k=3, stride=2) + BN + LeakyReLU
        ConvTranspose1d( 64 ->  32, k=5, stride=2) + BN + LeakyReLU
        ConvTranspose1d( 32 ->   1, k=7, stride=2)
        Crop / pad to input length L

The BetaVAE_Raman.encode() / .decode() / .forward() signatures are
intentionally **identical** to CVAE_Raman so that CVAERamanEncoderV2
can load it without modification (only the model_state_dict changes).

Checkpoint state-dict keys for BetaVAE_Raman
---------------------------------------------
    encoder.blocks.{0..3}.{0..2}   (Conv1d, BatchNorm1d, LeakyReLU)
    fc_mu / fc_log_var
    decoder.fc
    decoder.blocks.{0..3}.{0..2}   (ConvTranspose1d, BatchNorm1d, LeakyReLU)
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# CDAE
# ─────────────────────────────────────────────────────────────────────────────
class CDAE_Raman(nn.Module):
    """
    Convolutional Denoising Autoencoder for 1-D Raman spectra.

    Parameters
    ----------
    input_length : int
        Number of wavenumber channels in the input spectrum (e.g. 2001).
    latent_dim   : int
        Bottleneck dimensionality (default 64).
    noise_std    : float
        Gaussian noise std added to the input during ``forward`` (training).
        Set to 0.0 for deterministic inference (default in encoders).
    """

    def __init__(
        self,
        input_length: int = 2001,
        latent_dim:   int = 64,
        noise_std:    float = 0.1,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim   = int(latent_dim)
        self.noise_std    = float(noise_std)

        # After 3 MaxPool(2) layers, the spatial dimension is input_length // 8
        self._pooled_len  = self.input_length // 8
        self._flatten_dim = 128 * self._pooled_len

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc_block1 = nn.Sequential(
            nn.Conv1d(  1,  32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv1d( 32,  64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block3 = nn.Sequential(
            nn.Conv1d( 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.encoder_fc = nn.Linear(self._flatten_dim, self.latent_dim)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder_fc = nn.Linear(self.latent_dim, self._flatten_dim)
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose1d(128,  64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose1d( 64,  32, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose1d( 32,   1, kernel_size=7,
                               stride=2, padding=3, output_padding=1),
        )

    # ── Public API ──────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``(B, L)`` or ``(B, 1, L)`` spectra to a ``(B, latent_dim)`` latent."""
        if x.dim() == 2:
            x = x.unsqueeze(1)               # (B, 1, L)
        z = self.enc_block1(x)
        z = self.enc_block2(z)
        z = self.enc_block3(z)
        z = z.view(z.size(0), -1)            # (B, 128 * L//8)
        return self.encoder_fc(z)            # (B, latent_dim)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map a ``(B, latent_dim)`` latent back to a ``(B, L)`` spectrum."""
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 128, self._pooled_len)
        h = self.dec_block1(h)
        h = self.dec_block2(h)
        h = self.dec_block3(h)               # (B, 1, ~L)
        h = h.squeeze(1)
        # Crop / pad to the exact input length (ConvTranspose may overshoot/undershoot)
        if h.size(1) > self.input_length:
            h = h[:, : self.input_length]
        elif h.size(1) < self.input_length:
            pad = self.input_length - h.size(1)
            h = F.pad(h, (0, pad))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add denoising noise (training mode) and reconstruct the spectrum."""
        if self.training and self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return self.decode(self.encode(x))


# ─────────────────────────────────────────────────────────────────────────────
# CVAE
# ─────────────────────────────────────────────────────────────────────────────
class CVAE_Raman(nn.Module):
    """
    Convolutional Variational Autoencoder for 1-D Raman spectra.

    Shares the same conv backbone as :class:`CDAE_Raman` but exposes two
    bottleneck heads (``fc_mu``, ``fc_log_var``) and a reparameterised
    sampling step.  At inference the posterior mean ``mu`` is used as the
    deterministic latent representation.

    Parameters
    ----------
    input_length : int    -- number of wavenumber channels (default 2001)
    latent_dim   : int    -- latent dimensionality (default 64)
    """

    def __init__(
        self,
        input_length: int = 2001,
        latent_dim:   int = 64,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim   = int(latent_dim)

        self._pooled_len  = self.input_length // 8
        self._flatten_dim = 128 * self._pooled_len

        # ── Encoder ─────────────────────────────────────────────────────────
        self.enc_block1 = nn.Sequential(
            nn.Conv1d(  1,  32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv1d( 32,  64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.enc_block3 = nn.Sequential(
            nn.Conv1d( 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        self.fc_mu      = nn.Linear(self._flatten_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self._flatten_dim, self.latent_dim)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.decoder_fc = nn.Linear(self.latent_dim, self._flatten_dim)
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose1d(128,  64, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose1d( 64,  32, kernel_size=5,
                               stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose1d( 32,   1, kernel_size=7,
                               stride=2, padding=3, output_padding=1),
        )

    # ── Public API ──────────────────────────────────────────────────────────
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the posterior parameters ``(mu, log_var)``.

        Accepts inputs of shape ``(B, L)`` or ``(B, 1, L)``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.enc_block1(x)
        z = self.enc_block2(z)
        z = self.enc_block3(z)
        z = z.view(z.size(0), -1)

        mu = self.fc_mu(z)
        log_var = self.fc_log_var(z)

        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        
        return mu, log_var

    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        h = h.view(z.size(0), 128, self._pooled_len)
        h = self.dec_block1(h)
        h = self.dec_block2(h)
        h = self.dec_block3(h)
        h = h.squeeze(1)
        if h.size(1) > self.input_length:
            h = h[:, : self.input_length]
        elif h.size(1) < self.input_length:
            pad = self.input_length - h.size(1)
            h = F.pad(h, (0, pad))
        return h

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(reconstruction, mu, log_var)``."""
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var) if self.training else mu
        return self.decode(z), mu, log_var


# ─────────────────────────────────────────────────────────────────────────────
# β-VAE  (modular design)
# ─────────────────────────────────────────────────────────────────────────────

class Encoder1D(nn.Module):
    """
    Modular 1D convolutional encoder for Raman spectra.

    Architecture (4 conv blocks with LeakyReLU):
        Conv1d(  1→  32, k=7, p=3) + BN + LeakyReLU + MaxPool(2)
        Conv1d( 32→  64, k=5, p=2) + BN + LeakyReLU + MaxPool(2)
        Conv1d( 64→ 128, k=3, p=1) + BN + LeakyReLU + MaxPool(2)
        Conv1d(128→ 256, k=3, p=1) + BN + LeakyReLU + MaxPool(2)
        Flatten  →  (B, 256 * input_length//16)

    The output is a flat feature vector; variational heads (fc_mu, fc_log_var)
    are kept in the parent BetaVAE_Raman so this module can be reused for
    deterministic tasks.

    Parameters
    ----------
    input_length   : number of wavenumber channels (default 2001)
    leaky_slope    : negative slope for LeakyReLU (default 0.2)
    """

    # Channel progression at each block boundary
    _CHANNELS: List[int] = [1, 32, 64, 128, 256]
    _KERNELS:  List[int] = [7,  5,  3,   3]
    _PADDING:  List[int] = [3,  2,  1,   1]

    def __init__(
        self,
        input_length: int   = 2001,
        leaky_slope:  float = 0.2,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.leaky_slope  = float(leaky_slope)

        # After 4 MaxPool(2) ops: spatial dim = input_length // 16
        self._pooled_len  = self.input_length // 16
        self.flatten_dim  = self._CHANNELS[-1] * self._pooled_len

        # ── Build conv blocks dynamically from channel/kernel tables ──────────
        self.blocks = nn.ModuleList()
        for i in range(len(self._KERNELS)):
            in_ch  = self._CHANNELS[i]
            out_ch = self._CHANNELS[i + 1]
            k      = self._KERNELS[i]
            p      = self._PADDING[i]
            block  = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                nn.MaxPool1d(2),
            )
            self.blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, L) or (B, L)  — raw or preprocessed Raman spectra

        Returns
        -------
        h : (B, flatten_dim)  — flat feature vector
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # (B, L) → (B, 1, L)
        for block in self.blocks:
            x = block(x)
        return x.view(x.size(0), -1)   # (B, 256 * L//16)


class Decoder1D(nn.Module):
    """
    Modular 1D convolutional decoder for Raman spectra.

    Mirrors Encoder1D exactly: 4 transposed-conv blocks with LeakyReLU
    followed by a crop/pad to restore the exact input length.

    Parameters
    ----------
    input_length : target output length (must match Encoder1D.input_length)
    latent_dim   : size of the input latent vector
    leaky_slope  : negative slope for LeakyReLU (default 0.2)
    """

    _CHANNELS: List[int] = [256, 128, 64, 32, 1]
    _KERNELS:  List[int] = [3,    3,   5,  7]
    _PADDING:  List[int] = [1,    1,   2,  3]

    def __init__(
        self,
        input_length: int   = 2001,
        latent_dim:   int   = 64,
        leaky_slope:  float = 0.2,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim   = int(latent_dim)
        self.leaky_slope  = float(leaky_slope)

        self._pooled_len = self.input_length // 16
        self._fc_dim     = self._CHANNELS[0] * self._pooled_len

        # Project latent vector back to spatial feature map
        self.fc = nn.Linear(self.latent_dim, self._fc_dim)

        # ── Build transposed-conv blocks dynamically ──────────────────────────
        self.blocks = nn.ModuleList()
        for i in range(len(self._KERNELS)):
            in_ch  = self._CHANNELS[i]
            out_ch = self._CHANNELS[i + 1]
            k      = self._KERNELS[i]
            p      = self._PADDING[i]
            is_last = (i == len(self._KERNELS) - 1)

            if is_last:
                # Final block: no BN, no activation — raw output logits
                block = nn.Sequential(
                    nn.ConvTranspose1d(
                        in_ch, out_ch,
                        kernel_size=k, stride=2, padding=p, output_padding=1,
                        bias=True,
                    ),
                )
            else:
                block = nn.Sequential(
                    nn.ConvTranspose1d(
                        in_ch, out_ch,
                        kernel_size=k, stride=2, padding=p, output_padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm1d(out_ch),
                    nn.LeakyReLU(negative_slope=leaky_slope, inplace=True),
                )
            self.blocks.append(block)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        x_hat : (B, input_length)  — reconstructed spectrum
        """
        h = self.fc(z)                                       # (B, fc_dim)
        h = h.view(h.size(0), self._CHANNELS[0], self._pooled_len)  # (B, 256, L//16)
        for block in self.blocks:
            h = block(h)
        h = h.squeeze(1)                                     # (B, ~L)

        # Exact length restoration (ConvTranspose can overshoot by 1–2 samples)
        if h.size(1) > self.input_length:
            h = h[:, : self.input_length]
        elif h.size(1) < self.input_length:
            h = F.pad(h, (0, self.input_length - h.size(1)))
        return h                                             # (B, L)


class BetaVAE_Raman(nn.Module):
    """
    β-VAE for 1D Raman spectra.

    Composes ``Encoder1D`` and ``Decoder1D`` with variational heads
    (``fc_mu``, ``fc_log_var``) and the reparameterisation trick.

    The ``encode()`` / ``decode()`` / ``forward()`` signatures are
    **identical** to ``CVAE_Raman`` so that ``CVAERamanEncoderV2`` can
    load this model without modification — only the checkpoint path changes.

    Parameters
    ----------
    input_length : int   — wavenumber channels (default 2001)
    latent_dim   : int   — bottleneck dimensionality (default 64)
    leaky_slope  : float — LeakyReLU negative slope (default 0.2)

    Notes
    -----
    β is **not** stored on the model; it is passed to the loss function at
    training time (see ``beta_vae_loss`` in ``scripts/train_autoencoder.py``
    and ``BetaSchedule`` in ``src/autoencoder/beta_schedule.py``).
    This keeps the architecture independent of the training schedule so the
    same checkpoint can be evaluated under different β values.
    """

    def __init__(
        self,
        input_length: int   = 2001,
        latent_dim:   int   = 64,
        leaky_slope:  float = 0.2,
    ) -> None:
        super().__init__()
        self.input_length = int(input_length)
        self.latent_dim   = int(latent_dim)

        # ── Submodules ────────────────────────────────────────────────────────
        self.encoder   = Encoder1D(input_length=input_length, leaky_slope=leaky_slope)
        self.fc_mu     = nn.Linear(self.encoder.flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(self.encoder.flatten_dim, latent_dim)
        self.decoder   = Decoder1D(
            input_length=input_length,
            latent_dim=latent_dim,
            leaky_slope=leaky_slope,
        )

    # ── Variational bottleneck ────────────────────────────────────────────────
    @staticmethod
    def reparameterise(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Sample z ~ N(μ, σ²) using the reparameterisation trick.
        Gradients flow through μ and log_var, not through the noise ε.

            z = μ + ε · exp(0.5 · log_var),   ε ~ N(0, I)
        """
        std = torch.exp(0.5 * log_var)          # σ  = exp(log σ)
        eps = torch.randn_like(std)             # ε ~ N(0, I)
        return mu + eps * std

    # ── Public API (identical signatures to CVAE_Raman) ──────────────────────
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a batch of spectra to posterior parameters.

        Parameters
        ----------
        x : (B, L) or (B, 1, L)

        Returns
        -------
        mu      : (B, latent_dim)  — posterior mean
        log_var : (B, latent_dim)  — posterior log-variance, clamped to [-10, 10]
                                     to prevent exp() overflow during early training
        """
        h = self.encoder(x)               # (B, flatten_dim)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        # Clamp log_var for numerical stability: exp(10)≈22026, exp(-10)≈4.5e-5
        # Without this, beta=0 warm-up allows log_var to explode → 0*inf = NaN
        log_var = torch.clamp(log_var, min=-10.0, max=10.0)
        return mu, log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode a latent vector to a reconstructed spectrum.

        Parameters
        ----------
        z : (B, latent_dim)

        Returns
        -------
        x_hat : (B, input_length)
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.

        Training mode  : z is sampled via reparameterisation
        Evaluation mode: z = μ (deterministic posterior mean, no noise)

        Returns
        -------
        recon   : (B, input_length)  — reconstructed spectrum
        mu      : (B, latent_dim)    — posterior mean
        log_var : (B, latent_dim)    — posterior log-variance
        """
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var) if self.training else mu
        return self.decode(z), mu, log_var
