"""
src/autoencoder/beta_schedule.py
=================================
β annealing schedules for β-VAE training.

Two schedules are provided:

``LinearBetaSchedule``
    β stays at 0 for an optional delay period, then rises linearly from 0
    to ``max_beta`` over ``anneal_epochs`` epochs, and remains at ``max_beta``
    thereafter.  A warm-up delay (``delay_epochs``) lets the decoder learn a
    rough reconstruction before the KL term is introduced.

``CyclicalBetaSchedule``
    Implements the cyclical annealing schedule of Fu et al. (2019)
    "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL
    Vanishing".  β ramps from 0 → max_beta in the first ``ratio`` fraction of
    each cycle, then stays at ``max_beta`` for the remainder of the cycle.
    Multiple cycles encourage the model to learn increasingly structured
    latent representations without mode collapse.

Both classes expose a single method ``get_beta(epoch)`` that returns the
scalar β value for that training epoch (1-indexed).

Usage
-----
    from src.autoencoder.beta_schedule import LinearBetaSchedule, CyclicalBetaSchedule

    schedule = LinearBetaSchedule(max_beta=4.0, anneal_epochs=100, delay_epochs=20)
    for epoch in range(1, 301):
        beta = schedule.get_beta(epoch)
        loss, _, _ = beta_vae_loss(recon, x, mu, log_var, beta=beta)
        ...
"""

from __future__ import annotations

import math


class LinearBetaSchedule:
    """
    Linear warm-up β annealing schedule.

    Timeline (all in epochs, 1-indexed):
        [1 … delay_epochs]                     : β = 0
        [delay_epochs+1 … delay_epochs+anneal] : β linearly 0 → max_beta
        [delay_epochs+anneal+1 … ∞]            : β = max_beta

    Parameters
    ----------
    max_beta      : float  — maximum (target) β value
    anneal_epochs : int    — number of epochs over which β rises from 0 to max_beta
    delay_epochs  : int    — number of epochs to keep β = 0 before annealing starts
                             (default 0 — start annealing from epoch 1)

    Examples
    --------
    >>> sched = LinearBetaSchedule(max_beta=4.0, anneal_epochs=100, delay_epochs=20)
    >>> sched.get_beta(1)      # 0.0  (in delay window)
    0.0
    >>> sched.get_beta(21)     # 0.04 (first annealing step)
    0.04
    >>> sched.get_beta(120)    # 4.0  (fully annealed)
    4.0
    >>> sched.get_beta(300)    # 4.0  (plateau)
    4.0
    """

    def __init__(
        self,
        max_beta:      float,
        anneal_epochs: int,
        delay_epochs:  int = 0,
    ) -> None:
        if max_beta < 0:
            raise ValueError(f"max_beta must be non-negative, got {max_beta}")
        if anneal_epochs < 1:
            raise ValueError(f"anneal_epochs must be >= 1, got {anneal_epochs}")
        if delay_epochs < 0:
            raise ValueError(f"delay_epochs must be >= 0, got {delay_epochs}")

        self.max_beta      = float(max_beta)
        self.anneal_epochs = int(anneal_epochs)
        self.delay_epochs  = int(delay_epochs)

    def get_beta(self, epoch: int) -> float:
        """
        Return the β value for the given training epoch (1-indexed).

        Parameters
        ----------
        epoch : int  — current epoch number (starts at 1)

        Returns
        -------
        beta : float  — KL weight in [0, max_beta]
        """
        # Epoch before annealing starts → β = 0
        if epoch <= self.delay_epochs:
            return 0.0

        # How many epochs have elapsed since the ramp started?
        elapsed = epoch - self.delay_epochs  # in [1, ...]

        # Linearly interpolate: after anneal_epochs steps we reach max_beta
        frac = min(elapsed / self.anneal_epochs, 1.0)
        return frac * self.max_beta

    def __repr__(self) -> str:
        return (
            f"LinearBetaSchedule("
            f"max_beta={self.max_beta}, "
            f"anneal_epochs={self.anneal_epochs}, "
            f"delay_epochs={self.delay_epochs})"
        )


class CyclicalBetaSchedule:
    """
    Cyclical annealing schedule for β-VAE (Fu et al., 2019).

    The training run is divided into ``n_cycles`` equal-length cycles.
    Within each cycle the first ``ratio`` fraction of epochs ramp β from
    0 → ``max_beta`` linearly; the remaining ``(1 - ratio)`` fraction holds
    β at ``max_beta``.  Multiple cycles push the model out of posterior
    collapse valleys and encourage increasingly informative latent codes.

    Parameters
    ----------
    max_beta      : float  — maximum β reached at the top of each ramp
    total_epochs  : int    — total training budget in epochs
    n_cycles      : int    — number of complete annealing cycles (default 4)
    ratio         : float  — fraction of each cycle spent ramping (default 0.5)

    References
    ----------
    Fu, H. et al. (2019). "Cyclical Annealing Schedule: A Simple Approach to
    Mitigating KL Vanishing." NAACL-HLT 2019.
    https://arxiv.org/abs/1903.10145

    Examples
    --------
    >>> sched = CyclicalBetaSchedule(max_beta=4.0, total_epochs=300, n_cycles=4, ratio=0.5)
    >>> sched.get_beta(1)    # near 0 (start of first ramp)
    >>> sched.get_beta(37)   # ~ max_beta (end of first ramp)
    >>> sched.get_beta(38)   # == max_beta (plateau of first cycle)
    >>> sched.get_beta(76)   # near 0 again (start of second ramp)
    """

    def __init__(
        self,
        max_beta:     float,
        total_epochs: int,
        n_cycles:     int   = 4,
        ratio:        float = 0.5,
    ) -> None:
        if max_beta < 0:
            raise ValueError(f"max_beta must be non-negative, got {max_beta}")
        if total_epochs < 1:
            raise ValueError(f"total_epochs must be >= 1, got {total_epochs}")
        if n_cycles < 1:
            raise ValueError(f"n_cycles must be >= 1, got {n_cycles}")
        if not (0.0 < ratio <= 1.0):
            raise ValueError(f"ratio must be in (0, 1], got {ratio}")

        self.max_beta     = float(max_beta)
        self.total_epochs = int(total_epochs)
        self.n_cycles     = int(n_cycles)
        self.ratio        = float(ratio)

        # Length of each cycle in epochs (may be fractional)
        self._cycle_len = self.total_epochs / self.n_cycles
        # Number of ramp epochs per cycle
        self._ramp_len  = self._cycle_len * self.ratio

    def get_beta(self, epoch: int) -> float:
        """
        Return the β value for the given training epoch (1-indexed).

        If ``epoch > total_epochs`` the schedule stays at ``max_beta``.

        Parameters
        ----------
        epoch : int  — current epoch number (starts at 1)

        Returns
        -------
        beta : float  — KL weight in [0, max_beta]
        """
        # Convert to 0-indexed position within a cycle
        t = (epoch - 1) % self._cycle_len   # position in current cycle [0, cycle_len)

        if t < self._ramp_len:
            # Linear ramp: 0 → max_beta
            frac = t / self._ramp_len
            return frac * self.max_beta
        else:
            # Plateau
            return self.max_beta

    def __repr__(self) -> str:
        return (
            f"CyclicalBetaSchedule("
            f"max_beta={self.max_beta}, "
            f"total_epochs={self.total_epochs}, "
            f"n_cycles={self.n_cycles}, "
            f"ratio={self.ratio})"
        )
