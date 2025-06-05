import math

import numpy as np
import torch
from torch import nn, Tensor

from exprel import exprel


class GaverCell(nn.Module):
    s: Tensor
    exp_neg_s: Tensor
    _eta: Tensor
    tau_stars: Tensor

    def __init__(
        self,
        tau_min: float,
        tau_max: float,
        n_taus: int,
        fn_evals: int,
        g: int,
    ) -> None:
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_taus = n_taus
        self.fn_evals = fn_evals
        self.g = g

        tau_stars = torch.tensor(np.geomspace(tau_min, tau_max, n_taus))
        self.register_buffer("tau_stars", tau_stars, persistent=False)

        ndiv2 = fn_evals // 2

        eta = torch.zeros(fn_evals).double()
        beta = torch.zeros(fn_evals).double()

        logsum = np.concatenate(
            ([0], np.cumsum(np.log(np.arange(1, fn_evals + 1))))
        )
        for k in range(1, fn_evals + 1):
            inside_sum = 0.0
            for j in range((k + 1) // 2, min(k, ndiv2) + 1):
                inside_sum += math.exp(
                    (ndiv2 + 1) * np.log(j)
                    - logsum[ndiv2 - j]
                    + logsum[2 * j]
                    - 2 * logsum[j]
                    - logsum[k - j]
                    - logsum[2 * j - k]
                )
            eta[k - 1] = np.log(2.0) * (-1) ** (k + ndiv2) * inside_sum
            beta[k - 1] = k * np.log(2.0)

        self.register_buffer("_eta", eta, persistent=False)

        s = torch.outer(1 / self.tau_stars, beta)
        self.register_buffer("s", s, persistent=False)
        self.register_buffer("exp_neg_s", torch.exp(-self.s), persistent=False)
    
    def get_init_F(self, n_batch: int, n_feat: int, device) -> Tensor:
        return torch.zeros((n_batch, n_feat, self.n_taus, self.fn_evals), dtype=torch.float64, device=device)

    def forward(
        self,
        f: Tensor,  # (batch, feat)
        F: Tensor | None = None,  # (batch, feat, s)
        alpha: Tensor | None = None,  # (batch, feat)
    ) -> tuple[
        Tensor,  # til_F: (batch, feat, taustar)
        Tensor,  # F: (batch, feat, s)
    ]:
        alpha = alpha if alpha is not None else torch.ones_like(f)

        if f.shape != alpha.shape:
            raise ValueError(
                f"fs and alphas must have the same shape, but "
                f"have shapes {f.shape} and {alpha.shape}."
            )

        if F is None:
            # Generate initial F
            n_batch, n_feat = f.shape
            F = f.new_zeros((n_batch, n_feat, self.n_taus, self.fn_evals))

        # === Forward Laplace Transform ===
        s_mul_a = self.s * alpha[:, : , None, None]
        F = F * self.exp_neg_s ** alpha[:, :, None, None] + f[:, :, None, None] * exprel(-s_mul_a)

        # equivalent to: F = F * [e**(-s*a)] + f   # * (e**(-s*a)-1)/(-s*a)
        # s_mul_a = self.s * alpha[:, : , None, None]
        # F = F * torch.exp(-s_mul_a) + f * exprel(-s_mul_a)

        # === Inverse Laplace transform ===
        til_f = torch.inner(self._eta, F).real / self.tau_stars
        
        # if g=1, multiply by tau_stars and divide by number of s per til_f
        til_f = til_f * (self.tau_stars / self.fn_evals) ** self.g

        return til_f, F

    