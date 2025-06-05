import math

import numpy as np
import torch
from torch import nn, Tensor
from sympy import finite_diff_weights

from exprel import exprel


class PostFornbergCell(nn.Module):
    s: Tensor
    exp_neg_s: Tensor
    tau_stars: Tensor
    post: Tensor

    def __init__(
        self,
        tau_min: float,
        tau_max: float,
        n_taus: int,
        k: int,
        g: int,
    ) -> None:
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_taus = n_taus
        self.k = k
        self.g = g

        c = (tau_max / tau_min) ** (1 / (n_taus - 1))  # log spacing constant

        tau_stars = torch.tensor(np.geomspace(tau_min, tau_max, n_taus))
        self.register_buffer("tau_stars", tau_stars, persistent=False)

        assert k % 2 == 0, "k must be even"
        kd2 = k // 2
        s = k / (tau_min * c ** torch.arange(-kd2, n_taus + kd2, dtype=torch.float64))
        self.register_buffer("s", s, persistent=False)

        exp_neg_s = torch.exp(-self.s)
        self.register_buffer("exp_neg_s", exp_neg_s, persistent=False)
        
        n_s = len(s)
        self.n_s = n_s

        width = k + 1  # stencil width

        Dk = torch.zeros((n_taus, n_s), dtype=torch.float64)
        for tstr_ix in range(n_taus):
            s_ix = tstr_ix + kd2  # index of tau_star in s
            low = s_ix - (width // 2)
            high = s_ix + (width // 2)

            s_bar = k / tau_stars[tstr_ix]  # point to approximate, might not be a grid point

            s_points = s[low:high+1]  # grid points

            res = finite_diff_weights(k, s_points, s_bar)
            coeffs = res[k][-1]  # FD weights for kth derivative, using full x_list
            
            Dk[tstr_ix, low:high+1] = torch.tensor(coeffs, dtype=torch.float64)

        post: Tensor = (
            ((-1) ** k) * torch.exp(-math.lgamma(k + 1) + (k + 1) * torch.log(k / tau_stars)) * Dk.T 
        )

        # if g=1, multiply by tau_stars and divide by number of s per til_f
        post = post * (tau_stars / (k * 2 + 1)) ** g

        self.register_buffer("post", post, persistent=False)

    def get_init_F(self, n_batch: int, n_feat: int, device) -> Tensor:
        return torch.zeros((n_batch, n_feat, self.n_s), dtype=torch.float64, device=device)

    def forward(
        self,
        f: Tensor,  # (batch, feat)
        F: Tensor | None = None,  # (batch, feat, s)
        alpha: Tensor | None = None,  # (batch, feat)
    ) -> tuple[
        Tensor,  # til_f: (batch, feat, taustar)
        Tensor,  # F: (batch, feat, s)
    ]:
        alpha = alpha if alpha is not None else torch.ones_like(f)

        if f.shape != alpha.shape:
            raise ValueError(
                f"f and alpha must have the same shape, but "
                f"have shapes {f.shape} and {alpha.shape}."
            )

        if F is None:
            # Generate initial F
            n_batch, n_feat = f.shape
            F = self.get_init_F(n_batch, n_feat, f.device)

        # === Forward Laplace Transform ===
        s_mul_a = self.s * alpha[:, :, None]
        hh = self.exp_neg_s ** alpha[:, :, None]
        b = f[:, :, None] * exprel(-s_mul_a)  # input -> hidden
        
        # Update state
        F = F * hh + b

        ## === Inverse Laplace transform ===
        #til_f = F @ self.post
        #mean = torch.mean(F, dim=2)
        #std = torch.std(F, dim=2)
        #norm_mean = torch.mean(torch.asinh(F), dim=2)
        #norm_std = torch.std(torch.asinh(F), dim=2)
        #print(f"mean: {mean}\nstd: {std}\nnorm_mean: {norm_mean}\nnorm_std: {norm_std}")

        return torch.asinh(F)


# Keep the original PostFornberg class for backward compatibility
class PostFornberg(nn.Module):
    s: Tensor
    exp_neg_s: Tensor
    tau_stars: Tensor
    post: Tensor

    def __init__(
        self,
        tau_min: float,
        tau_max: float,
        n_taus: int,
        k: int,
        g: int,
        batch_first: bool = False,
    ) -> None:
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_taus = n_taus
        self.k = k
        self.g = g
        self.batch_first = batch_first

        c = (tau_max / tau_min) ** (1 / (n_taus - 1))  # log spacing constant

        tau_stars = torch.tensor(np.geomspace(tau_min, tau_max, n_taus))
        self.register_buffer("tau_stars", tau_stars, persistent=False)

        assert k % 2 == 0, "k must be even"
        kd2 = k // 2
        s = k / (tau_min * c ** torch.arange(-kd2, n_taus + kd2, dtype=torch.float64))
        self.register_buffer("s", s, persistent=False)

        exp_neg_s = torch.exp(-self.s)
        self.register_buffer("exp_neg_s", exp_neg_s, persistent=False)
        
        n_s = len(s)
        self.n_s = n_s

        width = k + 1  # stencil width

        Dk = torch.zeros((n_taus, n_s), dtype=torch.float64)
        for tstr_ix in range(n_taus):
            s_ix = tstr_ix + kd2  # index of tau_star in s
            low = s_ix - (width // 2)
            high = s_ix + (width // 2)

            s_bar = k / tau_stars[tstr_ix]  # point to approximate, might not be a grid point
            # s_bar = N(s_bar, 31)

            s_points = s[low:high+1]  # grid points
            # s_points = [N(p, 31) for p in s[low:high+1]]  # grid points

            res = finite_diff_weights(k, s_points, s_bar)
            coeffs = res[k][-1]  # FD weights for kth derivative, using full x_list
            
            Dk[tstr_ix, low:high+1] = torch.tensor(coeffs, dtype=torch.float64)

        post: Tensor = (
            ((-1) ** k) * torch.exp(-math.lgamma(k + 1) + (k + 1) * torch.log(k / tau_stars)) * Dk.T 
        )

        # if g=1, multiply by tau_stars and divide by number of s per til_f
        post = post * (tau_stars / (k * 2 + 1)) ** g

        self.register_buffer("post", post, persistent=False)

    def get_init_F(self, n_batch: int, n_feat: int, device) -> Tensor:
        return torch.zeros((n_batch, n_feat, self.n_s), dtype=torch.float64, device=device)

    def forward(
        self,
        fs: Tensor,  # (batch, seq, feat) if self.batch_first else (seq, batch, feat)
        F: Tensor | None = None,  # (batch, feat, s)
        alphas: Tensor | None = None,  #  <shape same as fs.shape>
    ) -> tuple[
        Tensor,  # til_F or Fs: (batch, seq, feat, taustar)
        Tensor,  # F: (batch, feat, s)
    ]:
        alphas = alphas if alphas is not None else torch.ones_like(fs)

        if fs.shape != alphas.shape:
            raise ValueError(
                f"fs and alphas must have the same shape, but "
                f"have shapes {fs.shape} and {alphas.shape}."
            )

        #if self.batch_first:
        #    # (batch, seq, feat) -> (seq, batch, feat)
        #    fs = fs.transpose(0, 1)
        #    alphas = alphas.transpose(0, 1)

        if F is None:
            # Generate initial F
            #_, n_batch, n_feat = fs.shape
            n_batch, n_feat = fs.shape
            F = fs.new_zeros((n_batch, n_feat, self.n_s))

        # === Forward Laplace Transform ===
        s_mul_a = self.s * alphas[..., None]
        # hh = torch.exp(-s_mul_a)
        hh = self.exp_neg_s ** alphas[:, :, None]
        b = fs[:, :, None]  * exprel(-s_mul_a)  # input -> hidden

        Fs = torch.empty_like(hh)
        for i in range(len(Fs)):
            # equivalent to: F = F * [e**(-s*a)] + [f * (e**(-s*a)-1)/(-s*a)]
            F = F * hh[i] + b[i]
            Fs[i] = F

        ## === Inverse Laplace transform ===
        #til_fs = Fs @ self.post

        #if self.batch_first:
        #    # (seq, batch, feat, taustar) -> (batch, seq, feat, taustar)
        #    til_fs = til_fs.transpose(0, 1)

        #return til_fs, F
        return torch.asinh(F)