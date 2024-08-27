import math

import torch
from torch import nn, Tensor

from exprel import exprel


class Post(nn.Module):
    _tau_stars_full: Tensor
    _post: Tensor

    def __init__(
        self,
        tau_min: float,
        tau_max: float,
        n_taus: int,
        k: int,
        g: int,
        batch_first: bool = False,
        ret_F: bool = False,
    ) -> None:
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_taus = n_taus
        self.k = k
        self.g = g
        self.batch_first = batch_first
        self.ret_F = ret_F

        c = (tau_max / tau_min) ** (1 / (n_taus - 1))  # log spacing constant
        n_s = n_taus + (2 * k)
        self.n_s = n_s

        self.tau_stars = tau_min * c ** torch.arange(n_taus, dtype=torch.float64)
        self.s = k / (tau_min * c ** torch.arange(-k, n_taus + k, dtype=torch.float64))

        post = self._make_post(self.s, self.tau_stars, k, g)
        self.register_buffer("post", post, persistent=False)

    @staticmethod
    def _make_post(
        s: Tensor,  # shape (n_s,)
        tau_stars: Tensor,  # shape (n_tau,)
        k: int,
        g: float,
    ) -> Tensor:
        n_s = s.shape[0]

        deriv_matrix = torch.zeros((n_s, n_s), dtype=torch.float64)
        for i in range(1, n_s - 1):
            sp = s[i + 1]
            si = s[i]
            sm = s[i - 1]
            deriv_matrix[i, i - 1] = -(sp - si) / (si - sm) / (sp - sm)
            deriv_matrix[i, i] = (((sp - si) / (si - sm)) - ((si - sm) / (sp - si))) / (
                sp - sm
            )
            deriv_matrix[i, i + 1] = (si - sm) / (sp - si) / (sp - sm)

        post: Tensor = (
            ((-1) ** k)
            * torch.matrix_power(deriv_matrix, k).T
            * torch.exp(-math.lgamma(k + 1) + (k + 1) * torch.log(s))
        )[:, k:-k]

        # if g=1, multiply by tau_stars and divide by number of s per til_f
        post = post * (tau_stars / (k * 2 + 1)) ** g
        return post

    @staticmethod
    def Fs_transform_hook(Fs: Tensor) -> Tensor:
        """
        Override this to do custom transformations in the Laplace domain
        """
        return Fs

    def forward(
        self,
        fs: Tensor,  # (batch, seq, feat) if self.batch_first else (seq, batch, feat)
        F: Tensor | None = None,  # (1, batch, feat, s)
        alphas: Tensor | None = None,  #  <shape same as fs.shape>
    ) -> tuple[
        Tensor,  # til_F or Fs: (batch, seq, feat, taustar)
        Tensor,  # F: (1, batch, seq, feat * s)
    ]:
        device = fs.device
        alphas = alphas if alphas is not None else torch.ones_like(fs)

        if fs.shape != alphas.shape:
            raise ValueError(
                f"fs and alphas must have the same shape, but "
                f"have shapes {fs.shape} and {alphas.shape}."
            )

        if self.batch_first:
            # (batch, seq, feat) -> (seq, batch, feat)
            fs = fs.transpose(0, 1)
            alphas = alphas.transpose(0, 1)

        if F is None:
            # Generate initial F
            _, n_batch, n_feat = fs.shape
            F = fs.new_zeros((n_batch, n_feat, self.n_s))

        # === Forward Laplace Transform ===
        self.s = self.s.to(device)
        s_mul_a = self.s * alphas[..., None]  # outer product
        hh = torch.exp(-s_mul_a)  # hidden -> hidden
        ih = exprel(-s_mul_a)  # input -> hidden
        b = fs[..., None] * ih

        Fs = torch.empty_like(hh)
        for i in range(len(Fs)):
            # equivalent to: F = F * [e**(-s*a)] + [f * (e**(-s*a)-1)/(-s*a)]
            F = F * hh[i] + b[i]
            Fs[i] = F

        Fs = self.Fs_transform_hook(Fs)

        # Inverse Laplace transform
        til_fs = Fs @ self.post

        out = Fs[:, :, :, self.k : -self.k] if self.ret_F else til_fs

        if self.batch_first:
            # (seq, batch, feat, taustar) -> (batch, seq, feat, taustar)
            out = out.transpose(0, 1)

        return out, F
