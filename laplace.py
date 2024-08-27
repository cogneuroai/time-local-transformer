## from James Mochizuki-Freeman @Jammf

import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from torch import Tensor


# def exprel(x: torch.Tensor) -> torch.Tensor:
#     # Taylor-series approximation of exprel(x) = (exp(x) - 1.0) / x
#     res = 1 + (x / 2) * (1 + (x / 3) * (1 + (x / 4) * (1 + (x / 5))))
#     return torch.where(x.abs() < 0.002, res, (x.exp() - 1.0) / x)


class ExpRel(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, *inputs: Tensor) -> Tensor:
        x, = inputs
        # Taylor-series approximation of exprel(x) = (exp(x) - 1.0) / x
        # taylor_exprel = 1 + (x / 2) * (1 + (x / 3) * (1 + (x / 4) * (1 + (x / 5))))

        taylor_exprel = 1 + (x / 2) + (x**2 / 6) + (x**3 / 24) + (x**4 / 120)  # + (x**5 / 720)
        out = torch.where(x.abs() > 0.002, (x.exp() - 1.0) / x, taylor_exprel)
        ctx.save_for_backward(x, out)
        return out

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tensor:  # vjp
        x, out = ctx.saved_tensors
        grad_out, = grad_outputs  # cotangent
        grad_x = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_out * torch.where(torch.isclose(x, torch.zeros_like(x)), 0.5, (out * (x - 1) + 1) / x)

        return grad_x


exprel = ExpRel.apply


class Laplace(torch.nn.Module, ABC):
    def __init__(self, tau_min: float, tau_max: float, n_taus: int, g: int, batch_first: bool = False) -> None:
        super().__init__()
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.n_taus = n_taus
        self.g = g
        self.batch_first = batch_first

        tau_stars = torch.tensor(np.geomspace(tau_min, tau_max, n_taus))
        self.register_buffer("tau_stars", tau_stars, persistent=False)

    def _laplace(self, f: Tensor, alpha: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        # f.shape = (seq, batch, feature)
        # alpha.shape = (seq, batch, feature)
        # h.shape = (batch, feature, s, s2)
        # h_acc.shape = (seq, batch, feature, s, s2)

        s_mul_a = self.s * alpha[..., None, None]  # outer product
        hh = torch.exp(-s_mul_a)  # hidden -> hidden
        ih = exprel(-s_mul_a)  # input -> hidden
        b = f[..., None, None] * ih

        h_acc = torch.empty_like(hh)
        for i in range(len(h_acc)):
            # equivalent to: h = h * e**(-s * a) + f * (e**(-s * a) - 1) / (-s * a)
            h = h * hh[i] + b[i]
            h_acc[i] = h

        if torch.isinf(hh).any():
            warnings.warn("Overflow encountered in Laplace transform. "
                          "This typically occurs when (-s * alpha) > 709. "
                          "Try increasing tau_min, or ensure alpha has no "
                          "large negative values.", RuntimeWarning)
            print(f"\n{(-s_mul_a.real).max() = }")
            print(f"\n{(-self.s).real.min() = }")
            print(f"\n{alpha.min() = }")

        return h_acc, h

    @abstractmethod
    def _inverse(self, h: Tensor) -> Tensor:
        pass

    @property
    @abstractmethod
    def s(self) -> Tensor:
        pass

    @staticmethod
    def hx_transform_hook(hx_acc: Tensor) -> Tensor:
        """
        Override this to do custom transformations in the Laplace domain
        """
        return hx_acc

    def forward(self, f: Tensor, hx: Tensor = None, alpha: Tensor = None) -> tuple[Tensor, Tensor]:
        alpha = alpha if alpha is not None else torch.ones_like(f)

        if f.shape != alpha.shape:
            raise ValueError(f"input and alpha must have the same shape, but "
                             f"have shapes {f.shape} and {alpha.shape}.")

        if self.batch_first:
            # (batch, seq, feat) -> (seq, batch, feat)
            f = f.transpose(0, 1)
            alpha = alpha.transpose(0, 1)

        f = f.double()
        alpha = alpha.double()

        if hx is None:
            hx = f.new_zeros((*f.shape[1:], *self.s.shape))
            #print("hx.shape = ", hx.shape)

        hx_acc, hx = self._laplace(f, alpha, hx)
        hx_acc = self.hx_transform_hook(hx_acc)
        til_f_acc = self._inverse(hx_acc)

        if self.batch_first:
            # (seq, batch, feat) -> (batch, seq, feat)
            til_f_acc = til_f_acc.transpose(0, 1)

        return til_f_acc, hx
