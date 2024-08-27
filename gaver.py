## from James Mochizuki-Freeman @Jammf

import math

import numpy as np
import torch

from laplace import Laplace


class Gaver(Laplace):
    def __init__(self, tau_min, tau_max, n_taus, max_fn_evals, g, batch_first=False):
        super().__init__(tau_min, tau_max, n_taus, g, batch_first)
        self.max_fn_evals = max_fn_evals

        ndiv2 = max_fn_evals // 2

        eta = torch.zeros(max_fn_evals).double()
        beta = torch.zeros(max_fn_evals).double()

        logsum = np.concatenate(([0], np.cumsum(np.log(np.arange(1, max_fn_evals + 1)))))
        for k in range(1, max_fn_evals + 1):
            inside_sum = 0.0
            for j in range((k + 1) // 2, min(k, ndiv2) + 1):
                inside_sum += math.exp(
                    (ndiv2 + 1) * np.log(j) - logsum[ndiv2 - j] + logsum[2 * j] - 2 * logsum[j] - logsum[k - j] -
                    logsum[2 * j - k])
            eta[k - 1] = np.log(2.0) * (-1) ** (k + ndiv2) * inside_sum
            beta[k - 1] = k * np.log(2.0)

        self.register_buffer("_eta", eta, persistent=False)
        self.register_buffer("_beta", beta, persistent=False)

        self.fn_evals = max_fn_evals

        # self._eta = torch.tensor(self._eta, dtype=torch.complex128)
        # self._beta = torch.tensor(self._beta)

    @property
    def s(self):
        return torch.outer(1 / self.tau_stars, self._beta)

    def _inverse(self, h):
        til_f = torch.inner(self._eta, h) / self.tau_stars

        # if g=1, multiply by tau_stars and divide by number of s per til_f
        til_f = til_f * (self.tau_stars / self.fn_evals) ** self.g
        return til_f
