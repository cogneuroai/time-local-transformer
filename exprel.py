from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor


class ExpRel(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(x: Tensor) -> Tensor:  # type: ignore[override]
        # Taylor-series approximation of exprel(x) = (exp(x) - 1.0) / x
        # taylor_exprel = 1 + (x / 2) * (1 + (x / 3) * (1 + (x / 4) * (1 + (x / 5))))

        taylor_exprel = (
            1 + (x / 2) + (x**2 / 6) + (x**3 / 24) + (x**4 / 120)
        )  # + (x**5 / 720)
        out = torch.where(x.abs() > 0.002, (x.exp() - 1.0) / x, taylor_exprel)
        return out

    apply: Callable[[Tensor], Tensor]

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple[Tensor, ...], output: Any) -> Any:
        (x,) = inputs
        ctx.save_for_backward(x, output)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Tensor) -> Tensor | None:  # vjp
        x, out = ctx.saved_tensors
        (grad_out,) = grad_outputs  # cotangent
        grad_x = None

        if ctx.needs_input_grad[0]:
            grad_x = grad_out * torch.where(
                torch.isclose(x, torch.zeros_like(x)), 0.5, (out * (x - 1) + 1) / x
            )

        return grad_x


exprel = ExpRel.apply  # alias for convenience
