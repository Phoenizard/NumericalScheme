"""Single-energy ESAV with r = exp(E) * sqrt(E + C), for L=0 and G=-I."""

import math

import torch


def step(parameters, loss_fn, state: dict, lr: float, *, total_shift: float,
         relax=False) -> dict:
    """Update tensors in place; return detached log_r without mutating input state.

    Initialize scalar log_r = E + 0.5 * log(E + C). Both updates use old
    parameters and log_r. The logarithmic dissipation includes the squared
    auxiliary ratio and the factor 1 + 1 / (2 * (E + C)). No Taylor expansion
    or relaxation is used. All parameter tensors retain their shapes.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if not math.isfinite(total_shift) or total_shift < 0:
        raise ValueError("ESAV total_shift must be finite and nonnegative")
    if relax:
        raise ValueError("ESAV implements the non-relaxed scheme only")
    parameters = tuple(parameters)
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    shifted = energy.detach() + total_shift
    if not torch.isfinite(shifted).all() or (shifted <= 0).any():
        raise FloatingPointError("ESAV requires finite E + C > 0 before the update")
    gradients = torch.autograd.grad(energy, parameters)
    ratio = (state["log_r"] - (energy.detach() + 0.5 * shifted.log())).exp()
    increments = [-lr * ratio * gradient for gradient in gradients]
    next_log_r = state["log_r"] - lr * (1 + 0.5 / shifted) * sum(
        (ratio * gradient).square().sum() for gradient in gradients
    )
    if not all(torch.isfinite(value).all() for value in (ratio, next_log_r, *increments)):
        raise FloatingPointError("Non-finite ESAV auxiliary ratio or update")
    with torch.no_grad():
        for parameter, increment in zip(parameters, increments):
            parameter.add_(increment)
        new_terms = loss_fn()
        new_energy = new_terms if isinstance(new_terms, torch.Tensor) else sum(new_terms)
        if not torch.isfinite(new_energy).all() or (new_energy + total_shift <= 0).any():
            raise FloatingPointError("ESAV requires finite E + C > 0 after the update")
    return {"log_r": next_log_r.detach()}
