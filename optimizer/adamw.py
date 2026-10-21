"""AdamW with bias correction and decoupled weight decay."""

import math

import torch


def step(parameters, loss_fn, state: dict | None, lr: float, *, betas, eps, weight_decay):
    """Update parameters in place; return detached moments without changing input state.

    Moments are flat vectors in parameter order. Pass None for the first step.
    Returning a new state permits rejected trials to discard their moment updates.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if len(betas) != 2 or not all(math.isfinite(beta) and 0 <= beta < 1 for beta in betas):
        raise ValueError("betas must contain two finite values in [0, 1)")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    parameters = tuple(parameters)
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    gradients = torch.cat([
        gradient.reshape(-1) for gradient in torch.autograd.grad(energy, parameters)
    ])
    if state is None:
        state = {
            "step": torch.zeros((), dtype=torch.int64, device=gradients.device),
            "exp_avg": torch.zeros_like(gradients),
            "exp_avg_sq": torch.zeros_like(gradients),
        }
    if state["exp_avg"].shape != gradients.shape or state["exp_avg_sq"].shape != gradients.shape:
        raise ValueError("AdamW parameter shapes must remain constant")
    next_step = state["step"] + 1
    exp_avg = betas[0] * state["exp_avg"] + (1 - betas[0]) * gradients
    exp_avg_sq = betas[1] * state["exp_avg_sq"] + (1 - betas[1]) * gradients.square()
    increment = -lr * (exp_avg / (1 - betas[0] ** next_step.item())) / (
        (exp_avg_sq / (1 - betas[1] ** next_step.item())).sqrt() + eps
    )
    if not all(torch.isfinite(value).all() for value in (exp_avg, exp_avg_sq, increment)):
        raise FloatingPointError("Non-finite AdamW moment or update")
    with torch.no_grad():
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.mul_(1 - lr * weight_decay)
            parameter.add_(increment[offset:offset + size].reshape_as(parameter))
            offset += size
    return {
        "step": next_step.detach(),
        "exp_avg": exp_avg.detach(),
        "exp_avg_sq": exp_avg_sq.detach(),
    }
