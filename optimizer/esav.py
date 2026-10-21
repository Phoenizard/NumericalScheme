"""Original E-SAV for L=0 and G=-I (Liu and Li, 2020, Eq. 2.7)."""

import math

import torch


def step(parameters, loss_fn, state: dict, lr: float, *, linearized=False) -> dict:
    """Update parameters in place and return detached log_r; initialize log_r = E.

    Both updates use the old ratio r / exp(E). The log update includes its
    square, as in the original E-SAV scheme. With linearized=True, initialize
    state["r"] = exp(E) instead and update signed r by the first-order Taylor
    polynomial. Negative r is retained; this branch has no logarithmic energy.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    gradients = torch.autograd.grad(energy, parameters)
    ratio = (
        state["r"] * (-energy.detach()).exp() if linearized
        else (state["log_r"] - energy.detach()).exp()
    )
    increments = [-lr * ratio * gradient for gradient in gradients]
    dissipation = lr * sum(
        (ratio * gradient).square().sum() for gradient in gradients
    )
    next_state = (
        {"r": state["r"] * (1 - dissipation)} if linearized
        else {"log_r": state["log_r"] - dissipation}
    )
    if not all(torch.isfinite(value).all() for value in (ratio, *next_state.values(), *increments)):
        raise FloatingPointError("Non-finite E-SAV auxiliary ratio or update")
    with torch.no_grad():
        for parameter, increment in zip(parameters, increments):
            parameter.add_(increment)
    return {key: value.detach() for key, value in next_state.items()}
