"""naiveESAV for L=0 and G=-I (Liu and Li, 2020, Eq. 2.7)."""

import math

import torch


def step(parameters, loss_fn, state: dict, lr: float, *, linearized=False, relax=False) -> dict:
    """Update parameters in place and return detached log_r; initialize log_r = E.

    Both updates use the old ratio r / exp(E). The log update includes its
    square, as in the naiveESAV scheme. With linearized=True, initialize
    state["r"] = exp(E) instead and update signed r by the first-order Taylor
    polynomial. Negative r is retained; this branch has no logarithmic energy.
    Full relaxation replaces log_r after the parameter update with
    min(E_new, log_r_old). It requires linearized=False; the input state is not mutated.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("lr must be finite and positive")
    if linearized and relax:
        raise ValueError("Full Relax requires naiveESAV (linearized=False)")
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
        raise FloatingPointError("Non-finite naiveESAV auxiliary ratio or update")
    with torch.no_grad():
        for parameter, increment in zip(parameters, increments):
            parameter.add_(increment)
        if relax:
            new_terms = loss_fn()
            new_energy = new_terms if isinstance(new_terms, torch.Tensor) else sum(new_terms)
            if not torch.isfinite(new_energy).all():
                raise FloatingPointError("Non-finite naiveESAV objective after the parameter update")
            # Returning all logarithmic dissipation caps the new log_r at its old value.
            next_state = {"log_r": torch.minimum(new_energy, state["log_r"])}
    return {key: value.detach() for key, value in next_state.items()}
