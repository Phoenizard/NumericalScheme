"""ESAV with r = exp(E/(4 E0)) and a frozen positive initial energy E0."""

import math

import torch


def initialize(terms):
    """Freeze positive E0; accept a scalar loss or component losses."""
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    if (not isinstance(energy, torch.Tensor) or energy.ndim != 0
            or not energy.is_floating_point()):
        raise ValueError("Require a scalar total energy")
    energy = energy.detach().clone()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError("The initial total energy must be finite and positive")
    return {"log_r": torch.full_like(energy, 0.25), "initial_energy": energy}


def _validate_state(state, energy):
    """Require finite scalar state on the loss dtype and device."""
    for name in ("log_r", "initial_energy"):
        value = state.get(name)
        if (not isinstance(value, torch.Tensor) or value.ndim != 0
                or value.dtype != energy.dtype or value.device != energy.device):
            raise ValueError(f"Initialize ESAV {name} with the loss dtype and device")
        if not torch.isfinite(value):
            raise FloatingPointError(f"Nonfinite ESAV state: {name}")
    if state["initial_energy"] <= 0:
        raise ValueError("The frozen initial energy must be positive")


def _relaxed_log_r(loss_fn, state):
    """Correct only log_r from the new total energy, keeping the old state intact."""
    with torch.no_grad():
        terms = loss_fn()
        energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
        if not isinstance(energy, torch.Tensor) or energy.ndim != 0:
            raise ValueError("Relax requires a scalar total energy")
        if not torch.isfinite(energy) or energy < 0:
            raise FloatingPointError("Relax requires a finite nonnegative total energy")
        next_log_r = torch.minimum(
            energy / (4 * state["initial_energy"]), state["log_r"],
        ).detach()
        if not torch.isfinite(next_log_r):
            raise FloatingPointError("Nonfinite relaxed ESAV state")
        return next_log_r


def step(parameters, loss_fn, state, lr, *, relax=False):
    """Update in place and return detached state without mutating input state.

    Use the old auxiliary ratio for the explicit parameter step and its square
    for the logarithmic energy decrement, with frozen E0 > 0.
    With relax=True, replace log_r_next by min(E_next/(4 E0), log_r_old).
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("Require at least one parameter tensor")
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    if not isinstance(energy, torch.Tensor) or energy.ndim != 0:
        raise ValueError("Require a scalar total energy")
    if not torch.isfinite(energy) or energy < 0:
        raise FloatingPointError("Expected a finite nonnegative total energy")
    _validate_state(state, energy)
    derivatives = torch.autograd.grad(
        energy, parameters, allow_unused=True,
    ) if energy.requires_grad else [None] * len(parameters)
    gradients = [
        derivative.detach() if derivative is not None else torch.zeros_like(parameter)
        for parameter, derivative in zip(parameters, derivatives)
    ]
    value = energy.detach() / (4 * state["initial_energy"])
    ratio = (state["log_r"] - value).exp()
    gradient_squared = sum(gradient.square().sum() for gradient in gradients)
    coefficient = lr * ratio
    if not all(torch.isfinite(value).all() for value in (
        ratio, gradient_squared, coefficient,
    )) or ratio <= 0 or coefficient <= 0:
        raise FloatingPointError("Non-finite ESAV update")
    next_log_r = state["log_r"]
    if not relax:
        next_log_r = state["log_r"] - (
            lr * ratio.square() * gradient_squared / (4 * state["initial_energy"])
        )
        if not torch.isfinite(next_log_r):
            raise FloatingPointError("Nonfinite ESAV logarithmic auxiliary update")
    with torch.no_grad():
        candidates = [
            parameter - coefficient * gradient
            for parameter, gradient in zip(parameters, gradients)
        ]
        if not all(torch.isfinite(candidate).all() for candidate in candidates):
            raise FloatingPointError("Nonfinite ESAV candidate parameters")
        previous = [parameter.clone() for parameter in parameters] if relax else []
        for parameter, candidate in zip(parameters, candidates):
            parameter.copy_(candidate)
        if relax:
            try:
                next_log_r = _relaxed_log_r(loss_fn, state)
            except Exception:
                for parameter, value in zip(parameters, previous):
                    parameter.copy_(value)
                raise
    return {
        "log_r": next_log_r.detach(),
        "initial_energy": state["initial_energy"].detach().clone(),
    }
