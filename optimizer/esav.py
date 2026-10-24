"""ESAV with r = exp(E/(4 E0)) and a frozen positive initial energy E0."""

import math

import torch


def initialize(terms):
    """Freeze positive E0; accept a scalar loss or component losses."""
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    energy = energy.detach().clone()
    if not torch.isfinite(energy) or energy <= 0:
        raise ValueError("The initial total energy must be finite and positive")
    return {"log_r": energy / (4 * energy), "initial_energy": energy}


def _relaxed_log_r(loss_fn, state):
    """Correct only log_r from the new total energy, keeping the old state intact."""
    with torch.no_grad():
        terms = loss_fn()
        energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
        if not torch.isfinite(energy) or energy < 0:
            raise FloatingPointError("Relax requires a finite nonnegative total energy")
        return torch.minimum(
            energy / (4 * state["initial_energy"]), state["log_r"],
        ).detach()


def step(parameters, loss_fn, state, lr, *, relax=False):
    """Update in place and return detached state without mutating input state.

    Only first derivatives are used. Without Relax, implicit gradient feedback gives
    log_r_next - log_r = -feedback/(1 + feedback) <= 0 for every positive lr.
    With relax=True, replace log_r_next by min(E_next/(4 E0), log_r_old).
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    if not torch.isfinite(energy) or energy < 0:
        raise FloatingPointError("Expected a finite nonnegative total energy")
    gradients = torch.autograd.grad(energy, parameters)
    value = energy.detach() / (4 * state["initial_energy"])
    derivative = 1 / (4 * state["initial_energy"])
    ratio = (state["log_r"] - value).exp()
    gradient_squared = sum(gradient.square().sum() for gradient in gradients)
    feedback = lr * ratio * derivative * gradient_squared
    coefficient = lr * ratio / (1 + feedback)
    next_log_r = state["log_r"] - feedback / (1 + feedback)
    if not all(torch.isfinite(value).all() for value in (
        ratio, gradient_squared, feedback, coefficient, next_log_r,
    )):
        raise FloatingPointError("Non-finite ESAV update")
    with torch.no_grad():
        for parameter, gradient in zip(parameters, gradients):
            parameter.add_(-coefficient * gradient)
    if relax:
        next_log_r = _relaxed_log_r(loss_fn, state)
    return {
        "log_r": next_log_r.detach(),
        "initial_energy": state["initial_energy"].detach().clone(),
    }
