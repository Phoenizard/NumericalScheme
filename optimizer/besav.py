"""BESAV with r = exp(E/(E+E0)) and frozen positive initial energy scales."""

import math

import torch

from optimizer.pb_esav import component_gradients


def log_definition(energy, initial_energy):
    """Return the logarithmic definition and its first energy derivative."""
    return energy / (energy + initial_energy), initial_energy / (energy + initial_energy).square()


def _initialize(terms, *, pullback=False):
    """Freeze positive initial energies; evolve only the total log auxiliary."""
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms:
        raise ValueError("Require nonempty energy terms")
    energies = torch.stack([term.detach() for term in terms])
    if not pullback:
        energies = energies.sum().reshape(1)
    if not torch.isfinite(energies).all() or (energies <= 0).any():
        raise ValueError("Initial energy scales must be finite and positive")
    total = sum(terms).detach().clone()
    return {
        "log_r": total / (total + total),
        "initial_energy": total.clone(),
        "initial_components": energies.clone(),
    }


def _step(parameters, loss_fn, state, lr, *, pullback=False, relax=False):
    """Update parameters in place and return a detached, independent state.

    PB uses an M by M Gram solve with first component gradients only.
    M=1 uses the scalar base recurrence exactly. Relax corrects log_r after
    the candidate parameter update; rejected trials may discard this state.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not pullback:
        terms = [sum(terms)]
    if not terms or len(terms) != state["initial_components"].numel():
        raise ValueError("Require a fixed nonempty component partition")
    energies = torch.stack([term.detach() for term in terms])
    if not torch.isfinite(energies).all() or (energies < 0).any():
        raise FloatingPointError("Require finite nonnegative energies")
    value, derivative = log_definition(energies.sum(), state["initial_energy"])
    ratio = (state["log_r"] - value).exp()
    scaled_lr = lr * ratio
    if not torch.isfinite(scaled_lr) or scaled_lr <= 0:
        raise FloatingPointError("Scaled learning rate overflowed or underflowed")
    gradients = component_gradients(terms, parameters)
    force = gradients.sum(dim=1)
    if len(terms) == 1:
        feedback = scaled_lr * derivative * force.square().sum()
        increment = -scaled_lr / (1 + feedback) * force
        log_change = -feedback / (1 + feedback)
    else:
        _, derivatives = log_definition(energies, state["initial_components"])
        columns = gradients * derivatives.sqrt()
        small = torch.eye(len(terms), dtype=force.dtype, device=force.device)
        small = small + scaled_lr * (columns.T @ columns)
        if not torch.isfinite(small).all():
            raise FloatingPointError("Nonfinite bounded PB linear system")
        try:
            increment = -scaled_lr * (
                columns @ torch.linalg.solve(small, derivatives.rsqrt())
            )
        except torch.linalg.LinAlgError as error:
            raise FloatingPointError("Bounded PB component solve failed") from error
        log_change = derivative * torch.dot(force, increment)
    next_log_r = state["log_r"] + log_change
    if not torch.isfinite(increment).all() or not torch.isfinite(next_log_r):
        raise FloatingPointError("Nonfinite bounded exponential update")
    if log_change > 1e-12:
        raise FloatingPointError("Bounded exponential auxiliary energy increased")
    with torch.no_grad():
        offset, candidates = 0, []
        for parameter in parameters:
            size = parameter.numel()
            candidates.append(parameter + increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if not all(torch.isfinite(candidate).all() for candidate in candidates):
            raise FloatingPointError("Nonfinite bounded exponential candidate parameters")
        for parameter, candidate in zip(parameters, candidates):
            parameter.copy_(candidate)
        if relax:
            next_terms = loss_fn()
            next_energy = next_terms if isinstance(next_terms, torch.Tensor) else sum(next_terms)
            if not torch.isfinite(next_energy) or next_energy < 0:
                raise FloatingPointError("Relax requires a finite nonnegative energy")
            next_value, _ = log_definition(next_energy, state["initial_energy"])
            next_log_r = torch.minimum(next_value, state["log_r"])
    return {
        "log_r": next_log_r.detach(),
        "initial_energy": state["initial_energy"].clone(),
        "initial_components": state["initial_components"].clone(),
    }


def initialize(terms):
    """Initialize the single-energy BESAV state from a scalar or component losses."""
    return _initialize(terms)


def step(parameters, loss_fn, state, lr, *, relax=False):
    """Update in place; optionally correct log_r using the new total energy."""
    return _step(parameters, loss_fn, state, lr, relax=relax)
