"""PB-SESAV with r = exp(k E/E0); design: doc/design/sesav.md."""

import math

import torch

from optimizer.pb_esav import component_gradients


def initialize(terms, k):
    """Freeze positive E0 and Ei0; the only auxiliary state is the total log_r = k."""
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms or any(
            not isinstance(term, torch.Tensor) or term.ndim != 0
            or not term.is_floating_point() for term in terms):
        raise ValueError("Require a nonempty partition of scalar floating-point energies")
    if not math.isfinite(k) or k <= 0:
        raise ValueError("The scale k must be finite and positive")
    references = torch.stack([term.detach() for term in terms])
    if not torch.isfinite(references).all() or (references <= 0).any():
        raise ValueError("Each initial component energy must be finite and positive")
    return {
        "log_r": torch.full_like(references[0], k),
        "initial_energy": references.sum().clone(),
        "initial_components": references.clone(),
        "k": torch.full_like(references[0], k),
    }


def total_ratio(state, energy):
    """Return r / exp(k E/E0) at the total energy E."""
    return (state["log_r"] - state["k"] * energy / state["initial_energy"]).exp()


def curvature_weights(state, energies):
    """Return (r/exp(k E/E0))^2 k / E_i0 for each component; B = sum_i w_i g_i g_i^T."""
    return total_ratio(state, energies.sum()).square() * state["k"] / state["initial_components"]


def step(parameters, loss_fn, state, lr, *, relax=False):
    """Update parameters in place; return a new detached state; the input is not mutated.

    theta_next = theta - lr ratio (I + lr B)^-1 grad E, then log_r follows its first-order
    equation; one component gives the explicit SESAV step (B = 0).
    With relax=True, log_r_next = min(k E_next/E0, log_r_old).
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("Require at least one parameter tensor")
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    energies = torch.stack([term.detach() for term in terms])
    if energies.shape != state["initial_components"].shape:
        raise ValueError("The number of energy components must remain constant")
    if not torch.isfinite(energies).all() or (energies < 0).any():
        raise FloatingPointError("Require finite nonnegative component energies")
    gradients = component_gradients(terms, parameters)
    force = gradients.sum(dim=1)
    energy = energies.sum()
    ratio = total_ratio(state, energy)
    if not torch.isfinite(ratio) or ratio <= 0:
        raise FloatingPointError(f"SESAV auxiliary ratio overflowed or underflowed: {ratio}")
    rhs = -lr * ratio * force
    if len(terms) == 1:
        increment = rhs
    else:
        columns = gradients * curvature_weights(state, energies).sqrt()  # B = columns columns^T
        small = torch.eye(len(terms), dtype=force.dtype, device=force.device)
        small = small + lr * (columns.T @ columns)
        if not torch.isfinite(small).all():
            raise FloatingPointError("Nonfinite PB-SESAV linear system")
        try:
            increment = rhs - lr * columns @ torch.linalg.solve(small, columns.T @ rhs)
        except torch.linalg.LinAlgError as error:
            raise FloatingPointError("PB-SESAV component solve failed") from error
    log_change = state["k"] / state["initial_energy"] * ratio * torch.dot(force, increment)
    next_log_r = state["log_r"] + log_change
    if not torch.isfinite(increment).all() or not torch.isfinite(next_log_r) or log_change > 0:
        raise FloatingPointError(f"Invalid SESAV update: log_change={log_change.item()}")
    with torch.no_grad():
        previous = [parameter.clone() for parameter in parameters] if relax else []
        offset = 0
        for parameter in parameters:
            size = parameter.numel()
            parameter.add_(increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if relax:
            new_terms = loss_fn()
            new_energy = (new_terms if isinstance(new_terms, torch.Tensor)
                          else sum(new_terms)).detach()
            if not torch.isfinite(new_energy):
                for parameter, value in zip(parameters, previous):
                    parameter.copy_(value)
                raise FloatingPointError("Relax requires a finite new energy")
            next_log_r = torch.minimum(
                state["k"] * new_energy / state["initial_energy"], state["log_r"])
    return {
        "log_r": next_log_r.detach(),
        "initial_energy": state["initial_energy"].detach().clone(),
        "initial_components": state["initial_components"].detach().clone(),
        "k": state["k"].detach().clone(),
    }
