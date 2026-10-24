"""PB-ESAV: global exponential drive and first-gradient component feedback."""

import math

import torch

from optimizer import esav


def initialize(terms):
    """Freeze E0 and all Ei0; evolve only the total log auxiliary variable."""
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms:
        raise ValueError("Require at least one component")
    references = torch.stack([term.detach() for term in terms])
    if not torch.isfinite(references).all() or (references <= 0).any():
        raise ValueError("Each initial component energy must be finite and positive")
    state = esav.initialize(sum(terms))
    state["initial_components"] = references.clone()
    return state


def component_gradients(terms, parameters):
    """Return first derivatives in columns, with shape (parameter_count, M)."""
    columns = []
    for index, term in enumerate(terms):
        derivatives = torch.autograd.grad(
            term, parameters, retain_graph=index < len(terms) - 1, allow_unused=True,
        ) if term.requires_grad else [None] * len(parameters)
        columns.append(torch.cat([
            (derivative.detach() if derivative is not None else torch.zeros_like(parameter))
            .reshape(-1)
            for parameter, derivative in zip(parameters, derivatives)
        ]))
    gradients = torch.stack(columns, dim=1)
    if not torch.isfinite(gradients).all():
        raise FloatingPointError("Nonfinite component gradient")
    return gradients


def step(parameters, loss_fn, state, lr, *, relax=False):
    """Update in place; return detached state without mutating the input state.

    B = sum_i grad(Ei) grad(Ei).T / (4 Ei0). The M-by-M Gram solve
    evaluates the same inverse action without subtractive Woodbury cancellation.
    M=1 calls ESAV. Optional Relax corrects only the total log_r after the step.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms or len(terms) != state["initial_components"].numel():
        raise ValueError("Require a fixed nonempty component partition")
    if len(terms) == 1:
        result = esav.step(
            parameters, lambda: terms[0], state, lr,
        )
        result["initial_components"] = state["initial_components"].clone()
        if relax:
            result["log_r"] = esav._relaxed_log_r(loss_fn, state)
        return result

    energies = torch.stack([term.detach() for term in terms])
    if not torch.isfinite(energies).all() or (energies < 0).any():
        raise FloatingPointError("Require finite nonnegative component energies")
    references = state["initial_components"]
    gradients = component_gradients(terms, parameters)
    force = gradients.sum(dim=1)
    ratio = (state["log_r"] - sum(energies) / (4 * state["initial_energy"])).exp()
    scaled_lr = lr * ratio
    if not torch.isfinite(scaled_lr) or scaled_lr <= 0:
        raise FloatingPointError("Scaled learning rate overflowed or underflowed")
    columns = gradients / (4 * references).sqrt()
    weights = (4 * references).sqrt()
    small = torch.eye(len(terms), dtype=force.dtype, device=force.device)
    small = small + scaled_lr * (columns.T @ columns)
    if not torch.isfinite(small).all():
        raise FloatingPointError("Nonfinite PB linear system")
    try:
        increment = -scaled_lr * (columns @ torch.linalg.solve(small, weights))
    except torch.linalg.LinAlgError as error:
        raise FloatingPointError("PB component solve failed") from error
    log_change = torch.dot(force, increment) / (4 * state["initial_energy"])
    if not torch.isfinite(log_change) or not -1 - 1e-10 <= log_change <= 0:
        raise FloatingPointError(f"Invalid logarithmic auxiliary increment: {log_change.item()}")
    next_log_r = state["log_r"] + log_change
    with torch.no_grad():
        offset, candidates = 0, []
        for parameter in parameters:
            size = parameter.numel()
            candidates.append(parameter + increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if not all(torch.isfinite(candidate).all() for candidate in candidates):
            raise FloatingPointError("Nonfinite PB candidate parameters")
        for parameter, candidate in zip(parameters, candidates):
            parameter.copy_(candidate)
    if relax:
        next_log_r = esav._relaxed_log_r(loss_fn, state)
    return {
        "log_r": next_log_r.detach(),
        "initial_energy": state["initial_energy"].clone(),
        "initial_components": references.clone(),
    }
