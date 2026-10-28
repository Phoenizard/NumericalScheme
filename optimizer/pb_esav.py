"""PB-ESAV with total and component exponential auxiliary states."""

import math

import torch

from optimizer import esav


def initialize(terms):
    """Freeze positive E0 and Ei0; initialize scalar log_r and length-M log_r_i."""
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms or any(
            not isinstance(term, torch.Tensor) or term.ndim != 0
            or not term.is_floating_point() for term in terms):
        raise ValueError("Require a nonempty partition of scalar floating-point energies")
    references = torch.stack([term.detach() for term in terms])
    if not torch.isfinite(references).all() or (references <= 0).any():
        raise ValueError("Each initial component energy must be finite and positive")
    state = esav.initialize(sum(terms))
    state["initial_components"] = references.clone()
    state["log_r_i"] = torch.full_like(references, 0.25)
    return state


def _validate_state(state, energies):
    """Require total state and fixed, finite length-M component state."""
    esav._validate_state(state, energies.sum())
    for name in ("initial_components", "log_r_i"):
        value = state.get(name)
        if (not isinstance(value, torch.Tensor) or value.shape != energies.shape
                or value.dtype != energies.dtype or value.device != energies.device):
            raise ValueError(f"Initialize PB-ESAV {name} for the fixed component partition")
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Nonfinite PB-ESAV state: {name}")
    if (state["initial_components"] <= 0).any():
        raise ValueError("Frozen component energies must be positive")


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

    B weights each gradient outer product by its old component ratio squared.
    A reduced QR solve evaluates (I + lr B)^-1 on the total gradient.
    Relax corrects only the total log_r; log_r_i always follows its log equation.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    parameters = tuple(parameters)
    if not parameters:
        raise ValueError("Require at least one parameter tensor")
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms or any(
            not isinstance(term, torch.Tensor) or term.ndim != 0 for term in terms):
        raise ValueError("Require a fixed nonempty partition of scalar energies")
    energies = torch.stack([term.detach() for term in terms])
    if not torch.isfinite(energies).all() or (energies < 0).any():
        raise FloatingPointError("Require finite nonnegative component energies")
    _validate_state(state, energies)
    references = state["initial_components"]
    gradients = component_gradients(terms, parameters)
    force = gradients.sum(dim=1)
    ratio = (state["log_r"] - energies.sum() / (4 * state["initial_energy"])).exp()
    component_ratios = (state["log_r_i"] - energies / (4 * references)).exp()
    component_ratios.masked_fill_(
        component_ratios == 0, torch.finfo(energies.dtype).tiny,
    )
    scaled_lr = lr * ratio
    if (not all(torch.isfinite(value).all() for value in (
            force, ratio, component_ratios, scaled_lr,
    )) or ratio <= 0 or scaled_lr <= 0):
        raise FloatingPointError("PB-ESAV auxiliary scaling overflowed or underflowed")
    try:
        basis, reduced = torch.linalg.qr(gradients, mode="reduced")
        weighted = reduced * (component_ratios / (4 * references).sqrt())
        small = torch.eye(basis.shape[1], dtype=force.dtype, device=force.device)
        small = small + lr * (weighted @ weighted.T)
        if not torch.isfinite(small).all():
            raise FloatingPointError("Nonfinite PB-ESAV linear system")
        increment = -scaled_lr * (
            basis @ torch.linalg.solve(small, basis.T @ force)
        )
    except torch.linalg.LinAlgError as error:
        raise FloatingPointError("PB component solve failed") from error
    next_log_r_i = state["log_r_i"] + (
        component_ratios * (gradients.T @ increment) / (4 * references)
    )
    if not torch.isfinite(increment).all() or not torch.isfinite(next_log_r_i).all():
        raise FloatingPointError("Nonfinite PB-ESAV parameter or component auxiliary update")
    next_log_r = state["log_r"]
    if not relax:
        log_change = ratio * torch.dot(force, increment) / (4 * state["initial_energy"])
        next_log_r = state["log_r"] + log_change
        if not torch.isfinite(next_log_r) or log_change > 0:
            raise FloatingPointError(f"Invalid PB-ESAV auxiliary increment: {log_change.item()}")
    with torch.no_grad():
        offset, candidates = 0, []
        for parameter in parameters:
            size = parameter.numel()
            candidates.append(parameter + increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if not all(torch.isfinite(candidate).all() for candidate in candidates):
            raise FloatingPointError("Nonfinite PB candidate parameters")
        previous = [parameter.clone() for parameter in parameters] if relax else []
        for parameter, candidate in zip(parameters, candidates):
            parameter.copy_(candidate)
        if relax:
            try:
                next_log_r = esav._relaxed_log_r(loss_fn, state)
            except Exception:
                for parameter, value in zip(parameters, previous):
                    parameter.copy_(value)
                raise
    return {
        "log_r": next_log_r.detach(),
        "log_r_i": next_log_r_i.detach(),
        "initial_energy": state["initial_energy"].detach().clone(),
        "initial_components": references.detach().clone(),
    }
