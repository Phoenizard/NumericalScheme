"""PB-MESAV: global energy drive with first-gradient component corrections."""

import math

import torch

from optimizer import mesav


def initialize(terms, total_shift: float) -> dict:
    """Initialize one total log_r and freeze the positive initial component scales."""
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms:
        raise ValueError("PB-MESAV requires at least one component")
    state = mesav.initialize(sum(terms).detach(), total_shift)
    references = torch.stack([term.detach() for term in terms]) + total_shift / len(terms)
    if not torch.isfinite(references).all() or (references <= 0).any():
        raise ValueError("PB-MESAV requires every E_i_initial + C/M to be positive")
    state["reference_components"] = references.clone()
    return state


def _component_gradients(terms, parameters):
    """Return detached first derivatives with shape (parameter_count, M)."""
    columns = []
    for index, term in enumerate(terms):
        gradients = torch.autograd.grad(
            term, parameters, retain_graph=index < len(terms) - 1, allow_unused=True,
        ) if term.requires_grad else [None] * len(parameters)
        columns.append(torch.cat([
            (gradient.detach() if gradient is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]))
    result = torch.stack(columns, dim=1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Non-finite PB-MESAV component gradient")
    return result


def step(parameters, loss_fn, state: dict, lr: float, *, total_shift: float,
         relax: bool = False) -> dict:
    """Update parameters in place and return the new total auxiliary state.

    B = V V.T, where V_i = sqrt(1/A_i + 1/(2*(E_i+C/M))) * grad(E_i)
    and A_i = E_i_initial+C/M stays frozen. Solve the M-by-M Gram system,
    avoiding subtraction of nearly equal Woodbury terms. M=1 calls MESAV
    directly. Optional Relax corrects only the total log_r after the step.
    No local auxiliary states or second derivatives are evolved.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("PB-MESAV lr must be finite and positive")
    if not math.isfinite(total_shift) or total_shift <= 0:
        raise ValueError("PB-MESAV total_shift must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
    if not terms or len(terms) != state["reference_components"].numel():
        raise ValueError("PB-MESAV requires a fixed nonempty component partition")
    if len(terms) == 1:
        next_state = mesav.step(
            parameters, lambda: terms[0], state, lr, total_shift=total_shift,
        )
        next_state["reference_components"] = state["reference_components"].detach().clone()
        if relax:
            next_state["log_r"] = mesav._relaxed_log_r(loss_fn, state, total_shift)
        return next_state

    shifted = torch.stack([term.detach() for term in terms]) + total_shift / len(terms)
    energy = sum(term.detach() for term in terms)
    reference = state["reference_energy"].detach()
    references = state["reference_components"].detach()
    if not torch.isfinite(shifted).all() or (shifted <= 0).any():
        raise FloatingPointError("PB-MESAV requires positive finite component energies E_i+C/M")
    if not all(torch.isfinite(value).all()
               for value in (reference, references, state["log_r"])):
        raise FloatingPointError("Non-finite PB-MESAV auxiliary input")
    if reference <= 0 or (references <= 0).any():
        raise ValueError("PB-MESAV initial energy scales must be positive")
    gradients = _component_gradients(terms, parameters)
    force = gradients.sum(dim=1)
    coefficients = references.reciprocal() + 0.5 / shifted
    log_ratio = state["log_r"] - energy / reference - 0.5 * (energy + total_shift).log()
    scaled_lr = (math.log(lr) + log_ratio).exp()
    if not torch.isfinite(scaled_lr) or scaled_lr <= 0:
        raise FloatingPointError("PB-MESAV scaled learning rate overflowed or underflowed")
    columns = gradients * coefficients.sqrt()
    weights = coefficients.rsqrt()
    small = torch.eye(len(terms), dtype=force.dtype, device=force.device)
    small = small + scaled_lr * (columns.T @ columns)
    if not torch.isfinite(small).all():
        raise FloatingPointError("Non-finite PB-MESAV linear system")
    try:
        increment = -scaled_lr * (columns @ torch.linalg.solve(small, weights))
    except torch.linalg.LinAlgError as error:
        raise FloatingPointError("PB-MESAV component solve failed") from error
    chain_coefficient = reference.reciprocal() + 0.5 / (energy + total_shift)
    log_change = chain_coefficient * torch.dot(force, increment)
    if not torch.isfinite(log_change) or log_change > 0 or log_change < -1 - 1e-10:
        raise FloatingPointError(f"Invalid PB-MESAV log auxiliary increment: {log_change.item()}")
    next_log_r = state["log_r"] + log_change
    if not all(torch.isfinite(value).all() for value in (increment, next_log_r)):
        raise FloatingPointError("Non-finite PB-MESAV update")
    with torch.no_grad():
        candidates, offset = [], 0
        for parameter in parameters:
            size = parameter.numel()
            candidates.append(parameter + increment[offset:offset + size].reshape_as(parameter))
            offset += size
        if not all(torch.isfinite(candidate).all() for candidate in candidates):
            raise FloatingPointError("Non-finite PB-MESAV candidate parameters")
        for parameter, candidate in zip(parameters, candidates):
            parameter.copy_(candidate)
    if relax:
        next_log_r = mesav._relaxed_log_r(loss_fn, state, total_shift)
    return {"log_r": next_log_r.detach(), "reference_energy": reference.clone(),
            "reference_components": references.clone()}
