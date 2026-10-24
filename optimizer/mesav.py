"""Modified-ESAV with r = exp(E / (E_initial + C)) * sqrt(E + C)."""

import math

import torch


def initialize(terms, total_shift: float) -> dict:
    """Initialize log_r and freeze the positive initial total-energy scale."""
    if not math.isfinite(total_shift) or total_shift <= 0:
        raise ValueError("MESAV total_shift must be finite and positive")
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    reference = energy.detach() + total_shift
    if reference.ndim != 0 or not torch.isfinite(reference) or reference <= 0:
        raise ValueError("MESAV requires a finite positive scalar E_initial + C")
    return {
        "log_r": (energy.detach() / reference + 0.5 * reference.log()).clone(),
        "reference_energy": reference.clone(),
    }


def _relaxed_log_r(loss_fn, state: dict, total_shift: float):
    """Correct total log_r using the new energy and the unchanged old state."""
    with torch.no_grad():
        terms = loss_fn()
        energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
        if not torch.isfinite(energy) or energy + total_shift <= 0:
            raise FloatingPointError("Relax-MESAV requires finite positive E_next+C")
        target = energy / state["reference_energy"] + 0.5 * (energy + total_shift).log()
        return torch.minimum(target, state["log_r"]).detach()


def step(parameters, loss_fn, state: dict, lr: float, *, total_shift: float,
         relax: bool = False) -> dict:
    """Update parameters in place; return a detached state without changing its input.

    Only the total energy and its first gradient enter the update. The parameter
    feedback is linearized in the log auxiliary increment; logaddexp evaluates
    the equivalent scalar denominator without subtractive cancellation.
    Without Relax, E+C>0 and lr>0 imply -1 < log_r_next-log_r <= 0 without line search.
    Optional Relax replaces log_r_next using the new total energy and old log_r.
    """
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("MESAV lr must be finite and positive")
    if not math.isfinite(total_shift) or total_shift <= 0:
        raise ValueError("MESAV total_shift must be finite and positive")
    parameters = tuple(parameters)
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    if energy.ndim != 0:
        raise ValueError("MESAV requires a scalar total energy")
    shifted = energy.detach() + total_shift
    reference = state["reference_energy"].detach()
    old_log_r = state["log_r"].detach()
    if not all(value.ndim == 0 and torch.isfinite(value)
               for value in (shifted, reference, old_log_r)):
        raise FloatingPointError("Non-finite or nonscalar MESAV auxiliary input")
    if shifted <= 0 or reference <= 0:
        raise FloatingPointError("MESAV requires E+C and its initial scale to be positive")
    gradients = tuple(gradient.detach() for gradient in torch.autograd.grad(energy, parameters))
    norm_sq = sum(gradient.square().sum() for gradient in gradients)
    if not torch.isfinite(norm_sq):
        raise FloatingPointError("Non-finite MESAV gradient norm")
    log_ratio = old_log_r - energy.detach() / reference - 0.5 * shifted.log()
    chain_coefficient = reference.reciprocal() + 0.5 / shifted
    log_lr_ratio = math.log(lr) + log_ratio
    if norm_sq.item() == 0:
        log_drop = torch.zeros_like(old_log_r)
        log_denominator = torch.zeros_like(old_log_r)
    else:
        log_product = log_lr_ratio + chain_coefficient.log() + norm_sq.log()
        log_denominator = torch.logaddexp(torch.zeros_like(log_product), log_product)
        log_drop = (log_product - log_denominator).exp()
    effective_lr = (log_lr_ratio - log_denominator).exp()
    next_log_r = old_log_r - log_drop
    increments = tuple(-effective_lr * gradient for gradient in gradients)
    if not all(torch.isfinite(value).all()
               for value in (next_log_r, effective_lr, *increments)):
        raise FloatingPointError("Non-finite MESAV auxiliary or parameter update")
    with torch.no_grad():
        candidates = [parameter + increment
                      for parameter, increment in zip(parameters, increments)]
        if not all(torch.isfinite(candidate).all() for candidate in candidates):
            raise FloatingPointError("Non-finite MESAV candidate parameters")
        for parameter, candidate in zip(parameters, candidates):
            parameter.copy_(candidate)
    if relax:
        next_log_r = _relaxed_log_r(loss_fn, state, total_shift)
    return {"log_r": next_log_r.detach(), "reference_energy": reference.clone()}
