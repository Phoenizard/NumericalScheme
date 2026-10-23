"""Whole-energy Relax for MESAV and PB-MESAV, without changing their parameter step."""

import torch

from optimizer import mesav, pb_mesav


def initialize(terms, total_shift: float, *, pullback: bool = False) -> dict:
    """Initialize the selected scheme with unchanged frozen initial energy scales."""
    scheme = pb_mesav if pullback else mesav
    return scheme.initialize(terms, total_shift)


def step(parameters, loss_fn, state: dict, lr: float, *, total_shift: float,
         relax: bool = True, pullback: bool = False) -> dict:
    """Return the corrected state; relax=False exactly recovers the selected base step.

    log_r_next = min(E_next/(E_initial+C) + log(E_next+C)/2, log_r_old).
    This uses only the total new energy and guarantees nonincrease of log_r
    without assuming objective descent. Input state and frozen scales are kept.
    """
    scheme = pb_mesav if pullback else mesav
    next_state = scheme.step(parameters, loss_fn, state, lr, total_shift=total_shift)
    if relax:
        with torch.no_grad():
            terms = loss_fn()
            energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
            if not torch.isfinite(energy) or energy + total_shift <= 0:
                raise FloatingPointError("Relax-MESAV requires finite positive E_next+C")
            target = energy / state["reference_energy"] + 0.5 * (energy + total_shift).log()
            next_state["log_r"] = torch.minimum(target, state["log_r"]).detach()
    return next_state
