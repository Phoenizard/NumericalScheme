"""SESAV is the one-component, explicit case of the PB-SESAV update."""

import torch

from optimizer import pb_sesav


def initialize(terms, k):
    """Freeze the total initial energy and initialize log_r = k."""
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    return pb_sesav.initialize(energy, k)


def step(parameters, loss_fn, state, lr, *, relax=False):
    """Update parameters in place; return detached state without mutating input state."""
    def total_loss():
        terms = loss_fn()
        return terms if isinstance(terms, torch.Tensor) else sum(terms)

    return pb_sesav.step(parameters, total_loss, state, lr, relax=relax)
