"""PB-BESAV with component feedback and a single total auxiliary state."""

from optimizer import besav


def initialize(terms):
    """Freeze the total and positive component initial energies."""
    return besav._initialize(terms, pullback=True)


def step(parameters, loss_fn, state, lr, *, relax=False):
    """Update in place with optional Relax; M=1 recovers the same BESAV mode."""
    return besav._step(parameters, loss_fn, state, lr, pullback=True, relax=relax)
