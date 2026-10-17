"""SAV is the one-component case of the PB-SAV update."""

import torch

from optimizer.pb_sav import step as pb_sav_step


def step(parameters, loss_fn, q, lr, total_shift, relax=False):
    def total_loss():
        terms = loss_fn()
        return terms if isinstance(terms, torch.Tensor) else sum(terms)

    return pb_sav_step(parameters, total_loss, q, lr, total_shift, relax)
