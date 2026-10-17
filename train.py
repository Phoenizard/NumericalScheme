"""Shared iteration loop for directly optimized tensors or model parameters."""

import math

import torch


def train(parameters, loss_fn, step_fn, steps=300, lr=10.0,
          total_shift=None, relax=False, *, esav_scheme=None, eta=None):
    if eta is not None:
        if esav_scheme != "plan1" or not relax:
            raise ValueError("eta is only supported by relaxed PB-Exp-SAV Plan 1")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must be finite and in [0, 1]")
    if esav_scheme is not None:
        if esav_scheme not in ("plan1", "plan2"):
            raise ValueError(f"Unsupported PB-Exp-SAV scheme: {esav_scheme}")
        if total_shift is not None:
            raise ValueError("PB-Exp-SAV does not support shift")
    else:
        total_shift = 1e-10 if total_shift is None else total_shift
    parameters = tuple(parameters)
    history = []
    q = None
    state = None
    for iteration in range(steps + 1):
        with torch.no_grad():
            terms = loss_fn()
            terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
            energy = sum(terms)
            row = {"iteration": iteration, "objective": energy.item()}
            if esav_scheme is None:
                Q = (energy + total_shift).sqrt()
                if q is None:
                    q = Q.detach().clone()
                row.update(q=q.item(), Q=Q.item(), q_over_Q=(q / Q).item(),
                           modified_energy=q.square().item())
            else:
                energies = torch.stack(terms)
                if state is None:
                    state = {"log_r": energy.detach().clone()}
                    if esav_scheme == "plan2":
                        state["log_r_i"] = energies.detach().clone()
                row.update(log_r=state["log_r"].item(),
                           r_over_exp_E=(state["log_r"] - energy).exp().item(),
                           modified_energy=state["log_r"].item())
                for i, value in enumerate(energies, start=1):
                    row[f"energy_{i}"] = value.item()
                if esav_scheme == "plan2":
                    for i, (log_r_i, value) in enumerate(
                            zip(state["log_r_i"], energies), start=1):
                        row[f"log_r_{i}"] = log_r_i.item()
                        row[f"r_{i}_over_exp_E_{i}"] = (log_r_i - value).exp().item()
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f"Non-finite result at iteration {iteration}")
        history.append(row)
        if iteration < steps:
            if esav_scheme is None:
                q = step_fn(parameters, loss_fn, q, lr, total_shift, relax)
            else:
                state = step_fn(parameters, loss_fn, state, lr, scheme=esav_scheme,
                                relax=relax, eta=eta)
    return history
