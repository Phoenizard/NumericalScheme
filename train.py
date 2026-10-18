"""Shared iteration loop for directly optimized tensors or model parameters."""

import math

import torch


def _armijo_step(parameters, loss_fn, update_fn, initial_lr, settings, iteration):
    """Recompute each trial at its learning rate, restarting from initial_lr every update."""
    original = [parameter.detach().clone() for parameter in parameters]
    terms = loss_fn()
    energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
    objective = energy.item()
    gradients = torch.autograd.grad(energy, parameters, allow_unused=True)
    rejected = 0
    nonfinite = 0

    def restore_parameters():
        with torch.no_grad():
            for parameter, value in zip(parameters, original):
                parameter.copy_(value)

    lr = initial_lr
    while True:
        trial_objective = float("nan")
        slope = float("nan")
        bound = float("nan")
        finite = False
        try:
            trial_state = update_fn(lr)
            with torch.no_grad():
                terms = loss_fn()
                trial_energy = terms if isinstance(terms, torch.Tensor) else sum(terms)
                trial_objective = trial_energy.item()
                slope = sum(
                    (gradient * (parameter - value)).sum().item()
                    for gradient, parameter, value in zip(gradients, parameters, original)
                    if gradient is not None
                )
                bound = objective + settings["armijo_c"] * slope
                states = trial_state.values() if isinstance(trial_state, dict) else [trial_state]
                finite = (
                    math.isfinite(trial_objective) and math.isfinite(slope)
                    and all(torch.isfinite(p).all().item() for p in parameters)
                    and all(torch.isfinite(s).all().item() for s in states if s is not None)
                )
        except FloatingPointError:
            # A nonfinite trial is counted and rejected; the accepted state is restored below.
            pass
        except Exception:
            restore_parameters()
            raise

        if finite and slope <= 0 and trial_objective <= bound:
            stats = {
                "initial_lr": initial_lr,
                "accepted_lr": lr,
                "rejected_trials": rejected,
                "nonfinite_trials": nonfinite,
                "directional_derivative": slope,
                "armijo_bound": bound,
            }
            return trial_state, stats

        restore_parameters()
        rejected += 1
        nonfinite += int(not finite)
        if lr <= settings["min_lr"]:
            raise FloatingPointError(
                f"Armijo failed at update {iteration}, lr={lr:g}: "
                f"objective={objective:.16g}, trial={trial_objective:.16g}, "
                f"bound={bound:.16g}, slope={slope:.16g}; "
                f"rejected_trials={rejected}, nonfinite_trials={nonfinite}. State restored."
            )
        lr = max(lr * settings["shrink"], settings["min_lr"])


def train(parameters, loss_fn, step_fn, steps=300, lr=10.0,
          total_shift=None, relax=False, *, esav_scheme=None, eta=None, metrics_fn=None,
          method=None, line_search=None, history=None):
    """Record initial/accepted states; rejected trials do not consume the update budget."""
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
    elif method != "gd":
        total_shift = 1e-10 if total_shift is None else total_shift
    if line_search is not None:
        if not all(math.isfinite(value) for value in (lr, *line_search.values())):
            raise ValueError("Line-search settings must be finite")
        if not 0 < line_search["min_lr"] <= lr:
            raise ValueError("Require 0 < min_lr <= initial learning rate")
        if not 0 < line_search["shrink"] < 1:
            raise ValueError("Require 0 < shrink < 1")
        if not 0 < line_search["armijo_c"] < 1:
            raise ValueError("Require 0 < armijo_c < 1")
    parameters = tuple(parameters)
    history = [] if history is None else history
    q = None
    state = None
    step_stats = {
        "initial_lr": 0.0, "accepted_lr": 0.0,
        "rejected_trials": 0, "nonfinite_trials": 0,
        "directional_derivative": 0.0, "armijo_bound": 0.0,
    }
    for iteration in range(steps + 1):
        with torch.no_grad():
            terms = loss_fn()
            terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
            energy = sum(terms)
            row = {"iteration": iteration, "objective": energy.item()}
            if esav_scheme is None and method != "gd":
                Q = (energy + total_shift).sqrt()
                if q is None:
                    q = Q.detach().clone()
                row.update(q=q.item(), Q=Q.item(), q_over_Q=(q / Q).item(),
                           modified_energy=q.square().item())
            elif esav_scheme is not None:
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
            if metrics_fn is not None:
                row.update(metrics_fn())
            if line_search is not None:
                row.update(step_stats)
                if iteration == 0:
                    row["armijo_bound"] = energy.item()
        if not all(math.isfinite(value) for value in row.values()):
            raise FloatingPointError(f"Non-finite result at iteration {iteration}")
        history.append(row)
        if iteration < steps:
            if line_search is not None:
                def trial_update(trial_lr):
                    if method == "gd":
                        return step_fn(parameters, loss_fn, trial_lr)
                    if esav_scheme is None:
                        return step_fn(
                            parameters, loss_fn, q.clone(), trial_lr, total_shift, relax,
                        )
                    return step_fn(
                        parameters, loss_fn, {key: value.clone() for key, value in state.items()},
                        trial_lr, scheme=esav_scheme, relax=relax, eta=eta,
                    )

                accepted_state, step_stats = _armijo_step(
                    parameters, loss_fn, trial_update, lr, line_search, iteration + 1,
                )
                if esav_scheme is None:
                    q = accepted_state
                else:
                    state = accepted_state
            elif method == "gd":
                step_fn(parameters, loss_fn, lr)
            elif esav_scheme is None:
                q = step_fn(parameters, loss_fn, q, lr, total_shift, relax)
            else:
                state = step_fn(parameters, loss_fn, state, lr, scheme=esav_scheme,
                                relax=relax, eta=eta)
    return history
