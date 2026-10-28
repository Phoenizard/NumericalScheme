"""Shared iteration loop for directly optimized tensors or model parameters."""

import math

import torch

from optimizer import besav, esav, mesav, pb_besav, pb_esav, pb_mesav, pb_sav_momentum
from optimizer import pb_esav_momentum


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
            return trial_state, lr

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
          method=None, line_search=None, history=None,
          naive_esav_linearized=False, initial_state=None, on_record=None):
    """Record initial/accepted states and pass a copy of each row to optional on_record."""
    is_esav = method in ("esav", "pb_esav")
    is_besav = method in ("besav", "pb_besav")
    is_esav_momentum = method == "pb_esav_momentum"
    if method in ("pb_sav_momentum", "pb_esav_momentum") and line_search is not None:
        raise ValueError("PB momentum currently requires a fixed learning rate")
    if (is_esav or is_besav or is_esav_momentum) and (
            esav_scheme is not None or total_shift is not None):
        raise ValueError("ESAV/BESAV uses frozen initial energies, without a Plan or total_shift")
    if eta is not None:
        if esav_scheme != "plan1" or not relax:
            raise ValueError("eta is only supported by relaxed PB-naiveESAV Plan 1")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must be finite and in [0, 1]")
    if esav_scheme is not None:
        if esav_scheme not in ("plan1", "plan2"):
            raise ValueError(f"Unsupported ESAV scheme: {esav_scheme}")
        if total_shift is not None:
            raise ValueError("ESAV auxiliary energies do not use total_shift")
    elif not (is_esav or is_besav or is_esav_momentum) and method not in (
            "gd", "sgd", "adamw", "naive_esav"):
        total_shift = 1e-10 if total_shift is None else total_shift
    is_mesav = method in ("mesav", "pb_mesav")
    has_esav_state = (
        is_esav or is_besav or is_mesav or esav_scheme is not None or method == "naive_esav"
    )
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
    state = initial_state
    accepted_lr = 0.0
    for iteration in range(steps + 1):
        with torch.no_grad():
            terms = loss_fn()
            terms = [terms] if isinstance(terms, torch.Tensor) else list(terms)
            energy = sum(terms)
            row = {"iteration": iteration, "objective": energy.item(), "lr": accepted_lr}
            if is_esav_momentum or is_esav:
                if not torch.isfinite(energy) or energy < 0:
                    raise FloatingPointError(f"ESAV requires finite E>=0 at iteration {iteration}")
                if state is None:
                    if is_esav_momentum:
                        state = pb_esav_momentum.initialize(parameters, terms)
                    else:
                        initializer = pb_esav if method == "pb_esav" else esav
                        state = initializer.initialize(terms)
                row.update(r=state["log_r"].exp().item(),
                           r_current=(energy / (4 * state["initial_energy"])).exp().item())
            elif method == "pb_sav_momentum":
                if state is None:
                    state = pb_sav_momentum.initialize(parameters, terms, total_shift)
                row.update(r=state["q"].item(), r_current=(energy + total_shift).sqrt().item())
            elif method in ("inverse_sav", "pb_inverse_sav", "csav"):
                shifted = energy + total_shift
                if not torch.isfinite(shifted).all() or (shifted <= 0).any():
                    raise FloatingPointError(f"{method} requires E+C>0 at iteration {iteration}")
                current = shifted if method == "csav" else shifted.reciprocal()
                if q is None:
                    q = current.detach().clone()
                row.update(r=q.item(), r_current=current.item())
            elif is_mesav:
                shifted = energy + total_shift
                if not torch.isfinite(shifted) or shifted <= 0:
                    raise FloatingPointError(f"MESAV requires E+C>0 at iteration {iteration}")
                if state is None:
                    initializer = pb_mesav if method == "pb_mesav" else mesav
                    state = initializer.initialize(terms, total_shift)
                current_log = energy / state["reference_energy"] + 0.5 * shifted.log()
                row.update(r=state["log_r"].exp().item(), r_current=current_log.exp().item())
            elif is_besav:
                if not torch.isfinite(energy) or energy < 0:
                    raise FloatingPointError(f"BESAV requires finite E>=0 at iteration {iteration}")
                if state is None:
                    initializer = pb_besav if method == "pb_besav" else besav
                    state = initializer.initialize(terms)
                current_log, _ = besav.log_definition(energy, state["initial_energy"])
                row.update(r=state["log_r"].exp().item(), r_current=current_log.exp().item())
            elif method == "naive_esav" and naive_esav_linearized:
                if state is None:
                    state = {"r": energy.exp().detach()}
                row.update(r=state["r"].item(), r_current=energy.exp().item())
            elif has_esav_state:
                if state is None:
                    state = {"log_r": energy.detach().clone()}
                    if esav_scheme == "plan2":
                        state["log_r_i"] = torch.stack(terms).detach().clone()
                row.update(r=state["log_r"].exp().item(), r_current=energy.exp().item())
            elif method not in ("gd", "sgd", "adamw"):
                current = (energy + total_shift).sqrt()
                if q is None:
                    q = current.detach().clone()
                row.update(r=q.item(), r_current=current.item())
            if method in ("pb_esav", "pb_esav_momentum"):
                energies = torch.stack(terms)
                if not torch.isfinite(energies).all() or (energies < 0).any():
                    raise FloatingPointError(f"PB-ESAV requires finite E_i>=0 at {iteration}")
                pb_esav._validate_state(state, energies)
            if len(terms) > 1:
                for i, value in enumerate(terms, start=1):
                    row[f"energy_{i}"] = value.item()
            if metrics_fn is not None:
                row.update(metrics_fn())
        if not all(math.isfinite(value) for value in row.values()):
            if on_record is not None:
                on_record(dict(row))
            raise FloatingPointError(f"Non-finite result at iteration {iteration}: {row}")
        history.append(row)
        if on_record is not None:
            on_record(dict(row))
        if iteration < steps:
            accepted_lr = lr
            if line_search is not None:
                def trial_update(trial_lr):
                    if method == "gd":
                        return step_fn(parameters, loss_fn, trial_lr)
                    if method == "sgd":
                        return step_fn(parameters, loss_fn, trial_lr, state=state)
                    if is_esav or is_besav or is_mesav or method == "naive_esav":
                        return step_fn(parameters, loss_fn, state, trial_lr, relax=relax)
                    if method == "adamw":
                        return step_fn(parameters, loss_fn, state, trial_lr)
                    if esav_scheme is None:
                        return step_fn(
                            parameters, loss_fn, q.clone(), trial_lr, total_shift, relax,
                        )
                    return step_fn(
                        parameters, loss_fn, {key: value.clone() for key, value in state.items()},
                        trial_lr, scheme=esav_scheme, relax=relax, eta=eta,
                    )

                accepted_state, accepted_lr = _armijo_step(
                    parameters, loss_fn, trial_update, lr, line_search, iteration + 1,
                )
                if not has_esav_state and method not in ("adamw", "sgd"):
                    q = accepted_state
                else:
                    state = accepted_state
            elif method == "gd":
                step_fn(parameters, loss_fn, lr)
            elif method == "sgd":
                state = step_fn(parameters, loss_fn, lr, state=state)
            elif method in ("pb_sav_momentum", "pb_esav_momentum"):
                state = step_fn(parameters, loss_fn, state, lr, relax=relax)
            elif is_esav or is_besav or is_mesav or method == "naive_esav":
                state = step_fn(parameters, loss_fn, state, lr, relax=relax)
            elif method == "adamw":
                state = step_fn(parameters, loss_fn, state, lr)
            elif esav_scheme is None:
                q = step_fn(parameters, loss_fn, q, lr, total_shift, relax)
            else:
                state = step_fn(parameters, loss_fn, state, lr, scheme=esav_scheme,
                                relax=relax, eta=eta)
    return history
