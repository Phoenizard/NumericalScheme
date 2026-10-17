"""Run a single quadratic experiment and save its complete history."""

import argparse
import csv
import json
import math
from functools import partial
from pathlib import Path

import torch

from optimizer import pb_esav, pb_sav, sav
from problem import quadratic
from train import train


def run_experiment(method="pb_sav", loss="split", lr=10.0, steps=300,
                   lambda_reg=0.0, total_shift=None, relax=False, output=None,
                   scheme="plan1", eta=None):
    if steps < 0 or not math.isfinite(lr) or lr <= 0:
        raise ValueError("steps must be nonnegative and lr must be finite and positive")
    if method == "pb_esav":
        if total_shift is not None:
            raise ValueError("PB-Exp-SAV does not support shift")
        if scheme not in ("plan1", "plan2"):
            raise ValueError(f"Unsupported PB-Exp-SAV scheme: {scheme}")
    else:
        total_shift = 1e-10 if total_shift is None else total_shift
        if not math.isfinite(total_shift) or total_shift <= 0:
            raise ValueError("total_shift must be finite and positive")
    if not math.isfinite(lambda_reg) or lambda_reg < 0:
        raise ValueError("lambda_reg must be finite and nonnegative")
    if eta is not None:
        if method != "pb_esav" or scheme != "plan1" or not relax:
            raise ValueError("eta is only supported by relaxed PB-Exp-SAV Plan 1")
        if not math.isfinite(eta) or not 0 <= eta <= 1:
            raise ValueError("eta must be finite and in [0, 1]")
    if method == "pb_esav" and scheme == "plan1" and relax:
        eta = 1.0 if eta is None else float(eta)

    if method == "sav":
        step_fn = sav.step
    elif method == "pb_sav":
        step_fn = pb_sav.step
    elif method == "pb_esav":
        step_fn = pb_esav.step
    else:
        raise ValueError(f"Unsupported method: {method}")

    x = quadratic.init_parameters()
    if loss == "split":
        loss_fn = partial(quadratic.split_energy_loss, x, lambda_reg=lambda_reg)
    elif loss == "total":
        loss_fn = partial(quadratic.total_energy_loss, x, lambda_reg=lambda_reg)
    else:
        raise ValueError(f"Unsupported loss: {loss}")

    history = train([x], loss_fn, step_fn, steps, lr, total_shift, relax,
                    esav_scheme=scheme if method == "pb_esav" else None, eta=eta)
    if method == "pb_esav":
        name = f"{method}_{scheme}" + ("_total" if loss == "total" else "")
        if relax:
            name += "_relaxed"
            if scheme == "plan1" and eta != 1:
                name += "_eta_" + str(eta).replace(".", "p")
        default_output = Path(__file__).parent / "results" / "quadratic" / name
    else:
        name = f"{method}_{loss}" + ("_relaxed" if relax else "")
        default_output = Path(__file__).parent / "results" / name
    output = Path(output) if output is not None else default_output
    output.mkdir(parents=True, exist_ok=True)
    with (output / "history.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    config = {
        "problem": "quadratic", "method": method, "loss": loss,
        "model": "none", "data": None, "dimension": x.numel(),
        "initial_value": 1.0, "device": str(x.device), "dtype": str(x.dtype),
        "lr": lr, "steps": steps, "lambda_reg": lambda_reg,
        "total_shift": total_shift, "relax": relax,
        "components": 2 if method in ("pb_sav", "pb_esav") and loss == "split" else 1,
        "torch_version": torch.__version__,
    }
    if method == "pb_esav":
        config.update(scheme=scheme, modified_energy="log_r")
        if scheme == "plan1" and relax:
            config["eta"] = eta
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    first = next((row["iteration"] for row in history if row["objective"] < 1e-10), None)
    print(f"{name}: final objective = {history[-1]['objective']:.8e}; "
          f"first < 1e-10 = {first if first is not None else 'not reached'}")
    print(f"  Results: {output.resolve()}")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=["quadratic"], default="quadratic")
    parser.add_argument("--method", choices=["sav", "pb_sav", "pb_esav"], default="pb_sav")
    parser.add_argument("--scheme", choices=["plan1", "plan2"], default="plan1",
                        help="PB-Exp-SAV scheme (default: plan1)")
    parser.add_argument("--loss", choices=["split", "total"], default="split")
    parser.add_argument("--model", choices=["none"], default="none")
    parser.add_argument("--data", default=None)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lambda-reg", type=float, default=0.0)
    parser.add_argument("--total-shift", type=float, default=None,
                        help="SAV/PB-SAV shift (default: 1e-10); unavailable for PB-Exp-SAV")
    parser.add_argument("--relax", action="store_true")
    parser.add_argument("--eta", type=float, default=None,
                        help="Relaxed PB-Exp-SAV Plan 1 dissipation budget in [0, 1] (default: 1)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.data is not None:
        parser.error("quadratic does not use external data; omit --data")
    if args.method == "pb_esav" and args.total_shift is not None:
        parser.error("PB-Exp-SAV does not support --total-shift")
    if args.eta is not None:
        if args.method != "pb_esav" or args.scheme != "plan1" or not args.relax:
            parser.error("--eta requires --method pb_esav --scheme plan1 --relax")
        if not math.isfinite(args.eta) or not 0 <= args.eta <= 1:
            parser.error("--eta must be finite and in [0, 1]")
    run_experiment(args.method, args.loss, args.lr, args.steps, args.lambda_reg,
                   args.total_shift, args.relax, args.output, args.scheme, args.eta)


if __name__ == "__main__":
    main()
