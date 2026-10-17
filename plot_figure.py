"""Plot saved experiments without running an optimizer."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_experiment(input_dir):
    input_dir = Path(input_dir)
    config = json.loads((input_dir / "config.json").read_text())
    with (input_dir / "history.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The experiment history is empty")
    return config, rows


def plot_experiment(input_dir, output):
    config, rows = load_experiment(input_dir)
    iterations = [int(row["iteration"]) for row in rows]
    objectives = [float(row["objective"]) for row in rows]
    modified = [float(row["modified_energy"]) for row in rows]
    method = config["method"]
    if method == "pb_esav":
        label = f"PB-Exp-SAV {config['scheme']}"
        if config.get("relax", False):
            label += " (relaxed)"
            if config["scheme"] == "plan1":
                label += f", eta = {config.get('eta', 1.0):g}"
        modified_label = r"$\log r^n$"
    elif method in ("sav", "pb_sav"):
        label = "SAV" if method == "sav" else "PB-SAV"
        if config.get("relax", False):
            label += " (relaxed)"
        modified_label = r"$(q^n)^2$"
    else:
        raise ValueError(f"Unsupported method in experiment config: {method}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    # Zero has no finite logarithm; omit it only from the logarithmic plot.
    axes[0].semilogy(iterations, [v if v > 0 else float("nan") for v in objectives],
                     color="#3568a8", linewidth=1.8)
    axes[0].set(title="Objective energy", ylabel=r"$E(x^n)$")
    axes[1].plot(iterations, modified, color="#d16a2c", linewidth=1.8)
    axes[1].set(title="Modified energy", ylabel=modified_label)
    for axis in axes:
        axis.set_xlabel("Iteration (epoch)")
        axis.grid(True, alpha=0.2)
    fig.suptitle(f"{config['problem'].capitalize()} | {label} | step = {config['lr']:g}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_comparison(input_dirs, output):
    experiments = [load_experiment(path) for path in input_dirs]
    reference = experiments[0][0]
    shared = ("problem", "dimension", "initial_value", "device", "dtype",
              "lr", "steps", "lambda_reg")
    for config, _ in experiments[1:]:
        if any(config[key] != reference[key] for key in shared):
            raise ValueError("Comparison requires matching problem and experiment settings")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), layout="constrained")
    shifts = set()
    has_esav = False
    for config, rows in experiments:
        method = config["method"]
        marker = None
        if method == "pb_esav":
            scheme = config["scheme"]
            label = f"PB-Exp-SAV {scheme}"
            color, style = {"plan1": ("#218c60", "-."),
                            "plan2": ("#8953b1", ":")}[scheme]
            if config.get("relax", False):
                label = "relaxed " + label
                style = "-"
                if scheme == "plan1":
                    eta = config.get("eta", 1.0)
                    label += f" (eta={eta:g})"
                    if eta == 0.95:
                        style, marker = "--", "s"
                    elif eta == 0.1:
                        style, marker = ":", "^"
            ratio_key = "r_over_exp_E"
            has_esav = True
        elif method in ("sav", "pb_sav"):
            label = "SAV" if method == "sav" else f"rank-{config['components']} PB-SAV"
            color = "#3568a8" if method == "sav" else "#d16a2c"
            style = "-" if config.get("relax", False) else "--"
            if config.get("relax", False):
                label = "relaxed " + label
            ratio_key = "q_over_Q"
            shifts.add(config["total_shift"])
        else:
            raise ValueError(f"Unsupported method in experiment config: {method}")
        iterations = [int(row["iteration"]) for row in rows]
        objectives = [float(row["objective"]) for row in rows]
        ratios = [float(row[ratio_key]) for row in rows]
        axes[0].semilogy(iterations, [v if v > 0 else float("nan") for v in objectives],
                         color=color, linestyle=style, linewidth=1.7, label=label,
                         marker=marker, markevery=30, markersize=4)
        axes[1].plot(iterations, ratios, color=color, linestyle=style, linewidth=1.7,
                     marker=marker, markevery=30, markersize=4)

    axes[0].axhline(1e-10, color="#888888", linestyle=":", linewidth=1)
    axes[0].set(title="Objective energy", ylabel=r"$E(x^n)$")
    ratio_label = (r"$q^n/Q(x^n)$ (SAV/PB-SAV)" if shifts else "")
    if has_esav:
        ratio_label += ("\n" if ratio_label else "") + r"$r^n/\exp(E(x^n))$ (PB-Exp-SAV)"
    axes[1].set(title="Auxiliary-variable ratio", ylabel=ratio_label)
    for axis in axes:
        axis.set_xlabel("Iteration (epoch)")
        axis.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               fontsize=10, frameon=False)
    shift_note = ""
    if shifts:
        shift_note = "SAV/PB-SAV shift = " + ", ".join(f"{v:g}" for v in sorted(shifts))
    if has_esav:
        shift_note += ("; " if shift_note else "") + "PB-Exp-SAV: no shift"
    fig.suptitle(f"{reference['problem'].capitalize()} benchmark | step = {reference['lr']:g}, "
                 f"regularization = {reference['lambda_reg']:g}\n{shift_note}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="One or more directories containing config.json and history.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path")
    args = parser.parse_args()
    if len(args.input) == 1:
        plot_experiment(args.input[0], args.output)
    else:
        plot_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
