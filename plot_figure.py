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
    config = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    with (input_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The experiment history is empty")
    return config, rows


def step_label(config):
    """Describe the recorded step policy without treating adaptive runs as fixed-step runs."""
    search = config.get("line_search")
    if search is not None:
        if config.get("step_policy") == "armijo_lr_reset":
            return f"Armijo: reset lr={config['lr']:g}, halve on rejection"
        return f"adaptive Armijo [{search['min_lr']:g}, {search['max_lr']:g}]"
    return f"fixed step = {config['lr']:g}"


def esav_label(config):
    """Distinguish current, naive, and historical diagnostic ESAV records."""
    if config["method"] == "esav":
        return "ESAV (linearized r)" if config.get("linearized") else "ESAV"
    variant = config.get("esav_variant")
    diagnostic = config.get("debug_curvature", "")
    if variant == "exp_log_energy":
        name = "PB-ESAV"
    elif variant == "sqrt_energy" or diagnostic.startswith("Square-root energy pullback"):
        name = "PB-ESAV (former sqrt scheme)"
    elif diagnostic:
        name = "PB-ESAV (residual diagnostic)"
    else:
        # Records predating the rename used method=pb_esav for the naive scheme.
        name = "PB Naive ESAV"
    return f"{name} Plan {config['scheme'][-1]}"


def plot_experiment(input_dir, output):
    config, rows = load_experiment(input_dir)
    if config["method"] in ("gd", "sgd", "adamw"):
        return plot_comparison([input_dir], output)
    iterations = [int(row["iteration"]) for row in rows]
    objectives = [float(row["objective"]) for row in rows]
    modified_key = "r" if config.get("linearized") else "modified_energy"
    modified = [float(row[modified_key]) for row in rows]
    method = config["method"]
    if method in ("esav", "pb_esav", "pb_naive_esav"):
        label = esav_label(config)
        if config.get("relax", False):
            label += " (relaxed)"
            if config["scheme"] == "plan1":
                label += f", eta = {config.get('eta', 1.0):g}"
        modified_label = r"$r^n$" if config.get("linearized") else r"$\log r^n$"
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
    axes[1].set(title="Auxiliary variable" if config.get("linearized") else "Modified energy",
                ylabel=modified_label)
    for axis in axes:
        axis.set_xlabel("Iteration (epoch)")
        axis.grid(True, alpha=0.2)
    fig.suptitle(f"{config['problem'].capitalize()} | {label} | {step_label(config)}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_comparison(input_dirs, output):
    experiments = [load_experiment(path) for path in input_dirs]
    if any(config["problem"] == "nonlinear_regression" for config, _ in experiments):
        return plot_training_curves(input_dirs, output)
    reference = experiments[0][0]
    shared = ("problem", "dimension", "initial_value", "device", "dtype",
              "lr", "steps", "lambda_reg", "line_search")
    for config, _ in experiments[1:]:
        if any(config.get(key) != reference.get(key) for key in shared):
            raise ValueError("Comparison requires matching problem and experiment settings")

    has_auxiliary = any("r_over_exp_E" in rows[0] or "q_over_Q" in rows[0]
                        for _, rows in experiments)
    fig, axes = plt.subplots(1, 2 if has_auxiliary else 1, figsize=(14, 6.2),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    shifts = set()
    has_esav = False
    signed_ratio = False
    for config, rows in experiments:
        method = config["method"]
        marker = None
        marker_start = 15
        linewidth = 1.9
        if method in ("gd", "sgd", "adamw"):
            label = {"gd": "GD", "sgd": "SGD", "adamw": "AdamW"}[method]
            color = "#222222" if method != "adamw" else "#a6761d"
            style = "-"
            marker = "o" if method != "adamw" else "P"
            marker_start = 10 if method != "adamw" else 30
            linewidth = 2.4 if method != "adamw" else 1.6
            ratio_key = None
        elif method == "esav":
            label = esav_label(config)
            color = "#b3448c" if config.get("linearized") else "#d55e00"
            style = ":" if config.get("linearized") else "--"
            marker = "v" if config.get("linearized") else "^"
            marker_start = 20 if config.get("linearized") else 30
            ratio_key = "r_over_exp_E"
            has_esav = True
        elif method in ("pb_esav", "pb_naive_esav"):
            scheme = config["scheme"]
            label = esav_label(config)
            color, style, marker = {
                "plan1": ("#009e73", "-", "d"),
                "plan2": ("#6a3d9a", "--", "x"),
            }[scheme]
            marker_start = 10 if scheme == "plan1" else 30
            linewidth = 2.4 if scheme == "plan1" else 1.7
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
            color = "#0072b2" if method == "sav" else "#d16a2c"
            style = "-" if config.get("relax", False) else "--"
            marker = "s" if method == "sav" else "P"
            if config.get("relax", False):
                label = "relaxed " + label
            ratio_key = "q_over_Q"
            shifts.add(config["total_shift"])
        else:
            raise ValueError(f"Unsupported method in experiment config: {method}")
        iterations = [int(row["iteration"]) for row in rows]
        objectives = [float(row["objective"]) for row in rows]
        axes[0].semilogy(iterations, [v if v > 0 else float("nan") for v in objectives],
                         color=color, linestyle=style, linewidth=linewidth, label=label,
                         marker=marker, markevery=(marker_start, 40), markersize=6,
                         markerfacecolor="white", markeredgewidth=1.5)
        if ratio_key is not None:
            ratios = [float(row[ratio_key]) for row in rows]
            signed_ratio |= any(value < 0 for value in ratios)
            axes[1].plot(iterations, ratios, color=color, linestyle=style, linewidth=linewidth,
                         marker=marker, markevery=(marker_start, 40), markersize=6,
                         markerfacecolor="white", markeredgewidth=1.5)

    axes[0].set(title="Objective energy", ylabel=r"$E(x^n)$")
    ratio_label = (r"$q^n/Q(x^n)$ (SAV/PB-SAV)" if shifts else "")
    if has_esav:
        ratio_label += ("\n" if ratio_label else "") + r"$r^n/\exp(E(x^n))$ (ESAV)"
    if has_auxiliary:
        axes[1].set(title="Auxiliary-variable ratio", ylabel=ratio_label)
        if signed_ratio:
            axes[1].set_yscale("symlog", linthresh=1e-3)
            axes[1].set_title("Auxiliary-variable ratio (symlog)")
    for axis in axes:
        axis.set_xlabel("Iteration (epoch)")
        axis.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               fontsize=11, frameon=False, handlelength=5, numpoints=2,
               columnspacing=2, labelspacing=0.8)
    shift_note = ""
    if shifts:
        shift_note = "SAV/PB-SAV shift = " + ", ".join(f"{v:g}" for v in sorted(shifts))
    if has_esav:
        shift_note += ("; " if shift_note else "") + "ESAV total auxiliary: exp(E)"
    fig.suptitle(f"{reference['problem'].capitalize()} benchmark | {step_label(reference)}, "
                 f"regularization = {reference['lambda_reg']:g}\n{shift_note}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_training_curves(input_dirs, output):
    """Compare saved training objectives and noise-free test MSE histories."""
    experiments = [load_experiment(path) for path in input_dirs]
    styles = {
        ("gd", None): ("GD", "#333333", "-.", None),
        ("sgd", None): ("SGD", "#333333", "-.", None),
        ("adamw", None): ("AdamW", "#007c91", "-", None),
        ("esav", None): ("ESAV", "#d89b00", "-", None),
        ("pb_sav", None): ("PB-SAV", "#3568a8", "-", None),
        ("pb_esav", "plan1"): ("PB-ESAV Plan 1", "#d16a2c", "--", None),
        ("pb_esav", "plan2"): ("PB-ESAV Plan 2", "#218c60", ":", "o"),
        ("pb_naive_esav", "plan1"): ("PB Naive ESAV Plan 1", "#8953b1", "--", None),
        ("pb_naive_esav", "plan2"): ("PB Naive ESAV Plan 2", "#8c564b", ":", "s"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for config, rows in experiments:
        style_method = config["method"]
        if style_method == "pb_esav" and esav_label(config).startswith("PB Naive ESAV"):
            style_method = "pb_naive_esav"
        label, color, style, marker = styles[(style_method, config.get("scheme"))]
        if config["method"] in ("esav", "pb_esav", "pb_naive_esav"):
            label = esav_label(config)
            if config.get("linearized"):
                color, style = "#cc503e", ":"
        label += f" ($M={config['components']}$)"
        iterations = [int(row["iteration"]) for row in rows]
        for axis, key in zip(axes, ("objective", "test_mse")):
            axis.plot(
                iterations, [float(row[key]) for row in rows], label=label,
                color=color, linestyle=style, linewidth=1.8, marker=marker,
                markevery=60, markersize=5, markerfacecolor="none",
            )
    axes[0].set(title="Training objective", ylabel=r"$F(\theta)$")
    axes[1].set(title="Test error (noise-free target)", ylabel="Test MSE")
    for axis in axes:
        axis.set_xlabel("Parameter updates")
        axis.grid(True, alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    reference = experiments[0][0]
    policies = list(dict.fromkeys(step_label(config) for config, _ in experiments))
    fig.suptitle(
        f"Nonlinear regression | {reference['steps']} updates\n" + "; ".join(policies)
    )
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
    parser.add_argument("--training", action="store_true",
                        help="Plot training objective and test MSE for nonlinear regression")
    args = parser.parse_args()
    if args.training:
        plot_training_curves(args.input, args.output)
    elif len(args.input) == 1:
        plot_experiment(args.input[0], args.output)
    else:
        plot_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
