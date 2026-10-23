"""Plot saved experiments without running an optimizer."""

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MESAV_STYLES = {
    "mesav": ("MESAV", "#60a5d1", "-", "o"),
    "pb_mesav": ("PB-MESAV", "#1765ad", "-", "s"),
    "relax_mesav": ("Relax-MESAV", "#9b6cc2", "--", "o"),
    "relax_pb_mesav": ("Relax-PB-MESAV", "#663b94", "--", "s"),
}


def load_experiment(input_dir):
    input_dir = Path(input_dir)
    config = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    # Read historical names without changing saved configurations or numerical records.
    if config["method"] == "esav" and config.get("esav_variant") != "exp_log_energy":
        config["method"] = "naive_esav"
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
    """Distinguish naiveESAV, PB-ESAV, and historical diagnostic records."""
    if config["method"] == "esav":
        return "ESAV"
    if config["method"] == "naive_esav":
        if config.get("linearized"):
            return "naiveESAV (linearized r)"
        return "naiveESAV (relaxed)" if config.get("relax") else "naiveESAV"
    variant = config.get("esav_variant")
    diagnostic = config.get("debug_curvature", "")
    if variant == "exp_log_energy":
        name = "PB-ESAV"
    elif variant == "sqrt_energy" or diagnostic.startswith("Square-root energy pullback"):
        name = "PB-ESAV (former sqrt scheme)"
    elif diagnostic:
        name = "PB-ESAV (residual diagnostic)"
    else:
        # Early PB-naiveESAV records used method=pb_esav without a variant tag.
        name = "PB-naiveESAV"
    return f"{name} Plan {config['scheme'][-1]}"


def set_objective_scale(axis, values):
    """Preserve recorded zeros and negative objectives on a labelled symlog axis."""
    if all(value > 0 for value in values):
        axis.set_yscale("log")
        axis.set(title="Objective energy", ylabel=r"$E(x^n)$")
    else:
        nonzero = [abs(value) for value in values if value != 0]
        threshold = 10 ** math.floor(math.log10(min(nonzero))) if nonzero else 1.0
        axis.set_yscale("symlog", linthresh=threshold)
        axis.set(title=f"Objective energy (symlog; linear below {threshold:g})",
                 ylabel=r"$E(x^n)$")


def plot_experiment(input_dir, output):
    config, rows = load_experiment(input_dir)
    if config["method"] in ("gd", "sgd", "adamw"):
        return plot_comparison([input_dir], output)
    iterations = [int(row["iteration"]) for row in rows]
    objectives = [float(row["objective"]) for row in rows]
    modified_key = "r" if config.get("linearized") else "modified_energy"
    modified = [float(row[modified_key]) for row in rows]
    method = config["method"]
    if method in MESAV_STYLES:
        label, modified_label = MESAV_STYLES[method][0], r"$\log r^n$"
    elif method in ("esav", "naive_esav", "pb_esav", "pb_naive_esav"):
        label = esav_label(config)
        if method in ("pb_esav", "pb_naive_esav") and config.get("relax", False):
            label += " (relaxed)"
            if config["scheme"] == "plan1":
                label += f", eta = {config.get('eta', 1.0):g}"
        modified_label = r"$r^n$" if config.get("linearized") else r"$\log r^n$"
    elif method == "csav":
        label, modified_label = "CSAV", r"$r^n$"
    elif method in ("inverse_sav", "pb_inverse_sav"):
        label = "Inverse SAV" if method == "inverse_sav" else "PB-Inverse SAV"
        modified_label = r"$1/r^n$"
    elif method in ("sav", "pb_sav"):
        label = "SAV" if method == "sav" else "PB-SAV"
        if config.get("relax", False):
            label += " (relaxed)"
        modified_label = r"$(q^n)^2$"
    else:
        raise ValueError(f"Unsupported method in experiment config: {method}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    axes[0].plot(iterations, objectives, color="#3568a8", linewidth=1.8)
    set_objective_scale(axes[0], objectives)
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
    sav_shifts = sorted({config["total_shift"] for config, _ in experiments
                         if config["method"] == "sav"})

    has_auxiliary = any(
        any(key in rows[0] for key in (
            "r_over_exp_E", "r_over_exp_auxiliary_energy", "q_over_Q", "r_over_E_plus_C",
            "r_times_E_plus_C",
        )) for _, rows in experiments
    )
    fig, axes = plt.subplots(1, 2 if has_auxiliary else 1, figsize=(14, 6.2),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    shifts = set()
    has_sav = False
    has_esav = False
    has_new_esav = False
    has_mesav = False
    has_csav = False
    has_inverse = False
    signed_ratio = False
    ratio_min = float("inf")
    ratio_max = 0.0
    ratio_zero = False
    all_objectives = []
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
        elif method in ("inverse_sav", "pb_inverse_sav"):
            label = "Inverse SAV" if method == "inverse_sav" else "PB-Inverse SAV"
            color = "#8b3fb0" if method == "inverse_sav" else "#009e73"
            style, marker = "-", "v" if method == "inverse_sav" else "D"
            ratio_key = "r_times_E_plus_C"
            shifts.add(config["total_shift"])
            has_inverse = True
        elif method == "csav":
            label = "CSAV"
            color, style, marker = "#8b3fb0", "-", "v"
            marker_start = 5
            linewidth = 2.4
            ratio_key = "r_over_E_plus_C"
            shifts.add(config["total_shift"])
            has_csav = True
        elif method in MESAV_STYLES:
            label, color, style, marker = MESAV_STYLES[method]
            ratio_key = "r_over_exp_auxiliary_energy"
            shifts.add(config["total_shift"])
            has_mesav = True
        elif method == "esav":
            label = esav_label(config)
            color, style, marker = "#009e73", "-", "D"
            marker_start = 0
            ratio_key = "r_over_exp_auxiliary_energy"
            shifts.add(config["total_shift"])
            has_new_esav = True
        elif method == "naive_esav":
            label = esav_label(config)
            color = "#b3448c" if config.get("linearized") else "#d55e00"
            style = ":" if config.get("linearized") else "--"
            marker = "v" if config.get("linearized") else "^"
            marker_start = 20 if config.get("linearized") else 30
            if config.get("relax"):
                color, style, marker = "#008b8b", "-", "D"
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
            has_sav = True
            label = "SAV" if method == "sav" else f"rank-{config['components']} PB-SAV"
            color = "#0072b2" if method == "sav" else "#d16a2c"
            style = "-" if config.get("relax", False) else "--"
            marker = "s" if method == "sav" else "P"
            if method == "sav" and len(sav_shifts) > 1:
                label += f" (C={config['total_shift']:g})"
                shift_index = sav_shifts.index(config["total_shift"]) % 4
                color = ("#0072b2", "#d55e00", "#009e73", "#6a3d9a")[shift_index]
                marker = ("s", "^", "d", "x")[shift_index]
                marker_start = 10 + 10 * shift_index
            if config.get("relax", False):
                label = "relaxed " + label
            ratio_key = "q_over_Q"
            shifts.add(config["total_shift"])
        else:
            raise ValueError(f"Unsupported method in experiment config: {method}")
        iterations = [int(row["iteration"]) for row in rows]
        objectives = [float(row["objective"]) for row in rows]
        all_objectives.extend(objectives)
        axes[0].plot(iterations, objectives,
                     color=color, linestyle=style, linewidth=linewidth, label=label,
                     marker=marker, markevery=(marker_start, 40), markersize=6,
                     markerfacecolor="white", markeredgewidth=1.5)
        if ratio_key is not None:
            ratios = [float(row[ratio_key]) for row in rows]
            signed_ratio |= any(value < 0 for value in ratios)
            ratio_zero |= any(value == 0 for value in ratios)
            ratio_min = min(ratio_min, min((value for value in ratios if value > 0),
                                           default=float("inf")))
            ratio_max = max(ratio_max, max(ratios))
            axes[1].plot(iterations, ratios, color=color, linestyle=style, linewidth=linewidth,
                         marker=marker, markevery=(marker_start, 40), markersize=6,
                         markerfacecolor="white", markeredgewidth=1.5)

    set_objective_scale(axes[0], all_objectives)
    ratio_label = (r"$q^n/\sqrt{E(x^n)+C}$ (SAV/PB-SAV)" if has_sav else "")
    if has_esav:
        names = "naiveESAV / PB" if any(
            config["method"] in ("pb_esav", "pb_naive_esav") for config, _ in experiments
        ) else "naiveESAV"
        ratio_label += ("\n" if ratio_label else "") + (
            r"$r^n/\exp(E(x^n))$" + f" ({names})"
        )
    if has_new_esav:
        ratio_label += ("\n" if ratio_label else "") + (
            r"$r^n/[\exp(E(x^n))\sqrt{E(x^n)+C}]$ (ESAV)"
        )
    if has_csav:
        ratio_label += ("\n" if ratio_label else "") + r"$r^n/[E(x^n)+C]$ (CSAV)"
    if has_mesav:
        ratio_label += ("\n" if ratio_label else "") + (
            r"$r^n/[\exp(E(x^n)/(E(x^0)+C))\sqrt{E(x^n)+C}]$ (MESAV/PB)"
        )
    if has_inverse:
        ratio_label += ("\n" if ratio_label else "") + r"$r^n[E(x^n)+C]$ (Inverse SAV/PB)"
    if has_auxiliary:
        axes[1].set(title="Auxiliary-variable ratio", ylabel=ratio_label)
        if signed_ratio:
            axes[1].set_yscale("symlog", linthresh=1e-3)
            axes[1].set_title("Auxiliary-variable ratio (symlog)")
        elif ratio_zero:
            threshold = 10 ** math.floor(math.log10(ratio_min)) if math.isfinite(ratio_min) else 1
            axes[1].set_yscale("symlog", linthresh=threshold)
            axes[1].set_title(f"Auxiliary ratio (symlog; linear below {threshold:g})")
            if ratio_max > 0:
                lower = math.floor(math.log10(threshold))
                upper = math.ceil(math.log10(ratio_max))
                spacing = max(1, math.ceil((upper - lower) / 5))
                # Keep the zero tick separate from the lowest logarithmic ticks.
                powers = range(upper, lower + max(1, spacing // 2) - 1, -spacing)
                axes[1].set_yticks([0.0, *(10.0 ** power for power in powers)])
        elif not ratio_zero and ratio_min < ratio_max * 1e-3:
            axes[1].set_yscale("log")
            axes[1].set_title("Auxiliary-variable ratio (log scale)")
    for axis in axes:
        axis.set_xlabel("Iteration (epoch)")
        axis.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               fontsize=11, frameon=False, handlelength=5, numpoints=2,
               columnspacing=2, labelspacing=0.8)
    shift_note = ""
    if shifts:
        shift_note = "Shift C = " + ", ".join(f"{v:g}" for v in sorted(shifts))
    fig.suptitle(f"{reference['problem'].capitalize()} benchmark | {step_label(reference)}, "
                 f"regularization = {reference['lambda_reg']:g}\n{shift_note}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_modified_energy(input_dirs, output):
    """Compare auxiliary energies with baseline objectives, grouped by learning rate."""
    experiments = [load_experiment(path) for path in input_dirs]
    reference = experiments[0][0]
    shared = ("problem", "dimension", "initial_value", "energy_weights", "device", "dtype",
              "steps", "lambda_reg", "line_search")
    groups = {}
    excluded = []
    for config, rows in experiments:
        if config["problem"] != "quadratic":
            raise ValueError("Modified-energy panels require quadratic experiments")
        if any(config.get(key) != reference.get(key) for key in shared):
            raise ValueError("Modified-energy panels require matching problem settings")
        groups.setdefault(config["lr"], [])
        if "modified_energy" not in rows[0] and config["method"] not in ("gd", "sgd", "adamw"):
            label = esav_label(config) if config["method"] == "naive_esav" else {
                "gd": "GD", "sgd": "SGD", "adamw": "AdamW",
            }[config["method"]]
            if label not in excluded:
                excluded.append(label)
        else:
            groups[config["lr"]].append((config, rows))
    if any(not group for group in groups.values()):
        raise ValueError("Each learning rate needs at least one energy curve")

    fig, axes = plt.subplots(1, len(groups), figsize=(7 * len(groups), 6.2),
                             layout="constrained", squeeze=False)
    for axis, (lr, group) in zip(axes[0], groups.items()):
        order = {"gd": 0, "sgd": 0, "adamw": 1, "csav": 2, "sav": 2, "pb_sav": 3,
                 "esav": 4, "naive_esav": 5, "pb_naive_esav": 6, "pb_esav": 7,
                 "inverse_sav": 8, "pb_inverse_sav": 9}
        order.update({method: 10 + index for index, method in enumerate(MESAV_STYLES)})
        group.sort(key=lambda item: order[item[0]["method"]])
        for config, rows in group:
            method = config["method"]
            energy_key = "modified_energy"
            if method in ("gd", "sgd", "adamw"):
                label = {"gd": "GD", "sgd": "SGD", "adamw": "AdamW"}[method]
                label += r": $E(x^n)$"
                color, marker, start = ("#a6761d", "P", 30) if method == "adamw" else (
                    "#222222", "o", 10,
                )
                style = "-"
                energy_key = "objective"
            elif method == "csav":
                label = r"CSAV: $r^n$"
                color, style, marker, start = "#8b3fb0", "-", "v", 5
            elif method in ("inverse_sav", "pb_inverse_sav"):
                label = "Inverse SAV" if method == "inverse_sav" else "PB-Inverse SAV"
                label += r": $1/r^n$"
                color = "#8b3fb0" if method == "inverse_sav" else "#009e73"
                style, marker, start = "-", "D", 5
            elif method in ("sav", "pb_sav"):
                label = "SAV" if method == "sav" else "PB-SAV"
                label += r": $(q^n)^2$"
                color, style, marker, start = "#0072b2", "--", "s", 15
            elif method in MESAV_STYLES:
                label, color, style, marker = MESAV_STYLES[method]
                label += r": $\ln r^n$"
                start = 0
            elif method == "esav":
                label = r"ESAV: $\ln r^n$"
                color, style, marker, start = "#009e73", "-", "D", 0
            elif method == "naive_esav":
                label = esav_label(config) + r": $\ln r^n$"
                color, style, marker, start = "#d55e00", "--", "^", 30
                if config.get("relax"):
                    color, style, marker = "#008b8b", "-", "D"
            elif method in ("pb_esav", "pb_naive_esav"):
                label = esav_label(config) + r": $\ln r^n$"
                color, style, marker, start = {
                    "plan1": ("#009e73", "-", "d", 10),
                    "plan2": ("#6a3d9a", "--", "x", 30),
                }[config["scheme"]]
            else:
                raise ValueError(f"Unsupported modified-energy method: {method}")
            axis.plot(
                [int(row["iteration"]) for row in rows],
                [float(row[energy_key]) for row in rows],
                label=label, color=color, linestyle=style, linewidth=1.9,
                marker=marker, markevery=(start, 40), markersize=6,
                markerfacecolor="white", markeredgewidth=1.5,
            )
        # Preserve negative logarithmic energies and resolve values near zero.
        axis.set_yscale("symlog", linthresh=1)
        axis.set(title=step_label(group[0][0]), xlabel="Iteration (epoch)",
                 ylabel="Energy (symlog; linear for |energy| <= 1)")
        axis.grid(True, alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3,
               fontsize=11, frameon=False, handlelength=5, numpoints=2,
               columnspacing=2, labelspacing=0.8)
    title = (f"Quadratic benchmark | {reference['steps']} updates | "
             "Modified energies and baseline objectives")
    if excluded:
        title += "\nNo recorded modified energy: " + ", ".join(excluded)
    fig.suptitle(title)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_training_curves(input_dirs, output):
    """Compare objectives, including historical test MSE only when every run records it."""
    experiments = [load_experiment(path) for path in input_dirs]
    styles = {
        ("gd", None): ("GD", "#333333", "-.", None),
        ("sgd", None): ("SGD", "#333333", "-.", None),
        ("adamw", None): ("AdamW", "#007c91", "-", None),
        ("csav", None): ("CSAV", "#8b3fb0", "-", "v"),
        ("inverse_sav", None): ("Inverse SAV", "#8b3fb0", "-", "v"),
        ("pb_inverse_sav", None): ("PB-Inverse SAV", "#009e73", "-", "D"),
        ("esav", None): ("ESAV", "#009e73", "-", "D"),
        ("naive_esav", None): ("naiveESAV", "#d89b00", "-", None),
        ("pb_sav", None): ("PB-SAV", "#3568a8", "-", None),
        ("pb_esav", "plan1"): ("PB-ESAV Plan 1", "#d16a2c", "--", None),
        ("pb_esav", "plan2"): ("PB-ESAV Plan 2", "#218c60", ":", "o"),
        ("pb_naive_esav", "plan1"): ("PB-naiveESAV Plan 1", "#8953b1", "--", None),
        ("pb_naive_esav", "plan2"): ("PB-naiveESAV Plan 2", "#8c564b", ":", "s"),
    }
    styles.update({(method, None): values for method, values in MESAV_STYLES.items()})
    keys = ("objective", "test_mse") if all(
        "test_mse" in rows[0] for _, rows in experiments
    ) else ("objective",)
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 4.8), layout="constrained", squeeze=False)
    axes = axes[0]
    for config, rows in experiments:
        style_method = config["method"]
        if style_method == "pb_esav" and esav_label(config).startswith("PB-naiveESAV"):
            style_method = "pb_naive_esav"
        label, color, style, marker = styles[(style_method, config.get("scheme"))]
        if config["method"] in ("esav", "naive_esav", "pb_esav", "pb_naive_esav"):
            label = esav_label(config)
            if config.get("linearized"):
                color, style = "#cc503e", ":"
            elif config["method"] == "naive_esav" and config.get("relax"):
                color, style, marker = "#008b8b", "--", "D"
        label += f" ($M={config['components']}$)"
        iterations = [int(row["iteration"]) for row in rows]
        for axis, key in zip(axes, keys):
            axis.plot(
                iterations, [float(row[key]) for row in rows], label=label,
                color=color, linestyle=style, linewidth=1.8, marker=marker,
                markevery=60, markersize=5, markerfacecolor="none",
            )
    axes[0].set(title="Objective loss", ylabel=r"$E(\theta)$")
    if len(keys) > 1:
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--training", action="store_true",
                      help="Plot training objective and test MSE for nonlinear regression")
    mode.add_argument("--modified-energy", action="store_true",
                      help="Plot quadratic modified energies and baseline objective references")
    args = parser.parse_args()
    if args.modified_energy:
        plot_modified_energy(args.input, args.output)
    elif args.training:
        plot_training_curves(args.input, args.output)
    elif len(args.input) == 1:
        plot_experiment(args.input[0], args.output)
    else:
        plot_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
