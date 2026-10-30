"""Plot saved experiments without running an optimizer."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "TMP/matplotlib"))

import matplotlib

from problem.config import WANDB
from tracking import _sdk, read_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterMathtext, LogLocator, MaxNLocator, NullLocator

ESAV_STYLES = {
    "esav": ("ESAV", "#009e73", "-", "D"),
    "pb_esav": ("PB-ESAV", "#218c60", "-", "s"),
    "relax_esav": ("Relax-ESAV", "#b3448c", "--", "D"),
    "relax_pb_esav": ("Relax-PB-ESAV", "#663b94", "--", "s"),
}

MESAV_STYLES = {
    "mesav": ("MESAV", "#60a5d1", "-", "o"),
    "pb_mesav": ("PB-MESAV", "#1765ad", "-", "s"),
    "relax_mesav": ("Relax-MESAV", "#9b6cc2", "--", "o"),
    "relax_pb_mesav": ("Relax-PB-MESAV", "#663b94", "--", "s"),
}

BESAV_STYLES = {
    "besav": ("BESAV", "#006d9c", "-", "D"),
    "pb_besav": ("PB-BESAV", "#006d9c", "-", "s"),
    "relax_besav": ("Relax-BESAV", "#ad477b", "--", "D"),
    "relax_pb_besav": ("Relax-PB-BESAV", "#ad477b", "--", "s"),
}


def load_experiment(input_dir):
    input_dir = Path(input_dir)
    config = json.loads((input_dir / "config.json").read_text(encoding="utf-8"))
    config = read_config(config)
    with (input_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The experiment history is empty")
    # Interpret archived names and diagnostic columns without rewriting raw files.
    aliases = {
        "bounded_exp": "besav", "pb_bounded_exp": "pb_besav",
        "relax_pb_bounded_exp": "relax_pb_besav",
    }
    config["method"] = aliases.get(config["method"], config["method"])
    if config.get("relax") and config["method"] in (
        "esav", "pb_esav", "besav", "pb_besav", "mesav", "pb_mesav",
    ):
        config["method"] = "relax_" + config["method"]
    if config["method"] in BESAV_STYLES:
        for row in rows:
            if "r_over_exp_auxiliary_energy" not in row and "auxiliary_ratio" in row:
                row["r_over_exp_auxiliary_energy"] = row["auxiliary_ratio"]
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
    """Use the current ESAV names; Plan numbers belong only to PB-naiveESAV."""
    if config["method"] in ESAV_STYLES:
        return ESAV_STYLES[config["method"]][0]
    if config["method"] == "naive_esav":
        if config.get("linearized"):
            return "naiveESAV (linearized r)"
        return "naiveESAV (relaxed)" if config.get("relax") else "naiveESAV"
    return f"PB-naiveESAV Plan {config['scheme'][-1]}"


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


def load_wandb_csv(input_file):
    """Read objective series from a W&B chart export without resampling."""
    with Path(input_file).open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        columns = [field for field in fields if field.endswith(" - objective")]
        if "iteration" not in fields or not columns:
            raise ValueError("W&B CSV requires iteration and '<run> - objective' columns")
        if len(fields) != len(set(fields)):
            raise ValueError("W&B CSV contains duplicate column names")
        rows = list(reader)
    if not rows:
        raise ValueError("The W&B CSV is empty")

    def numbers(column):
        values = []
        for line, row in enumerate(rows, start=2):
            try:
                value = float(row[column])
            except (TypeError, ValueError) as error:
                raise ValueError(f"CSV row {line}, {column}: missing or nonnumeric value") from error
            if not math.isfinite(value):
                raise ValueError(f"CSV row {line}, {column}: nonfinite value {row[column]}")
            values.append(value)
        return values

    iterations = numbers("iteration")
    if any(value < 0 or not value.is_integer() for value in iterations):
        raise ValueError("CSV iterations must be nonnegative integers")
    if any(right <= left for left, right in zip(iterations, iterations[1:])):
        raise ValueError("CSV iterations must be strictly increasing")
    series = [(column.removesuffix(" - objective"), numbers(column)) for column in columns]
    return iterations, series


def plot_wandb_csv(input_file, output, *, labels=None, title=None):
    """Plot exported objectives with labelled log ticks and unmodified samples."""
    iterations, series = load_wandb_csv(input_file)
    curves = [(name, iterations, values) for name, values in series]
    return plot_objective_series(curves, output, labels=labels, title=title)


def plot_objective_series(series, output, *, labels=None, title=None, statuses=None):
    """Plot complete recorded curves, retaining each run's actual stopping iteration."""
    if labels is not None and len(labels) != len(series):
        raise ValueError(f"Expected {len(series)} labels in CSV objective-column order")
    labels = labels if labels is not None else [name for name, _, _ in series]
    statuses = statuses if statuses is not None else ["completed"] * len(series)
    if len(statuses) != len(series):
        raise ValueError("Each objective series needs one status")
    styles = {style[0]: style[1:] for style in ESAV_STYLES.values()}
    styles.update({
        "SAV": ("#0072b2", "-", "o"),
        "Relax-SAV": ("#d55e00", "--", "^"),
        "PB-SAV": ("#0072b2", "-", "s"),
        "Relax-PB-SAV": ("#d55e00", "--", "^"),
        "SGD": ("#555555", ":", "x"),
        "AdamW": ("#8064c6", "-", ""),
    })
    fig, axis = plt.subplots(figsize=(9.6, 5.8), layout="constrained")
    all_values = []
    for index, ((name, iterations, values), label, status) in enumerate(
        zip(series, labels, statuses)
    ):
        marker_step = max(1, len(iterations) // 12)
        color, line, marker = styles.get(label, (f"C{index % 10}", "-", ""))
        shown_label = label if status == "completed" else f"{label} ({status})"
        axis.plot(
            iterations, values, label=shown_label, color=color, linestyle=line,
            linewidth=1.6, marker=marker, markersize=5, markerfacecolor="none",
            markevery=(index * 3 % marker_step, marker_step),
        )
        if status != "completed":
            axis.plot(iterations[-1], values[-1], marker="x", color=color,
                      markersize=9, markeredgewidth=2)
        all_values.extend(values)
        print(f"{name}: {len(values)} points, last iteration={iterations[-1]:g}, "
              f"objective={values[-1]:.9e}")
    set_objective_scale(axis, all_values)
    if axis.get_yscale() == "log":
        axis.yaxis.set_major_locator(LogLocator(base=10, numticks=9))
        axis.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_title(title or "Objective")
    elif title:
        axis.set_title(f"{title}\n{axis.get_title()}")
    axis.set(xlabel="Iteration", ylabel=r"Objective $E(\theta^n)$")
    axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    axis.tick_params(labelsize=11)
    axis.margins(x=0, y=0.04)
    axis.grid(True, which="major", color="#d8d8d8", linewidth=0.6, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14),
                ncol=3 if len(series) >= 6 else 2,
                frameon=False, fontsize=10, handlelength=3)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_recorded_results(experiments, output, *, auxiliary=False):
    """Plot the current objective and optional total auxiliary values as recorded."""
    fig, axes = plt.subplots(1, 2 if auxiliary else 1, figsize=(10, 4.2),
                             layout="constrained", squeeze=False)
    axes = axes[0]
    objectives = []
    for config, rows in experiments:
        iterations = [int(row["iteration"]) for row in rows]
        values = [float(row["objective"]) for row in rows]
        objectives.extend(values)
        label = config["method"]
        if config.get("relax") and not label.startswith("relax_"):
            label += " (relaxed)"
        if config.get("components", 1) > 1:
            label += f" M={config['components']}"
        label += " | Armijo" if config.get("line_search") else f" | lr={config['lr']:g}"
        axes[0].plot(iterations, values, label=label)
        if auxiliary:
            for key, style in (("r", "-"), ("r_current", "--")):
                axes[1].plot(iterations, [float(row[key]) for row in rows], style, label=key)
            axes[1].set(title="Auxiliary variable", ylabel="r")
    set_objective_scale(axes[0], objectives)
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(True, alpha=0.2)
        axis.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    return output


def plot_experiment(input_dir, output):
    config, rows = load_experiment(input_dir)
    if "r_current" in rows[0]:
        return plot_recorded_results([(config, rows)], output, auxiliary=True)
    if config["problem"] == "darcy_deeponet" or config["method"] in ("gd", "sgd", "adamw"):
        return plot_comparison([input_dir], output)
    iterations = [int(row["iteration"]) for row in rows]
    objectives = [float(row["objective"]) for row in rows]
    modified_key = "r" if config.get("linearized") else "modified_energy"
    modified = [float(row[modified_key]) for row in rows]
    method = config["method"]
    if method in BESAV_STYLES:
        label, modified_label = BESAV_STYLES[method][0], r"$\log r^n$"
    elif method in MESAV_STYLES:
        label, modified_label = MESAV_STYLES[method][0], r"$\log r^n$"
    elif method in ESAV_STYLES:
        label, modified_label = esav_label(config), r"$4E^0\log r^n$"
    elif method in ("naive_esav", "pb_naive_esav"):
        label = esav_label(config)
        if method == "pb_naive_esav" and config.get("relax", False):
            label += " (relaxed)"
            if config["scheme"] == "plan1":
                label += f", eta = {config.get('eta', 1.0):g}"
        modified_label = r"$r^n$" if config.get("linearized") else r"$\log r^n$"
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
    if any(config["problem"] == "darcy_deeponet" for config, _ in experiments):
        return plot_darcy_curves(experiments, output)
    if all("lr" in rows[0] for _, rows in experiments):
        return plot_recorded_results(experiments, output)
    if any(config["problem"] == "nonlinear_regression" for config, _ in experiments):
        return plot_training_curves(input_dirs, output)
    reference = experiments[0][0]
    shared = ("problem", "dimension", "initial_value", "energy_weights", "device", "dtype",
              "steps", "lambda_reg")
    for config, _ in experiments[1:]:
        if any(config.get(key) != reference.get(key) for key in shared):
            raise ValueError("Comparison requires matching problem and experiment settings")
    sav_shifts = sorted({config["total_shift"] for config, _ in experiments
                         if config["method"] == "sav"})

    has_auxiliary = any(
        any(key in rows[0] for key in (
            "r_over_exp_E", "r_over_exp_auxiliary_energy", "q_over_Q",
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
    has_besav = False
    has_mesav = False
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
        elif method in BESAV_STYLES:
            label, color, style, marker = BESAV_STYLES[method]
            ratio_key = "r_over_exp_auxiliary_energy"
            has_besav = True
        elif method in MESAV_STYLES:
            label, color, style, marker = MESAV_STYLES[method]
            ratio_key = "r_over_exp_auxiliary_energy"
            shifts.add(config["total_shift"])
            has_mesav = True
        elif method in ESAV_STYLES:
            label, color, style, marker = ESAV_STYLES[method]
            marker_start = 0
            ratio_key = "r_over_exp_auxiliary_energy"
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
        elif method == "pb_naive_esav":
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
        label += " | Armijo" if config.get("line_search") else f" | fixed lr={config['lr']:g}"
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
            config["method"] == "pb_naive_esav" for config, _ in experiments
        ) else "naiveESAV"
        ratio_label += ("\n" if ratio_label else "") + (
            r"$r^n/\exp(E(x^n))$" + f" ({names})"
        )
    if has_new_esav:
        ratio_label += ("\n" if ratio_label else "") + (
            r"$r^n/\exp(E(x^n)/(4E^0))$ (ESAV/PB)"
        )
    if has_besav:
        ratio_label += ("\n" if ratio_label else "") + (
            r"$r^n/\exp(E(x^n)/(E(x^n)+E^0))$ (BESAV/PB)"
        )
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
    policies = "; ".join(dict.fromkeys(step_label(config) for config, _ in experiments))
    fig.suptitle(f"{reference['problem'].capitalize()} benchmark | {policies}\n"
                 f"regularization = {reference['lambda_reg']:g} | {shift_note}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_modified_energy(input_dirs, output):
    """Compare auxiliary energies with baseline objectives, grouped by learning rate."""
    experiments = [load_experiment(path) for path in input_dirs]
    if any("r_current" in rows[0] for _, rows in experiments):
        raise ValueError("New records contain r/r_current; use the default plot, not modified energy")
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
        order = {"gd": 0, "sgd": 0, "adamw": 1, "sav": 2, "pb_sav": 3,
                 "esav": 4, "naive_esav": 5, "pb_naive_esav": 6, "pb_esav": 7,
                 "inverse_sav": 8, "pb_inverse_sav": 9}
        order.update({method: 10 + index for index, method in enumerate(MESAV_STYLES)})
        order.update({method: 4 + index for index, method in enumerate(ESAV_STYLES)})
        order.update({method: 14 + index for index, method in enumerate(BESAV_STYLES)})
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
            elif method in ("inverse_sav", "pb_inverse_sav"):
                label = "Inverse SAV" if method == "inverse_sav" else "PB-Inverse SAV"
                label += r": $1/r^n$"
                color = "#8b3fb0" if method == "inverse_sav" else "#009e73"
                style, marker, start = "-", "D", 5
            elif method in ("sav", "pb_sav"):
                label = "SAV" if method == "sav" else "PB-SAV"
                label += r": $(q^n)^2$"
                color, style, marker, start = "#0072b2", "--", "s", 15
            elif method in BESAV_STYLES:
                label, color, style, marker = BESAV_STYLES[method]
                label += r": $\ln r^n$"
                start = 0
            elif method in MESAV_STYLES:
                label, color, style, marker = MESAV_STYLES[method]
                label += r": $\ln r^n$"
                start = 0
            elif method in ESAV_STYLES:
                label, color, style, marker = ESAV_STYLES[method]
                label += r": $4E^0\ln r^n$"
                start = 0
            elif method == "naive_esav":
                label = esav_label(config) + r": $\ln r^n$"
                color, style, marker, start = "#d55e00", "--", "^", 30
                if config.get("relax"):
                    color, style, marker = "#008b8b", "-", "D"
            elif method == "pb_naive_esav":
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


def plot_darcy_curves(experiments, output):
    """Plot saved Darcy data losses and mean test relative L2, without rerunning models."""
    reference = experiments[0][0]
    for config, _ in experiments:
        if config["problem"] != "darcy_deeponet" or config["method"] not in (
                "sgd", "pb_sav_momentum", "pb_esav_momentum"):
            raise ValueError("Darcy plots require SGD or PB momentum records")
        if any(config.get(key) != reference.get(key) for key in (
                "data_config", "init_seed", "device", "dtype", "dimension", "lambda_reg",
                "conv_channels", "conv_kernels", "conv_strides", "conv_paddings",
                "pool_size", "latent_dim", "trunk_hidden_layers")):
            raise ValueError("Darcy comparison requires matching data and model settings")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for config, rows in experiments:
        momentum = config["optimizer_settings"]["momentum"]
        if config["method"] in ("pb_sav_momentum", "pb_esav_momentum"):
            method_label = "PB-ESAV" if config["method"] == "pb_esav_momentum" else "PB-SAV"
            label = f"{method_label} + momentum ({momentum:g})"
            if config["optimizer_settings"].get("mobility") == "amsgrad":
                label += " + adaptive"
            if config.get("relax"):
                label += " + Relax"
        else:
            label = f"SGD + momentum ({momentum:g})" if momentum else "SGD"
        label += f" | lr={config['lr']:g}"
        iterations = [int(row["iteration"]) for row in rows]
        for axis, key in zip(axes, ("data_loss", "test_relative_l2")):
            axis.plot(iterations, [float(row[key]) for row in rows], label=label)
    axes[0].set(title="Training data loss", ylabel=r"$E_{\mathrm{data}}$")
    axes[1].set(title="Test relative L2", ylabel="Mean per-field relative L2")
    for axis, key in zip(axes, ("data_loss", "test_relative_l2")):
        if all(float(row[key]) > 0 for _, rows in experiments for row in rows):
            axis.set_yscale("log")
        axis.set_xlabel("Parameter updates")
        axis.grid(True, alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False)
    fig.suptitle(f"Darcy DeepONet | seed {reference['init_seed']}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"Figure: {output.resolve()}")
    return output


def plot_training_curves(input_dirs, output):
    """Compare objectives, including historical test MSE only when every run records it."""
    experiments = [load_experiment(path) for path in input_dirs]
    if any(config["problem"] == "darcy_deeponet" for config, _ in experiments):
        return plot_darcy_curves(experiments, output)
    if all("lr" in rows[0] for _, rows in experiments):
        return plot_recorded_results(experiments, output)
    styles = {
        ("gd", None): ("GD", "#333333", "-.", None),
        ("sgd", None): ("SGD", "#333333", "-.", None),
        ("adamw", None): ("AdamW", "#007c91", "-", None),
        ("inverse_sav", None): ("Inverse SAV", "#8b3fb0", "-", "v"),
        ("pb_inverse_sav", None): ("PB-Inverse SAV", "#009e73", "-", "D"),
        ("naive_esav", None): ("naiveESAV", "#d89b00", "-", None),
        ("pb_sav", None): ("PB-SAV", "#3568a8", "-", None),
        ("pb_naive_esav", "plan1"): ("PB-naiveESAV Plan 1", "#8953b1", "--", None),
        ("pb_naive_esav", "plan2"): ("PB-naiveESAV Plan 2", "#8c564b", ":", "s"),
    }
    styles.update({(method, None): values for method, values in MESAV_STYLES.items()})
    styles.update({(method, None): values for method, values in ESAV_STYLES.items()})
    styles.update({(method, None): values for method, values in BESAV_STYLES.items()})
    keys = ("objective", "test_mse") if all(
        "test_mse" in rows[0] for _, rows in experiments
    ) else ("objective",)
    fig, axes = plt.subplots(1, len(keys), figsize=(11, 4.8), layout="constrained", squeeze=False)
    axes = axes[0]
    for config, rows in experiments:
        style_method = config["method"]
        label, color, style, marker = styles[(style_method, config.get("scheme"))]
        if config["method"] in ("naive_esav", "pb_naive_esav"):
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


ROOT = Path(__file__).resolve().parent
PROJECT = f"{WANDB['entity']}/{WANDB['project']}"
GROUPS = ROOT / "doc/experiments/esav_plot_groups.json"
CACHE = ROOT / "TMP/wandb_exports"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Replace a local manifest only after its complete content has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".partial")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                       encoding="utf-8")
    pending.replace(path)


def validate_records(directory):
    """Check accepted iterations and summaries without repairing numerical records."""
    directory = Path(directory)
    config = read_json(directory / "config.json")
    summary = read_json(directory / "summary.json")
    with (directory / "history.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if not {"iteration", "objective"}.issubset(fields):
            raise ValueError(f"{directory}: missing iteration/objective columns")
        if len(fields) != len(set(fields)):
            raise ValueError(f"{directory}: duplicate history columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{directory}: empty history")
    for index, row in enumerate(rows):
        for key, text in row.items():
            try:
                value = float(text)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{directory}: row {index + 2}, invalid {key}") from error
            if not math.isfinite(value):
                raise ValueError(f"{directory}: row {index + 2}, nonfinite {key}={text}")
        if float(row["iteration"]) != index:
            raise ValueError(f"{directory}: iteration {index} missing, duplicated or out of order")
    status = summary.get("status")
    if status not in ("completed", "failed", "interrupted"):
        raise ValueError(f"{directory}: unknown status {status!r}")
    planned = config["experiment"]["steps"]
    actual = len(rows) - 1
    if actual > planned or (status == "completed" and actual != planned):
        raise ValueError(f"{directory}: {status}, recorded {actual}/{planned} updates")
    if status != "completed":
        failure = read_json(directory / "failure.json")
        if not failure.get("error_type"):
            raise ValueError(f"{directory}: failed run has no recorded failure reason")
    for key, value in rows[-1].items():
        if key not in summary or not math.isclose(
            float(value), float(summary[key]), rel_tol=1e-12, abs_tol=0
        ):
            raise ValueError(f"{directory}: final {key} differs from summary")
    objectives = [float(row["objective"]) for row in rows]
    if not math.isclose(min(objectives), float(summary["best_objective"]),
                        rel_tol=1e-12, abs_tol=0):
        raise ValueError(f"{directory}: best_objective differs from history")
    checks = {"status": status, "rows": len(rows), "actual_updates": actual,
              "planned_updates": planned, "final_objective": objectives[-1],
              "best_objective": min(objectives)}
    return config, rows, checks


class RecordReader:
    """Use immutable artifact versions; a valid cache needs no network connection."""

    def __init__(self, cache, offline=False):
        self.cache = Path(cache)
        self.offline = offline
        self.api = None
        self.sdk = None

    def remote(self, operation):
        if self.offline:
            raise FileNotFoundError("Required records are not cached; retry without --offline")
        if self.api is None:
            self.sdk = _sdk(ROOT)
            self.api = self.sdk.Api(timeout=WANDB["init_timeout"])
        for attempt in range(3):
            try:
                return operation(self.api)
            except (self.sdk.CommError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                print(f"Network operation failed; retry {attempt + 1}/2", flush=True)
                time.sleep(attempt + 1)

    def fetch(self, run_id, version=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError(f"Invalid run ID: {run_id!r}")
        if version is not None and not re.fullmatch(r"v\d+", version):
            raise ValueError("records_version must be an immutable version such as v0")
        directory = self.cache / run_id
        metadata_path = directory / "run.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            if metadata["project"] != PROJECT or metadata["run_id"] != run_id:
                raise ValueError(f"{directory}: cached run identity differs")
            if version and not metadata["records_artifact"].endswith(f":{version}"):
                raise ValueError(f"{run_id}: cached version differs; use a separate --cache")
        else:
            run = self.remote(lambda api: api.run(f"{PROJECT}/{run_id}"))
            if run.state not in ("finished", "failed", "crashed", "killed"):
                raise ValueError(f"{run_id}: run is {run.state}, not a closed experiment")
            artifacts = self.remote(lambda api: list(run.logged_artifacts()))
            records = [item for item in artifacts
                       if item.type == "experiment-record"
                       and item.name.split(":")[0] == f"records-{run_id}"
                       and (version is None or item.version == version)]
            if len(records) != 1:
                raise ValueError(f"{run_id}: expected one records artifact, found {len(records)}")
            artifact = records[0]
            entries = self.remote(lambda api: sorted(artifact.manifest.entries))
            required = {"config.json", "history.csv", "summary.json"}
            if not required.issubset(entries):
                raise ValueError(f"{run_id}: records artifact lacks {sorted(required - set(entries))}")
            metadata = {
                "project": PROJECT, "run_id": run_id, "name": run.name, "url": run.url,
                "state": run.state, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "records_artifact": f"{PROJECT}/records-{run_id}:{artifact.version}",
                "source_artifacts": [f"{PROJECT}/source-{run_id}:{item.version}"
                                     for item in artifacts if item.type == "experiment-source"
                                     and item.name.split(":")[0] == f"source-{run_id}"],
                "files": sorted(required | ({"failure.json"} & set(entries))),
            }
            write_json(metadata_path, metadata)
        missing = [name for name in metadata["files"] if not (directory / name).exists()]
        if missing:
            artifact = self.remote(lambda api: api.artifact(metadata["records_artifact"]))
            for name in missing:
                self.remote(lambda api, name=name: artifact.get_entry(name).download(
                    root=str(directory),
                ))
        config, rows, checks = validate_records(directory)
        print(f"{run_id}: {'downloaded' if missing else 'cached'}, "
              f"{checks['rows']} rows, {checks['status']}", flush=True)
        return metadata, config, rows, checks


def export_group(group_id, group, reader, output_root):
    """Export one explicitly selected comparison without combining different PB sizes."""

    runs = group["runs"]
    if len({item["id"] for item in runs}) != len(runs):
        raise ValueError(f"{group_id}: duplicate run IDs")
    loaded = [reader.fetch(item["id"], item.get("records_version")) for item in runs]
    reference = loaded[0][1]
    components = set()
    for _, config, _, _ in loaded:
        if (config["problem"] != reference["problem"]
                or config["experiment"] != reference["experiment"]):
            raise ValueError(f"{group_id}: common problem/data/initialization/budget differ")
        if "pb" in config.get("algorithm", {}):
            components.add(config["algorithm"]["pb"]["M"])
    if len(components) > 1:
        raise ValueError(f"{group_id}: multiple PB component counts {sorted(components)}")

    output = output_root / group_id.lower()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "objectives.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["run_id", "label", "iteration", "objective"])
        for item, (_, _, rows, _) in zip(runs, loaded):
            writer.writerows([item["id"], item["label"], row["iteration"], row["objective"]]
                             for row in rows)
    summaries = []
    for item, (metadata, config, _, checks) in zip(runs, loaded):
        summaries.append({"run_id": item["id"], "label": item["label"], **checks,
                          "method": config["method"],
                          "control": config["control"]["strategy"],
                          "lr": config["control"]["lr"], "url": metadata["url"]})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    curves = [(item["id"], [int(row["iteration"]) for row in rows],
               [float(row["objective"]) for row in rows])
              for item, (_, _, rows, _) in zip(runs, loaded)]
    for extension in ("png", "svg"):
        plot_objective_series(curves, output / f"objective_log.{extension}",
                              labels=[item["label"] for item in runs], title=group["title"],
                              statuses=[checks["status"] for _, _, _, checks in loaded])
    write_json(output / "manifest.json", {
        "project": PROJECT, "group": group_id, "title": group["title"],
        "runs": [{"label": item["label"], **metadata, "validation": checks}
                 for item, (metadata, _, _, checks) in zip(runs, loaded)],
    })
    print(f"{group_id}: validated and exported to {output}", flush=True)


def export_selected_groups(args, parser):
    for path in (args.cache, args.output):
        if not path.resolve().is_relative_to(ROOT):
            parser.error("Cache and output must be inside this project")
    os.environ["MPLCONFIGDIR"] = str(ROOT / "TMP/matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(ROOT / "TMP/cache")
    groups = read_json(args.groups)
    selected = list(groups) if args.all else args.group
    if any(key not in groups for key in selected):
        parser.error(f"Unknown group; available: {', '.join(groups)}")
    reader = RecordReader(args.cache, args.offline)
    status_path = args.output / "status.json"
    status = {"groups_file": str(args.groups.resolve()), "groups": {}}
    for group_id in selected:
        status["groups"][group_id] = {"status": "running"}
        write_json(status_path, status)
        try:
            export_group(group_id, groups[group_id], reader, args.output)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            status["groups"][group_id] = {"status": "failed", "error": message}
            print(f"{group_id}: FAILED: {message}", flush=True)
        else:
            status["groups"][group_id] = {"status": "completed"}
        write_json(status_path, status)
    if any(item["status"] != "completed" for item in status["groups"].values()):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+",
                        help="Experiment directories, or one chart CSV with --wandb-csv")
    parser.add_argument("--output", type=Path, help="Output figure path or group export directory")
    parser.add_argument("--labels", nargs="+", help="W&B CSV legend labels in column order")
    parser.add_argument("--title", help="W&B CSV figure title")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--wandb-csv", action="store_true",
                      help="Plot objective columns from one exported W&B chart CSV")
    mode.add_argument("--training", action="store_true",
                      help="Plot training objective and test MSE for nonlinear regression")
    mode.add_argument("--modified-energy", action="store_true",
                      help="Plot quadratic modified energies and baseline objective references")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--group", nargs="+", help="Download and plot comparison groups")
    selection.add_argument("--all", action="store_true", help="Download and plot all groups")
    parser.add_argument("--groups", type=Path, default=GROUPS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--offline", action="store_true", help="Use cached W&B records")
    args = parser.parse_args()
    if args.group or args.all:
        if args.input or args.wandb_csv or args.training or args.modified_energy or args.labels or args.title:
            parser.error("Group export cannot be combined with local plotting options")
        args.output = args.output or ROOT / "TMP/experiment_exports"
        export_selected_groups(args, parser)
        return
    if not args.input or not args.output:
        parser.error("Local plotting requires --input and --output")
    if (args.labels is not None or args.title is not None) and not args.wandb_csv:
        parser.error("--labels and --title require --wandb-csv")
    if args.wandb_csv:
        if len(args.input) != 1:
            parser.error("--wandb-csv requires exactly one CSV file")
        plot_wandb_csv(args.input[0], args.output, labels=args.labels, title=args.title)
    elif args.modified_energy:
        plot_modified_energy(args.input, args.output)
    elif args.training:
        plot_training_curves(args.input, args.output)
    elif len(args.input) == 1:
        plot_experiment(args.input[0], args.output)
    else:
        plot_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
