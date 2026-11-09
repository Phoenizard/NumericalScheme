"""Plot saved experiments (config.json + history.csv) without running an optimizer."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / "TMP/matplotlib"))

import matplotlib

from tracking import read_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_experiment(input_dir):
    """Return the resolved config and the history rows of one saved experiment."""
    input_dir = Path(input_dir)
    config = read_config(json.loads((input_dir / "config.json").read_text(encoding="utf-8")))
    with (input_dir / "history.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The experiment history is empty")
    return config, rows


def label(config):
    """Method name with Relax and momentum markers."""
    name = config["method"].replace("_", "-").upper()
    return ("Relax-" if config.get("relax") else "") + name


def plot_comparison(input_dirs, output):
    """Plot the training objective of each run; add a validation panel when recorded."""
    experiments = [load_experiment(path) for path in input_dirs]
    validation = next((key for key in ("validation_relative_l2", "test_relative_l2", "test_mse")
                       if any(key in rows[0] and rows[0][key] != "" for _, rows in experiments)),
                      None)
    figure, axes = plt.subplots(1, 2 if validation else 1, figsize=(11 if validation else 6, 4.5),
                                squeeze=False)
    for config, rows in experiments:
        steps = [int(row["iteration"]) for row in rows]
        axes[0, 0].plot(steps, [float(row["objective"]) for row in rows], label=label(config))
        if validation:
            pairs = [(int(row["iteration"]), float(row[validation]))
                     for row in rows if row.get(validation, "") != ""]
            if pairs:
                axes[0, 1].plot(*zip(*pairs), label=label(config))
    axes[0, 0].set(yscale="log", xlabel="Iteration", ylabel=r"Objective $E(\theta^n)$")
    if validation:
        axes[0, 1].set(yscale="log", xlabel="Iteration", ylabel=validation)
    axes[0, 0].legend()
    figure.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Experiment directories containing config.json and history.csv")
    parser.add_argument("--output", type=Path, required=True, help="Output figure path")
    args = parser.parse_args()
    plot_comparison(args.input, args.output)


if __name__ == "__main__":
    main()
