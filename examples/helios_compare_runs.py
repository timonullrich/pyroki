"""Offline comparison of saved Helios IK runs; needs only numpy and matplotlib.

Each approach is a folder in helios_runs (``config_<name>`` or ``<name>``) whose
run folders hold the controller's rollout.csv and summary.json. Every run found
below an approach folder is included, so newly collected runs appear automatically.

Per-run averages are computed first; bars show their mean across runs, with
error bars of one standard deviation across runs (none for a single run).
"""

import csv
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_APPROACHES = ("testsv3", "testsv3.1", "testsv4", "testsv5")
# Validated categorical order (CVD-safe adjacent pairs); colour follows CLI order.
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
SURFACE, GRID, TEXT, TEXT_SECONDARY = "#fcfcfb", "#e4e3df", "#0b0b0b", "#52514e"
TIME_BIN_S = 0.05


@dataclass
class Approach:
    name: str
    color: str
    runs: list = field(default_factory=list)  # [(run name, outcome, columns)]

    @property
    def label(self):
        interrupted = sum(outcome != "completed" for _, outcome, _ in self.runs)
        extra = f", {interrupted} not completed" if interrupted else ""
        return f"{self.name} ({len(self.runs)} run{'s' * (len(self.runs) != 1)}{extra})"

    def per_run(self, metric):
        """One value per run; NaN where the run lacks the data."""
        values = []
        for _, _, columns in self.runs:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN columns give NaN.
                values.append(float(metric(columns)))
        return np.array(values)


def read_rollout(path):
    """Numeric rollout columns; empty or nonnumeric cells become NaN, text columns are skipped."""
    with open(path, newline="") as stream:
        reader = csv.reader(stream, delimiter=";")
        header = next(reader, [])
        rows = list(reader)
    columns = {}
    for index, name in enumerate(header):
        cells = [row[index] if index < len(row) else "" for row in rows]
        try:
            columns[name] = np.array([float(cell) if cell.strip() else np.nan for cell in cells])
        except ValueError:
            continue  # phase, accepted
    return columns


def resolve_approach(name, runs_dir):
    for candidate in (Path(name), runs_dir / name, runs_dir / f"config_{name}"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No approach folder for {name!r} in {runs_dir}")


def load_approaches(names, runs_dir, completed_only):
    if len(names) > len(PALETTE):
        raise ValueError(f"Compare at most {len(PALETTE)} approaches at once")
    approaches = []
    for name, color in zip(names, PALETTE):
        approach = Approach(name.removeprefix("config_"), color)
        for path in sorted(resolve_approach(name, runs_dir).rglob("rollout.csv")):
            summary = path.with_name("summary.json")
            outcome = json.loads(summary.read_text()).get("outcome", "unknown") if summary.exists() else "unknown"
            if completed_only and outcome != "completed":
                continue
            columns = read_rollout(path)
            if not columns.get("time_s", np.empty(0)).size:
                print(f"Skipping empty run {path.parent}")
                continue
            approach.runs.append((path.parent.name, outcome, columns))
        if not approach.runs:
            raise ValueError(f"{name}: no usable runs")
        approaches.append(approach)
    return approaches


def ordered_union(approaches, select):
    names = []
    for approach in approaches:
        for _, _, columns in approach.runs:
            names += [name for name in select(columns) if name not in names]
    return names


def wrists(approaches):
    return [name.removesuffix("/fx_n") for name in ordered_union(
        approaches, lambda columns: [c for c in columns if c.endswith("/fx_n")])]


def norm(columns, prefix, axes):
    keys = [f"{prefix}/{axis}" for axis in axes]
    if not all(key in columns for key in keys):
        return np.full(1, np.nan)
    return np.sqrt(sum(columns[key] ** 2 for key in keys))


def force(prefix):
    return lambda columns: np.nanmean(norm(columns, prefix, ("fx_n", "fy_n", "fz_n")))


def torque(prefix):
    return lambda columns: np.nanmean(norm(columns, prefix, ("tx_nm", "ty_nm", "tz_nm")))


def column_mean(name, transform=lambda values: values):
    return lambda columns: np.nanmean(transform(columns[name])) if name in columns else np.nan


def mean_std(values):
    values = values[np.isfinite(values)]
    if not values.size:
        return np.nan, np.nan
    return values.mean(), values.std(ddof=1) if values.size > 1 else np.nan


def style(plt):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "text.color": TEXT, "axes.labelcolor": TEXT_SECONDARY, "axes.titlecolor": TEXT,
        "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
        "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 10, "axes.titlesize": 12,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
    })


def spread(values, gap):
    """Nudge sorted label positions apart by at least gap."""
    order = np.argsort(values)
    placed = np.array(values, dtype=float)
    for previous, current in zip(order, order[1:]):
        placed[current] = max(placed[current], placed[previous] + gap)
    return placed


def plot_tracking(plt, approaches, path):
    fig, ax = plt.subplots(figsize=(11, 5.5), layout="constrained")
    ends = []
    for approach in approaches:
        t_max = max(np.nanmax(columns["time_s"]) for _, _, columns in approach.runs)
        bins = np.arange(int(t_max / TIME_BIN_S) + 1)
        binned = np.full((len(approach.runs), bins.size), np.nan)
        for row, (_, _, columns) in enumerate(approach.runs):
            index = (columns["time_s"] / TIME_BIN_S).astype(int)
            error = columns["tracking_error_m"] * 1000
            valid = np.isfinite(error) & (index >= 0)
            sums = np.bincount(index[valid], error[valid], bins.size)
            counts = np.bincount(index[valid], minlength=bins.size)
            binned[row] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean, low, high = np.nanmean(binned, 0), np.nanmin(binned, 0), np.nanmax(binned, 0)
        time = (bins + 0.5) * TIME_BIN_S
        rmse = mean_std(approach.per_run(lambda c: np.sqrt(np.nanmean((c["tracking_error_m"] * 1000) ** 2))))[0]
        ax.fill_between(time, low, high, color=approach.color, alpha=0.1, linewidth=0)
        ax.plot(time, mean, color=approach.color, linewidth=2, solid_capstyle="round",
                label=f"{approach.label}, RMSE {rmse:.1f} mm")
        last = np.flatnonzero(np.isfinite(mean))[-1]
        ax.plot(time[last], mean[last], "o", color=approach.color, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2)
        ends.append((time[last], mean[last], approach.name))
    ax.set_ylim(bottom=0)
    gap = 0.045 * np.diff(ax.get_ylim())[0]
    for (x, _, name), y in zip(ends, spread([y for _, y, _ in ends], gap)):
        ax.annotate(name, (x, y), xytext=(8, 0), textcoords="offset points", va="center", color=TEXT)
    ax.set_xlim(left=0)
    ax.margins(x=0.08)
    ax.set_title("RU tracking error: mean across runs (band: min–max across runs)")
    ax.set_xlabel("Simulation time since release (s)")
    ax.set_ylabel("Tracking error (mm)")
    ax.legend(loc="upper left")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def grouped_bars(plt, approaches, categories, metrics, title, unit, path, horizontal=False, value_labels=True):
    """categories[i] is plotted from metrics[i], a per-run metric function."""
    stats = np.array([[mean_std(a.per_run(metric)) for a in approaches] for metric in metrics])
    count = len(approaches)
    width = 0.8 / count
    positions = np.arange(len(categories))
    size = (11, max(4, 0.13 * count * len(categories) + 1.5)) if horizontal else (max(8, 1.0 * count * len(categories) + 2), 5.5)
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    for k, approach in enumerate(approaches):
        offset = positions - 0.4 + width * (k + 0.5)
        mean, std = stats[:, k, 0], stats[:, k, 1]
        bar = ax.barh if horizontal else ax.bar
        bar(offset, mean, width, color=approach.color, edgecolor=SURFACE, linewidth=1.5, label=approach.label)
        error_kw = dict(fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=3)
        if horizontal:
            ax.errorbar(mean, offset, xerr=std, **error_kw)
        else:
            ax.errorbar(offset, mean, yerr=std, **error_kw)
        if value_labels:
            tops = mean + np.nan_to_num(std)
            for x, top, value in zip(offset, tops, mean):
                if np.isfinite(value):
                    ax.annotate(f"{value:.3g}", (x, top), xytext=(0, 3), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, color=TEXT_SECONDARY)
    if horizontal:
        ax.set_yticks(positions, categories)
        ax.set_ylim(len(categories) - 0.5, -0.5)  # First category on top, no empty margins.
        ax.set_xlabel(unit)
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper right")
    else:
        ax.set_xticks(positions, categories)
        ax.set_ylabel(unit)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_title(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return stats


def single_bars(plt, approaches, metric, title, unit, fmt, path):
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(approaches) + 2), 5), layout="constrained")
    positions = np.arange(len(approaches))
    for x, approach in zip(positions, approaches):
        values = approach.per_run(metric)
        mean, std = mean_std(values)
        ax.bar(x, mean, 0.6, color=approach.color, edgecolor=SURFACE, linewidth=1.5)
        ax.errorbar(x, mean, yerr=std, fmt="none", ecolor=TEXT_SECONDARY, elinewidth=1, capsize=4)
        jitter = np.linspace(-0.15, 0.15, values.size) if values.size > 1 else np.zeros(1)
        ax.scatter(x + jitter, values, s=36, color=TEXT_SECONDARY, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    # Values sit in the tick labels so they never collide with run dots.
    means = [mean_std(a.per_run(metric))[0] for a in approaches]
    ax.set_xticks(positions, [f"{a.name}: {fmt.format(m)}\n{a.label.removeprefix(a.name).strip()}"
                              for a, m in zip(approaches, means)])
    ax.set_ylabel(unit)
    ax.set_ylim(bottom=0)
    ax.margins(y=0.12)
    ax.grid(axis="x", visible=False)
    ax.set_title(f"{title} (mean ± SD; dots: individual runs)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(
    approaches: tuple[str, ...] = DEFAULT_APPROACHES,
    runs_dir: Path = Path("helios_runs"),
    out_dir: Path | None = None,
    completed_only: bool = False,
):
    """Compare IK approaches; writes PNGs and summary.csv to out_dir (default runs_dir/comparison)."""
    style(plt)
    loaded = load_approaches(approaches, runs_dir, completed_only)
    out_dir = out_dir or runs_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    for approach in loaded:
        print(f"{approach.label}: {', '.join(run for run, _, _ in approach.runs)}")

    plot_tracking(plt, loaded, out_dir / "tracking_error.png")
    wrist_names = wrists(loaded)
    wrist_labels = [name.replace("_wrist", "").upper() + " wrist" for name in wrist_names]
    rows = {}
    stats = grouped_bars(plt, loaded, wrist_labels, [force(w) for w in wrist_names],
                         "Average attached-wrist force magnitude |F| (mean ± SD across runs)", "Force (N)",
                         out_dir / "wrist_forces.png")
    rows.update({f"{w}_force_n": s for w, s in zip(wrist_names, stats)})
    stats = grouped_bars(plt, loaded, wrist_labels, [torque(w) for w in wrist_names],
                         "Average attached-wrist torque magnitude |T| (mean ± SD across runs)", "Torque (N·m)",
                         out_dir / "wrist_torques.png")
    rows.update({f"{w}_torque_nm": s for w, s in zip(wrist_names, stats)})

    joints = [name.removeprefix("effort_nm/") for name in ordered_union(
        loaded, lambda columns: [c for c in columns if c.startswith("effort_nm/")])]
    efforts = [column_mean(f"effort_nm/{joint}", np.abs) for joint in joints]
    all_joints = lambda columns: np.nanmean([metric(columns) for metric in efforts])
    stats = grouped_bars(plt, loaded, ["All joints (mean)", *joints], [all_joints, *efforts],
                         "Average actuator effort |τ| per joint (mean ± SD across runs)", "Effort (N·m)",
                         out_dir / "actuator_efforts.png", horizontal=True, value_labels=False)
    rows.update({f"effort_nm/{name}": s for name, s in zip(["all_joints", *joints], stats)})

    single_bars(plt, loaded, column_mean("solve_ms"), "Average computation time per IK step",
                "Computation time (ms)", "{:.1f} ms", out_dir / "computation_time.png")
    single_bars(plt, loaded, column_mean("solver_steps"), "Average solver iterations per IK step",
                "Solver iterations", "{:.1f}", out_dir / "solver_steps.png")
    for name, metric in (("solve_ms", column_mean("solve_ms")), ("solver_steps", column_mean("solver_steps")),
                         ("tracking_rmse_mm", lambda c: np.sqrt(np.nanmean((c["tracking_error_m"] * 1000) ** 2))),
                         ("tracking_max_mm", lambda c: np.nanmax(c["tracking_error_m"]) * 1000)):
        rows[name] = [mean_std(a.per_run(metric)) for a in loaded]

    with open(out_dir / "summary.csv", "w", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["metric", *(f"{a.name} {part}" for a in loaded for part in ("mean", "sd"))])
        writer.writerow(["runs", *(v for a in loaded for v in (len(a.runs), ""))])
        for name, values in rows.items():
            writer.writerow([name, *(f"{v:.4g}" if np.isfinite(v) else "" for pair in values for v in pair)])
    print(f"Saved comparison to {out_dir}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
