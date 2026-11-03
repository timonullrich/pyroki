"""One CSV, JSON summary, and PNG per Helios rollout. No live plotting."""

import atexit
import csv
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PLOT_SMOOTHING_ALPHA = 0.15  # Lower values smooth more; CSV and summary stay raw.


def round_saved_numbers(value):
    """Round exported floats to four significant figures, including nested settings."""
    if isinstance(value, dict):
        return {key: round_saved_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_saved_numbers(item) for item in value]
    return float(f"{value:.4g}") if isinstance(value, float) else value


def smooth(values):
    """Exponential moving average, restarting after missing samples."""
    values = np.asarray(values, dtype=float).copy()
    for i in range(1, len(values)):
        if np.isfinite(values[i]) and np.isfinite(values[i - 1]):
            values[i] = PLOT_SMOOTHING_ALPHA * values[i] + (1 - PLOT_SMOOTHING_ALPHA) * values[i - 1]
    return values


class RolloutMetrics:
    def __init__(self, directory, run_id, sim_time, effort_names, settings):
        stamp = datetime.now(ZoneInfo("Europe/Zurich")).strftime("%Y%m%d_%H%M%S")
        self.path = Path(directory) / f"{stamp}_{run_id[:8]}"
        self.start_time = sim_time
        self.effort_names = list(effort_names)
        self.settings = settings
        self.rows = []
        self.saved = False
        atexit.register(self.save, "interrupted")

    def add(self, sim_time, hand, target, efforts, solve_ms, steps, costs, accepted, phase, rl_wrist_wrench=None, ru_roll_rad=None):
        row = {
            "time_s": float(sim_time - self.start_time), "phase": phase,
            "solve_ms": float(solve_ms), "solver_steps": int(steps), "accepted": bool(accepted),
            "tracking_error_m": float(np.linalg.norm(np.asarray(hand) - target)),
            "ru_roll_rad": float(ru_roll_rad) if ru_roll_rad is not None else np.nan,
        }
        for i, axis in enumerate("xyz"):
            row[f"hand_{axis}_m"] = float(hand[i])
            row[f"target_{axis}_m"] = float(target[i])
        row.update({f"effort_nm/{name}": float(efforts.get(name, np.nan)) for name in self.effort_names})
        row.update({f"cost/{name}": float(value) for name, value in costs.items()})
        wrench = np.full(6, np.nan) if rl_wrist_wrench is None else np.asarray(rl_wrist_wrench, dtype=float)
        if wrench.shape != (6,):
            raise ValueError("Expected six lower-right wrist force/torque components")
        for name, value in zip(("fx_n", "fy_n", "fz_n", "tx_nm", "ty_nm", "tz_nm"), wrench):
            row[f"rl_wrist/{name}"] = float(value)
        self.rows.append(row)

    def save(self, outcome="completed"):
        if self.saved or not self.rows:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / "rollout.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]), delimiter=";")
            writer.writeheader()
            writer.writerows(round_saved_numbers(row) for row in self.rows)
        column = lambda key: np.array([row[key] for row in self.rows])
        error = column("tracking_error_m")
        missing = [name for name in self.effort_names if not np.isfinite(column(f"effort_nm/{name}")).all()]
        wrist_force = np.column_stack([column(f"rl_wrist/f{axis}_n") for axis in "xyz"])
        wrist_torque = np.column_stack([column(f"rl_wrist/t{axis}_nm") for axis in "xyz"])
        wrist_valid = np.isfinite(wrist_force).all(axis=1) & np.isfinite(wrist_torque).all(axis=1)
        summary = {
            "outcome": outcome, "ik_calls": len(self.rows),
            "duration_sim_s": float(self.rows[-1]["time_s"]),
            "average_computation_ms": float(column("solve_ms").mean()),
            "average_solver_steps": float(column("solver_steps").mean()),
            "tracking_rmse_mm": float(np.sqrt(np.mean(error**2)) * 1000),
            "tracking_max_mm": float(error.max() * 1000),
            "rejected_commands": int(np.count_nonzero(~column("accepted"))),
            "joints_with_missing_effort_samples": missing,
            "rl_wrist": {
                "joint": "or4r_right_wrist", "frame": "child joint frame",
                "valid_samples": int(wrist_valid.sum()), "missing_samples": int((~wrist_valid).sum()),
                "max_force_norm_n": float(np.linalg.norm(wrist_force[wrist_valid], axis=1).max()) if wrist_valid.any() else None,
                "max_torque_norm_nm": float(np.linalg.norm(wrist_torque[wrist_valid], axis=1).max()) if wrist_valid.any() else None,
            },
            "settings": self.settings,
            "definitions": {
                "computation": "Synchronized IK call including returned diagnostics; excludes JIT warmup, logging and plots.",
                "steps": "jaxls SolveSummary.iterations per IK call, including rejected solves.",
                "tracking": "Measured RU attachment-point position versus the previously published target at feedback sampling time.",
                "ru_roll": "RU forearm direction (wrist parent frame +Z) projected into global Y-Z: atan2(direction_z, abs(direction_y)), 0 = horizontal along either +/-Y. Independent of wrist rotation. CSV in radians, plot in degrees. Missing samples are NaN.",
                "efforts": "Isaac get_measured_joint_efforts(), N m; missing samples are NaN, not zero.",
                "rl_wrist": "Direct get_measured_joint_forces() at or4r_right_wrist: Fx/Fy/Fz in N and Tx/Ty/Tz in N m, expressed in the child joint frame. Missing readings are NaN. This is the transmitted wrist-joint wrench, not the external D6 attachment wrench.",
                "costs": "The IK objective IS the sum of squared weighted soft residuals. total_objective is evaluated at the optimizer's returned solution; initial_objective is JAXLS's recorded value before solving. Hard-constraint curves show squared violations separately. These are model-space values, not measured Isaac tracking costs, and exclude internal augmented-Lagrangian penalties.",
                "sampling": "One row per active control-loop solve (release through attachment); latest Isaac sample may repeat.",
            },
        }
        for name in ("initial_objective", "total_objective"):
            key = f"cost/{name}"
            if key in self.rows[0]:
                values = column(key)
                finite = values[np.isfinite(values)]
                summary[f"mean_{name}"] = float(finite.mean()) if finite.size else None
                summary[f"max_{name}"] = float(finite.max()) if finite.size else None
        (self.path / "summary.json").write_text(json.dumps(round_saved_numbers(summary), indent=2) + "\n")
        self.saved = True
        atexit.unregister(self.save)

        t = column("time_s")
        fig, axes = plt.subplots(6, 1, figsize=(16, 23), sharex=True, layout="constrained")
        weights = ", ".join(f"{name}={value:.4g}" for name, value in self.settings.get("residual_weights", {}).items())
        fig.suptitle(
            f"Helios — {outcome} | mean IK {summary['average_computation_ms']:.2f} ms | "
            f"mean steps {summary['average_solver_steps']:.1f} | curves smoothed (EMA)"
            + (f"\nResidual weights: {weights}" if weights else "")
        )
        axes[0].plot(t, error * 1000, label="RU position error", color="black")
        axes[0].set(title="Measured right-hand tracking error", ylabel="Error (mm)")
        axes[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        for i, name in enumerate(self.effort_names):
            axes[1].plot(t, column(f"effort_nm/{name}"), label=name,
                         color=plt.get_cmap("tab20")(i % 20), linestyle="-" if i < 20 else "--", linewidth=1)
        axes[1].set(title="Measured actuator efforts (fingers excluded)", ylabel="Effort (N·m)")
        axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=7, ncols=2)
        if missing:
            axes[1].text(0.01, 0.98, "Missing effort samples: see summary.json", transform=axes[1].transAxes, va="top")
        for key in self.rows[0]:
            if key.startswith("cost/"):
                name = key.removeprefix("cost/")
                hard = name.endswith("_violation")
                if name in ("initial_objective", "total_objective"):
                    initial = name == "initial_objective"
                    axes[2].plot(t, column(key), color="black", linestyle=":" if initial else "-", linewidth=2,
                                 label="Objective BEFORE solve" if initial else "Objective AFTER solve")
                else:
                    label = name.replace("_violation", " (violation²)").replace("_", " ")
                    axes[2].plot(t, column(key), label=label, linestyle="--" if hard else "-", linewidth=1)
        axes[2].set(title="IK objective before/after optimization and final term contributions",
                    ylabel="Objective value / squared constraint violation", xlabel="Simulation time since trigger (s)")
        axes[2].set_yscale("symlog", linthresh=1e-8)
        axes[2].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
        for axis, values, title, unit, prefix in (
            (axes[3], wrist_force, "Lower-right wrist transmitted force", "N", "F"),
            (axes[4], wrist_torque, "Lower-right wrist transmitted torque", "N·m", "T"),
        ):
            for i, component in enumerate("xyz"):
                axis.plot(t, values[:, i], label=f"{prefix}{component}")
            axis.plot(t, np.linalg.norm(values, axis=1), label="Magnitude", color="black", linestyle="--")
            axis.set(title=f"{title} (child joint frame)", ylabel=unit)
            axis.legend(loc="upper left", bbox_to_anchor=(1, 1))
            if not wrist_valid.all():
                axis.text(0.01, 0.98, "Missing wrist samples: see summary.json", transform=axis.transAxes, va="top")
        axes[2].set_xlabel("")
        axes[5].plot(t, np.degrees(column("ru_roll_rad")), label="Measured forearm tilt")
        axes[5].axhline(0, color="gray", linestyle="--", label="Target (0°)")
        axes[5].set(title="Measured RU forearm tilt in global Y-Z plane", ylabel="Tilt (°)", xlabel="Simulation time since trigger (s)")
        axes[5].legend(loc="upper left", bbox_to_anchor=(1, 1))
        for axis in axes:
            for line in axis.lines:
                line.set_ydata(smooth(line.get_ydata()))
            axis.relim()
            axis.autoscale_view()
            axis.grid(True, alpha=0.25)
        fig.savefig(self.path / "metrics.png", dpi=160)
        plt.close(fig)
        print(f"Metrics saved: {self.path} | mean IK {summary['average_computation_ms']:.2f} ms, "
              f"mean steps {summary['average_solver_steps']:.1f}, RU RMSE {summary['tracking_rmse_mm']:.2f} mm")
        if missing:
            print(f"Effort data missing for some samples: {', '.join(missing)}")
        if not wrist_valid.all():
            print(f"Lower-right wrist wrench missing for {int((~wrist_valid).sum())} samples")
