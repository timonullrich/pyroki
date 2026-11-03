import csv
import json
import tempfile
import unittest

from examples.helios_metrics import RolloutMetrics


class MetricsTests(unittest.TestCase):
    def test_summary_raw_data_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = RolloutMetrics(directory, "test-run", 10, ["Shoulder1_RU", "Rcj2_RU"],
                                     {"test": True, "nested": [{"value": 1.234567e-7}]})
            costs = {"motion": 2.0, "fixed_positions_violation": 0.1, "total_objective": 2.0, "initial_objective": 10.0}
            metrics.add(10.1, [0.003, 0, 0], [0, 0, 0], {"Shoulder1_RU": -2.123456, "Rcj2_RU": 3}, 10, 4, costs, True, "retreat",
                        rl_wrist_wrench=[3, 4, 0, -2, 0, 0], ru_roll_rad=0.123456,
                        lu_wrist_wrench=[5, 0, 0, 0, 3, 0], ll_wrist_wrench=[-4, 0, 0, 0, -1, 0])
            metrics.add(10.2, [0, 0.004, 0], [0, 0, 0], {"Shoulder1_RU": 1}, 30, 6, costs, False, "travel")
            metrics.save()
            summary = json.loads((metrics.path / "summary.json").read_text())
            self.assertEqual(summary["average_computation_ms"], 20)
            self.assertEqual(summary["average_solver_steps"], 5)
            self.assertEqual(summary["tracking_rmse_mm"], 3.536)
            self.assertEqual(summary["settings"]["nested"][0]["value"], 1.235e-7)
            self.assertIs(summary["settings"]["test"], True)
            self.assertEqual(summary["rejected_commands"], 1)
            self.assertEqual(summary["mean_initial_objective"], 10.0)
            self.assertEqual(summary["mean_total_objective"], 2.0)
            self.assertEqual(summary["rl_wrist"]["max_force_norm_n"], 5)
            self.assertEqual(summary["rl_wrist"]["max_torque_norm_nm"], 2)
            self.assertEqual(summary["rl_wrist"]["missing_samples"], 1)
            self.assertEqual(summary["joints_with_missing_effort_samples"], ["Rcj2_RU"])
            with (metrics.path / "rollout.csv").open() as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
            self.assertEqual(len(rows), 2)
            self.assertEqual(float(rows[0]["effort_nm/Shoulder1_RU"]), -2.123)
            self.assertEqual(metrics.rows[0]["effort_nm/Shoulder1_RU"], -2.123456)
            self.assertEqual(rows[1]["effort_nm/Rcj2_RU"], "nan")
            self.assertEqual(float(rows[0]["rl_wrist/tx_nm"]), -2)
            self.assertEqual(rows[1]["rl_wrist/fx_n"], "nan")
            self.assertEqual(float(rows[0]["lu_wrist/fx_n"]), 5)
            self.assertEqual(float(rows[0]["ll_wrist/ty_nm"]), -1)
            self.assertEqual(rows[1]["lu_wrist/fx_n"], "nan")
            self.assertEqual(float(rows[0]["ru_roll_rad"]), 0.1235)
            self.assertEqual(rows[1]["ru_roll_rad"], "nan")
            self.assertTrue((metrics.path / "metrics.png").stat().st_size > 1000)
            metrics.save("interrupted")
            self.assertEqual(json.loads((metrics.path / "summary.json").read_text())["outcome"], "completed")


if __name__ == "__main__":
    unittest.main()
