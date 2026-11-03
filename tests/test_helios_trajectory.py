"""Offline checks for the clearance path and Isaac attachment handshake."""

import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from examples.helios_trajectory import ClearanceTrajectory


class TrajectoryTests(unittest.TestCase):
    def make_trajectory(self):
        return ClearanceTrajectory([0, 0, 0], [0.2, 0, 0.3], [0, 0.15, 0], [2, 3, 2], 0.02, 0.05)

    def test_clearance_and_approach_directions(self):
        trajectory = self.make_trajectory()
        for phase in range(3):
            trajectory.phase = phase
            samples = np.array([trajectory.sample(t) for t in np.linspace(0, trajectory.durations[phase], 101)])
            np.testing.assert_allclose(samples[0], trajectory.waypoints[phase])
            np.testing.assert_allclose(samples[-1], trajectory.waypoints[phase + 1], atol=1e-12)
            if phase == 0:
                np.testing.assert_allclose(samples[:, [0, 2]], 0)
                self.assertTrue(np.all(np.diff(samples[:, 1]) >= 0))
            elif phase == 1:
                np.testing.assert_allclose(samples[:, 1], 0.15)
            else:
                np.testing.assert_allclose(samples[:, 0], 0.2)
                np.testing.assert_allclose(samples[:, 2], 0.3)
                self.assertTrue(np.all(np.diff(samples[:, 1]) <= 0))

    def test_smooth_endpoints(self):
        trajectory = self.make_trajectory()
        h = 1e-4
        for phase in range(3):
            trajectory.phase = phase
            duration = trajectory.durations[phase]
            for t, step in ((0, h), (duration, -h)):
                p0, p1, p2 = [trajectory.sample(t + i * step) for i in range(3)]
                self.assertLess(np.linalg.norm((p1 - p0) / step), 1e-6)
                self.assertLess(np.linalg.norm((p2 - 2 * p1 + p0) / step**2), 0.001)

    def test_progress_waits_for_hand_and_successful_command(self):
        trajectory = self.make_trajectory()
        _, elapsed = trajectory.propose(10, [0, 0, 0])
        self.assertEqual(elapsed, 0.05)  # A slow frame cannot jump ahead.
        self.assertEqual(trajectory.elapsed, 0)  # Rejected IK leaves progress unchanged.
        trajectory.commit(elapsed, [0, 0, 0])
        _, held = trajectory.propose(0.02, [1, 1, 1])
        self.assertEqual(held, elapsed)
        trajectory.commit(2, [0, 0, 0])
        self.assertEqual(trajectory.phase, 0)  # Cannot travel until hand clears bar.
        trajectory.commit(2, [0, 0.15, 0])
        self.assertEqual(trajectory.phase, 1)
        trajectory.commit(3, trajectory.waypoints[2])
        trajectory.commit(2, trajectory.waypoints[3])
        self.assertTrue(trajectory.done)
        np.testing.assert_allclose(trajectory.sample(100), trajectory.waypoints[-1])

    def test_pause_does_not_advance(self):
        trajectory = self.make_trajectory()
        _, elapsed = trajectory.propose(0, trajectory.waypoints[0])
        self.assertEqual(elapsed, 0)

    def test_invalid_config(self):
        for durations in ([0, 1, 1], [1, float("nan"), 1], [1, 2]):
            with self.assertRaises(ValueError):
                ClearanceTrajectory([0]*3, [1]*3, [0, 0.15, 0], durations, 0.02, 0.05)


class AttachmentTests(unittest.TestCase):
    def setUp(self):
        # USD/Isaac are not installed in the terminal venv. Exercise the real
        # helper callbacks with an in-memory joint attribute and fake scene poses.
        usd = SimpleNamespace(EditContext=lambda *_: contextlib.nullcontext())
        pxr = SimpleNamespace(Gf=Mock(), Sdf=Mock(), Usd=usd, UsdGeom=Mock(), UsdPhysics=Mock())
        source = Path(__file__).resolve().parents[1] / "examples/helios_attach_isaac.py"
        with patch.dict("sys.modules", {"pxr": pxr}):
            module = runpy.run_path(str(source))
        self.attach_distance = module["ATTACH_DISTANCE"]
        helper = module["HeliosAttachment"].__new__(module["HeliosAttachment"])
        self.helper = helper
        self.enabled = False
        attr = Mock()
        attr.Get.side_effect = lambda: self.enabled
        attr.Set.side_effect = lambda value: setattr(self, "enabled", value)
        helper.ru = Mock()
        helper.ru.GetJointEnabledAttr.return_value = attr
        helper.stage, helper.layer = object(), object()
        helper.usd_context = Mock()
        helper.usd_context.get_stage.return_value = helper.stage
        helper.timeline = Mock()
        helper.timeline.is_stopped.return_value = False
        helper.timeline.is_playing.return_value = True
        helper.timeline.get_current_time.return_value = 5.0
        helper.closed, helper.owner_pid = False, None
        helper.armed, helper.command_id = False, None
        helper.effort_names = []
        helper.effort_warning = False
        helper.articulation = None
        helper.wrist_force_index = None
        helper.wrist_warning = False
        helper.joints, helper.metres = [0, 1, 2, 3], 1.0
        helper.ros, helper.publisher, helper.node = Mock(), Mock(), Mock()
        helper.message_type = SimpleNamespace
        helper.transform_point = Mock()
        helper.close = Mock()
        self.hand = np.zeros(3)
        helper.update.__func__.__globals__["anchor_position"] = (
            lambda _joint, side, _transform: self.hand if side == 1 else np.zeros(3)
        )

    def command(self, action, command_id="transfer-1"):
        self.helper.receive_command(SimpleNamespace(data=json.dumps({"action": action, "id": command_id})))

    def update(self):
        self.helper.last_update = 0
        self.helper.update(None)
        self.helper.close.assert_not_called()
        return json.loads(self.helper.publisher.publish.call_args.args[0].data)

    def test_nearby_hand_does_not_attach_until_armed(self):
        self.assertFalse(self.update()["attached"])
        self.command("release")
        self.command("arm", "wrong-id")
        self.assertFalse(self.update()["attached"])
        self.command("arm")
        status = self.update()
        self.assertTrue(status["attached"])
        self.assertEqual(status["command_id"], "transfer-1")
        self.assertEqual(status["hand_m"], [0, 0, 0])
        self.assertEqual(status["sim_time_s"], 5.0)

    def test_release_and_duplicate_commands(self):
        self.enabled = True
        self.command("release")
        self.assertFalse(self.enabled)
        self.assertFalse(self.helper.armed)
        self.command("arm")
        self.update()
        self.command("release")  # Delayed duplicate must not detach again.
        self.assertTrue(self.enabled)
        self.assertTrue(self.helper.armed)
        self.command("release", "transfer-2")
        self.assertFalse(self.enabled)
        self.assertFalse(self.helper.armed)

    def test_armed_still_requires_distance_threshold(self):
        self.command("release")
        self.command("arm")
        self.hand = np.array([0, self.attach_distance, 0])
        self.assertFalse(self.update()["attached"])
        self.hand = np.array([0, self.attach_distance - 0.001, 0])
        self.assertTrue(self.update()["attached"])

    def test_measured_efforts_use_selected_dof_indices(self):
        helper = self.helper
        helper.effort_names = helper.measured_names = ["Shoulder1_RU", "right_wrist"]
        helper.effort_indices = np.array([0, 2])
        helper.articulation = Mock()
        helper.articulation.get_measured_joint_efforts.return_value = np.array([-2.5, 1.5])
        status = self.update()
        self.assertEqual(status["efforts_nm"], {"Shoulder1_RU": -2.5, "right_wrist": 1.5})
        indices = helper.articulation.get_measured_joint_efforts.call_args.kwargs["joint_indices"]
        np.testing.assert_array_equal(indices, [0, 2])
        helper.articulation.get_measured_joint_efforts.return_value = None
        self.assertEqual(self.update()["efforts_nm"], {})

    def test_wrist_wrench_uses_joint_metadata_plus_one(self):
        helper = self.helper
        helper.articulation = Mock()
        helper.articulation._articulation_view._metadata.joint_indices = {"or4r_right_wrist": 22}
        helper.articulation.get_dof_index.return_value = 7  # Deliberately different from the force row.
        helper.articulation.get_measured_joint_forces.return_value = np.array([[3, -4, 5, -1, 2, -3]])
        self.assertEqual(self.update()["rl_wrist_wrench"], [3, -4, 5, -1, 2, -3])
        indices = helper.articulation.get_measured_joint_forces.call_args.kwargs["joint_indices"]
        np.testing.assert_array_equal(indices, [23])
        helper.articulation.get_dof_index.assert_not_called()
        helper.articulation.get_measured_joint_forces.return_value = np.full((1, 6), np.nan)
        self.assertIsNone(self.update()["rl_wrist_wrench"])
        helper.articulation.get_measured_joint_forces.return_value = None
        self.assertIsNone(self.update()["rl_wrist_wrench"])

    def test_ru_tilt_comes_from_forearm_not_hand(self):
        import jaxlie

        helper = self.helper
        helper.ru_forearm_path = "/forearm"
        helper.use_backend = lambda *_args, **_kwargs: contextlib.nullcontext()
        helper.backend = "fabric"
        rotation = (jaxlie.SO3.from_x_radians(-np.pi / 2 + 0.4)
                    @ jaxlie.SO3.from_z_radians(-0.7))
        orientation = Mock()
        orientation.numpy.return_value = np.asarray(rotation.wxyz)[None]
        forearm, palm = Mock(), Mock()
        forearm.get_world_poses.return_value = (None, orientation)
        helper.prims = {"/forearm": forearm, "/palm": palm}
        self.assertAlmostEqual(self.update()["ru_roll_rad"], 0.4, places=6)
        palm.get_world_poses.assert_not_called()
        orientation.numpy.return_value = np.full((1, 4), np.nan)
        self.assertIsNone(self.update()["ru_roll_rad"])

    def test_root_pose_preserves_world_rotation_and_converts_to_metres(self):
        self.helper.articulation = Mock()
        self.helper.metres = 0.01
        self.helper.articulation.get_world_pose.return_value = (
            np.array([100, 200, 300]), np.array([0.5, -0.5, 0.5, -0.5])
        )
        self.assertEqual(self.helper.read_root_pose(), [0.5, -0.5, 0.5, -0.5, 1, 2, 3])
        self.helper.articulation = None
        self.assertIsNone(self.helper.read_root_pose())


class ControllerTests(unittest.TestCase):
    def test_key_release_ack_clearance_then_docking(self):
        examples = Path(__file__).resolve().parents[1] / "examples"
        with patch.object(sys, "path", [str(examples), *sys.path]):
            module = runpy.run_path(str(examples / "helios_retargetiing.py"))
        jnp, jaxlie = module["jnp"], module["jaxlie"]
        supports = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]])
        goal = np.array([0.2, 0, 0.3])
        status = dict(anchors_m=np.vstack([goal, supports]), hand_m=np.zeros(3),
                      sim_time_s=0.0, attached=False, armed=False, command_id=None,
                      rl_wrist_wrench=[1, 2, 3, 4, 5, 6], root_pose_wxyz_xyz=[1, 0, 0, 0, 10, 20, 30])
        published, commands, seeds, orientation_refs = [], [], [], []
        seen_reattached = False
        reads, key_reads, release_wait = 0, 0, 0
        pending_id = None

        def read():
            nonlocal reads, release_wait
            reads += 1
            status["sim_time_s"] += 0.02
            if release_wait:
                release_wait -= 1
                if not release_wait:
                    status["command_id"] = pending_id
            # Deliberately different later feedback must not overwrite the IK seed.
            return jnp.zeros(3) if reads == 1 else jnp.full(3, 99.0)

        def command(action, command_id):
            nonlocal pending_id, release_wait
            commands.append(action)
            if action == "release":
                pending_id, release_wait = command_id, 3
            else:
                self.assertEqual(command_id, status["command_id"])
                np.testing.assert_allclose(status["hand_m"], goal, atol=1e-6)
                status["armed"], status["attached"] = True, True

        def publish(_names, cfg):
            if status["command_id"] is None:
                np.testing.assert_allclose(cfg, 0)  # Wait for release acknowledgement.
            status["hand_m"] = np.array(cfg)
            published.append(np.array(cfg))
            return True

        def key():
            nonlocal key_reads
            key_reads += 1
            if key_reads > 100:
                self.fail("Controller did not complete the transfer")
            if key_reads in (3, 4):  # Second press while running must be ignored.
                return " "
            return "q" if seen_reattached else ""

        def fk(cfg):
            positions = jnp.vstack([cfg, jnp.array(supports)])
            rotations = jnp.tile(jnp.array([1.0, 0, 0, 0]), (4, 1))
            rotations = rotations.at[0].set(jaxlie.SO3.from_z_radians(cfg[0]).wxyz)
            return jnp.hstack([rotations, positions])

        def solve(_robot, _indices, targets, _orientations, cfg, _base, _attached, _orientation_weight, _ik_to_world):
            nonlocal seen_reattached
            seeds.append(np.array(cfg))
            orientation_refs.append(np.array(_orientations))
            self.assertEqual(_orientation_weight, 25.0)
            seen_reattached = _attached and status["attached"]
            return jnp.array(targets[0]), jaxlie.SE3.identity(), jnp.array(5), {"total_objective": jnp.array(0.0)}

        bridge = SimpleNamespace(read=read, attachment_status=lambda: dict(status),
                                 command_attachment=command, publish=publish,
                                 state=(SimpleNamespace(name=["x", "y", "z"], position=[99]*3), 0))
        robot = SimpleNamespace(forward_kinematics=fk, joints=SimpleNamespace(names=["x", "y", "z"], get_full_config=lambda cfg: cfg))
        viewer = Mock()
        run = module["run_controller"]
        settings = dict(durations_s=[0.1, 0.1, 0.1], waypoint_tolerance_m=0.02, tracking_tolerance_m=0.05)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(run.__globals__, {"solve_ik": solve}), patch("time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                run(robot, jnp.arange(4), ["x", "y", "z"], jnp.arange(3), bridge, SimpleNamespace(read=key),
                    0.15, settings, orientation_weight=25.0, metrics_dir=Path(directory), viewer=viewer)
            summaries = list(Path(directory).glob("*/summary.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(summary["outcome"], "completed")
            self.assertEqual(summary["average_solver_steps"], 5)
            self.assertEqual(summary["ik_calls"], len(seeds) - 2)  # Warmup excluded.
            self.assertEqual(summary["rl_wrist"]["valid_samples"], summary["ik_calls"])
            self.assertTrue((summaries[0].parent / "metrics.png").is_file())
        self.assertEqual(commands, ["release", "arm"])
        self.assertTrue(status["attached"])
        np.testing.assert_allclose(list(viewer.publish.call_args.args[0].values()), goal, atol=1e-6)
        self.assertIs(viewer.publish.call_args.args[2], bridge.state[0])
        self.assertEqual(viewer.publish.call_args.args[3], status["root_pose_wxyz_xyz"])
        # Support references stay latched; RU latches its new orientation on docking.
        np.testing.assert_allclose(np.array(orientation_refs)[:, 1:], np.tile([1, 0, 0, 0], (len(orientation_refs), 3, 1)))
        np.testing.assert_allclose(orientation_refs[-1][0], jaxlie.SO3.from_z_radians(goal[0]).wxyz)
        self.assertLess(np.max(np.abs(seeds)), 1)  # Never reseeded from the 99-valued feedback.
        samples = np.array(published)
        travel = samples[(samples[:, 0] > 1e-6) & (samples[:, 0] < goal[0] - 1e-6)]
        self.assertGreater(len(travel), 0)
        np.testing.assert_allclose(travel[:, 1], 0.15, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
