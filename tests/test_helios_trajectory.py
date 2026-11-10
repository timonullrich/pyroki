"""Offline checks for the clearance path and manually triggered Isaac grasp."""

import contextlib
import builtins
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import jaxlie

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
with patch.object(sys, "path", [str(EXAMPLES), *sys.path]):
    CONTROLLER = runpy.run_path(str(EXAMPLES / "helios_retargeting.py"))
ClearanceTrajectory = CONTROLLER["ClearanceTrajectory"]


class LauncherTests(unittest.TestCase):
    def test_launcher_refuses_playing_and_passes_configuration_when_stopped(self):
        source = "def start_grasp(*args, **kwargs):\n import builtins\n builtins._helios_test_args = (args, kwargs)\n"
        start = CONTROLLER["start_grasp_helper"]
        request = Mock(return_value={})
        actuators = {"body": {"joint_names_expr": ["motor"], "stiffness": 71.1}}
        grasp = {"closing_speed_rad_s": 0.2}
        with patch.dict(start.__globals__, {"isaac_request": request}), \
                patch.object(Path, "read_text", return_value=source):
            start(8226, ["motor"], actuators, grasp)
        code = request.call_args.args[0]
        timeline = SimpleNamespace(get_timeline_interface=lambda: SimpleNamespace(is_stopped=lambda: False))
        with patch.dict(sys.modules, {"omni": SimpleNamespace(timeline=timeline), "omni.timeline": timeline}):
            with self.assertRaisesRegex(RuntimeError, "Stop Isaac"):
                exec(code, {})
            timeline.get_timeline_interface = lambda: SimpleNamespace(is_stopped=lambda: True)
            try:
                exec(code, {})
                self.assertEqual(builtins._helios_test_args, ((os.getpid(), actuators, ["motor"], CONTROLLER["RU_TARGET_M"], CONTROLLER["HAND_OFFSET_M"]), grasp))
            finally:
                del builtins._helios_test_args

    def test_close_uses_one_direct_isaac_call(self):
        bridge = CONTROLLER["IsaacBridge"].__new__(CONTROLLER["IsaacBridge"])
        bridge.isaac_port = 8226
        helper = SimpleNamespace(start_closing=Mock())
        request = lambda code, *_args, **_kwargs: exec(code, {})
        with patch.object(builtins, "helios_grasp", helper, create=True), \
                patch.dict(bridge.start_closing.__func__.__globals__, {"isaac_request": request}):
            bridge.start_closing()
            helper.start_closing.assert_called_once()


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


class GraspTests(unittest.TestCase):
    def setUp(self):
        # Exercise the real helper callbacks without an Isaac runtime.
        usd = SimpleNamespace(EditContext=lambda *_: contextlib.nullcontext())
        pxr = SimpleNamespace(Gf=Mock(), Sdf=Mock(), Usd=usd, UsdGeom=Mock(), UsdPhysics=Mock())
        source = Path(__file__).resolve().parents[1] / "examples/helios_grasp_isaac.py"
        with patch.dict("sys.modules", {"pxr": pxr}):
            module = runpy.run_path(str(source))
        helper = module["HeliosGrasp"].__new__(module["HeliosGrasp"])
        self.helper = helper
        helper.stage, helper.layer = object(), object()
        helper.usd_context = Mock()
        helper.usd_context.get_stage.return_value = helper.stage
        helper.timeline = Mock()
        helper.timeline.is_stopped.return_value = False
        helper.timeline.is_playing.return_value = True
        helper.timeline.get_current_time.return_value = 5.0
        helper.closed, helper.owner_pid, helper.has_played = False, os.getpid(), False
        helper.closing = False
        helper.ru_palm_path = "/palm"
        helper.hand_offset = np.zeros(3)
        helper.target_m = np.zeros(3)
        helper.grasp_indices = np.array([3, 8])
        helper.closed_targets = np.array([1.75, 1.87])
        helper.closing_speed = 0.35
        helper.closing_start = None
        helper.closing_elapsed_s = 0.0
        # Capture the articulation actions the helper sends to Isaac.
        types = SimpleNamespace(ArticulationAction=lambda **kwargs: kwargs)
        isaac = {"isaacsim": SimpleNamespace(), "isaacsim.core": SimpleNamespace(),
                 "isaacsim.core.utils": SimpleNamespace(), "isaacsim.core.utils.types": types}
        patcher = patch.dict("sys.modules", isaac)
        patcher.start()
        self.addCleanup(patcher.stop)
        helper.effort_names = []
        helper.articulation = None
        helper.metres = 1.0
        helper.publisher, helper.node = Mock(), Mock()
        helper.message_type = SimpleNamespace
        helper.transform_point = Mock()
        helper.close = Mock()
        helper.transform_point.return_value = np.zeros(3)

    def update(self):
        self.helper.update(None)
        self.helper.close.assert_not_called()
        return json.loads(self.helper.publisher.publish.call_args.args[0].data)

    def test_arrival_does_not_close_without_command(self):
        status = self.update()
        self.assertFalse(status["closing"])
        self.assertNotIn("attached", status)
        self.assertEqual(status["hand_m"], [0, 0, 0])
        self.assertEqual(status["sim_time_s"], 5.0)

    def targets(self):
        return [call.args[0] for call in self.helper.articulation.apply_action.call_args_list]

    def test_command_ramps_pd_targets_without_touching_gains_or_efforts(self):
        helper = self.helper
        helper.articulation = Mock()
        helper.articulation.get_joint_positions.return_value = np.array([0.1, 1.9])
        helper.physics_step(0.01)
        helper.articulation.apply_action.assert_not_called()
        helper.start_closing()
        self.assertTrue(helper.closing)
        # Simulation time drives the target ramp at closing_speed, toward each closing limit.
        helper.physics_step(1.0)
        helper.start_closing()  # Repeated command must not restart the ramp.
        helper.physics_step(10.0)
        expected = [[0.1, 1.9], [0.45, 1.87], [1.75, 1.87]]
        self.assertEqual(len(self.targets()), len(expected))
        for action, target in zip(self.targets(), expected):
            np.testing.assert_allclose(action["joint_positions"], target)
            np.testing.assert_array_equal(action["joint_indices"], [3, 8])
        helper.articulation.set_joint_efforts.assert_not_called()
        helper.articulation._articulation_view.set_gains.assert_not_called()
        self.assertTrue(self.update()["closing"])

    def test_invalid_closing_speed_is_rejected_before_isaac_setup(self):
        for value in (0, -1, np.nan, np.inf):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "closing_speed_rad_s"):
                type(self.helper)(0, {}, [], [0, 0, 0], [0, 0, 0], closing_speed_rad_s=value)

    def test_stop_reopens_closing_targets(self):
        helper = self.helper
        helper.articulation = Mock()
        helper.closing = True
        helper.subscription = Mock()
        helper.layer = SimpleNamespace(identifier="grasp")
        helper.stage = Mock()
        helper.usd_context.get_stage.return_value = helper.stage
        helper.stage.GetSessionLayer.return_value.subLayerPaths = ["grasp"]
        type(helper).close(helper)
        np.testing.assert_array_equal(self.targets()[-1]["joint_positions"], [0, 0])
        np.testing.assert_array_equal(self.targets()[-1]["joint_indices"], [3, 8])
        self.assertFalse(helper.closing)
        self.assertTrue(helper.closed)
        self.assertEqual(helper.stage.GetSessionLayer().subLayerPaths, [])

    def test_owner_exit_closes_helper(self):
        with patch("os.kill", side_effect=ProcessLookupError):
            self.helper.update(None)
        self.helper.close.assert_called_once()

    def test_measured_efforts_use_selected_dof_indices(self):
        helper = self.helper
        helper.effort_names = ["Shoulder1_RU", "right_wrist"]
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
        helper.articulation._articulation_view._metadata.joint_indices = {
            "or4r_right_wrist": 22, "left_wrist": 10, "or4l_left_wrist": 30,
        }
        helper.articulation.get_dof_index.return_value = 7  # Deliberately different from the force row.
        helper.articulation.get_measured_joint_forces.return_value = np.array([[3, -4, 5, -1, 2, -3]])
        status = self.update()
        for hand, call, index in zip(("rl", "lu", "ll"), helper.articulation.get_measured_joint_forces.call_args_list, (23, 11, 31)):
            self.assertEqual(status[f"{hand}_wrist_wrench"], [3, -4, 5, -1, 2, -3])
            np.testing.assert_array_equal(call.kwargs["joint_indices"], [index])
        helper.articulation.get_dof_index.assert_not_called()
        helper.articulation.get_measured_joint_forces.return_value = np.full((1, 6), np.nan)
        self.assertIsNone(self.update()["rl_wrist_wrench"])
        helper.articulation.get_measured_joint_forces.return_value = None
        self.assertIsNone(self.update()["rl_wrist_wrench"])

    def test_ru_tilt_comes_from_forearm_not_hand(self):
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
    def test_arrival_waits_for_manual_closing(self):
        self.check_transfer(jaxlie.SE3.identity())

    def test_initial_world_transform_uses_isaac_root_pose_once(self):
        root = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_z_radians(0.7), np.array([1., 2., 3.]))
        self.check_transfer(root)

    def check_transfer(self, root):
        examples = Path(__file__).resolve().parents[1] / "examples"
        with patch.object(sys, "path", [str(examples), *sys.path]):
            module = runpy.run_path(str(examples / "helios_retargeting.py"))
        jnp, jaxlie = module["jnp"], module["jaxlie"]
        supports = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]])
        goal = np.array([0.2, 0, 0.3])
        status = dict(target_m=np.asarray(root @ goal), hand_m=np.asarray(root @ np.zeros(3)),
                      sim_time_s=0.0, closing=False,
                      rl_wrist_wrench=[1, 2, 3, 4, 5, 6], root_pose_wxyz_xyz=np.asarray(root.wxyz_xyz).tolist())
        published, commands, seeds, orientation_refs = [], [], [], []
        reads, key_reads, arrival_wait = 0, 0, 0

        def read():
            nonlocal reads
            reads += 1
            status["sim_time_s"] += 0.02
            return jnp.zeros(3) if reads == 1 else jnp.full(3, 99.0)

        def start_closing():
            commands.append("close")
            status["closing"] = True

        def publish(_names, cfg):
            status["hand_m"] = np.asarray(root @ cfg)
            published.append(np.array(cfg))
            return True

        def key():
            nonlocal key_reads, arrival_wait
            key_reads += 1
            if key_reads > 100:
                self.fail("Controller did not complete the transfer")
            if published and np.linalg.norm(published[-1] - goal) < 1e-6:
                arrival_wait += 1
                if arrival_wait <= 3:
                    self.assertFalse(status["closing"])
                    return ""
                return "q" if status["closing"] else "c"
            return ""

        def fk(cfg):
            positions = jnp.vstack([cfg, jnp.array(supports)])
            rotations = jnp.tile(jnp.array([1.0, 0, 0, 0]), (4, 1))
            rotations = rotations.at[0].set(jaxlie.SO3.from_z_radians(cfg[0]).wxyz)
            return jnp.hstack([rotations, positions])

        def solve(_robot, _indices, targets, _orientations, cfg, _base):
            seeds.append(np.array(cfg))
            orientation_refs.append(np.array(_orientations))
            return jnp.array(targets[0]), jaxlie.SE3.identity(), jnp.array(5), {"total_objective": jnp.array(0.0)}

        bridge = SimpleNamespace(read=read, grasp_status=lambda: dict(status),
                                 start_closing=start_closing, publish=publish,
                                 state=(SimpleNamespace(name=["x", "y", "z"], position=[99]*3), 0))
        robot = SimpleNamespace(forward_kinematics=fk, joints=SimpleNamespace(names=["x", "y", "z"], get_full_config=lambda cfg: cfg))
        viewer = Mock()
        run = module["run_controller"]
        settings = dict(durations_s=[0.1, 0.1, 0.1], waypoint_tolerance_m=0.02, tracking_tolerance_m=0.05)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(run.__globals__, {"solve_ik": solve}), patch("time.sleep"), contextlib.redirect_stdout(io.StringIO()):
                run(robot, jnp.arange(4), ["x", "y", "z"], jnp.arange(3), bridge, SimpleNamespace(read=key),
                    0.15, settings, metrics_dir=Path(directory), viewer=viewer)
            summaries = list(Path(directory).glob("*/summary.json"))
            self.assertEqual(len(summaries), 1)
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(summary["outcome"], "completed")
            np.testing.assert_allclose(summary["settings"]["target_world_m"], root @ goal, rtol=5e-4, atol=5e-5)
            self.assertEqual(summary["average_solver_steps"], 5)
            self.assertLessEqual(summary["ik_calls"], len(seeds) - 1)  # Warmup and holding excluded.
            self.assertEqual(summary["rl_wrist"]["valid_samples"], summary["ik_calls"])
            self.assertTrue((summaries[0].parent / "metrics.png").is_file())
        self.assertEqual(commands, ["close"])
        self.assertTrue(status["closing"])
        np.testing.assert_allclose(list(viewer.publish.call_args.args[0].values()), goal, atol=1e-6)
        self.assertIs(viewer.publish.call_args.args[2], bridge.state[0])
        self.assertEqual(viewer.publish.call_args.args[3], status["root_pose_wxyz_xyz"])
        np.testing.assert_allclose(viewer.publish.call_args.args[1].wxyz_xyz, root.wxyz_xyz, atol=1e-6)
        # Support references stay latched.
        np.testing.assert_allclose(np.array(orientation_refs)[:, 1:], np.tile([1, 0, 0, 0], (len(orientation_refs), 3, 1)))
        self.assertLess(np.max(np.abs(seeds)), 1)  # Never reseeded from the 99-valued feedback.
        samples = np.array(published)
        offset = np.asarray(root.rotation().inverse() @ np.array([0., 0.15, 0.]))
        for waypoint in (offset, goal + offset, goal):
            self.assertLess(np.linalg.norm(samples - waypoint, axis=1).min(), 1e-6)


if __name__ == "__main__":
    unittest.main()
