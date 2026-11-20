"""Offline checks for Quest teleoperation: target mapping, pair handling and attachment release."""

from pathlib import Path
import runpy
import sys
import time
import unittest
from unittest.mock import patch

import jaxlie
import numpy as np
import zmq

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
with patch.object(sys, "path", [str(EXAMPLES), *sys.path]):
    TELEOP = runpy.run_path(str(EXAMPLES / "helios_teleop.py"))


def pose(xyz, rpy=(0.0, 0.0, 0.0)):
    return jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_rpy_radians(*rpy), np.asarray(xyz, float))


def message(se3):
    wxyz, xyz = np.asarray(se3.wxyz_xyz[:4]), np.asarray(se3.wxyz_xyz[4:])
    return [*xyz, *wxyz[1:], wxyz[0]]  # ORBIT_Teleop order: x y z qx qy qz qw.


class TargetTests(unittest.TestCase):
    def test_palm_follows_operator_motion_from_its_activation_pose(self):
        palm, start = pose([0.3, -0.2, -0.1], (0.1, 0.2, 0.3)), pose([1.0, 2.0, 3.0], (0.5, -0.4, 0.2))
        np.testing.assert_allclose(TELEOP["teleop_target"](palm, start, start).wxyz_xyz, palm.wxyz_xyz, atol=1e-6)
        # Root-frame motion of the operator frame moves the palm the same way.
        motion = pose([0.02, -0.03, 0.04], (0.0, 0.0, 0.3))
        moved = jaxlie.SE3.from_rotation_and_translation(motion.rotation() @ start.rotation(),
                                                         start.translation() + motion.translation())
        target = TELEOP["teleop_target"](palm, start, moved)
        np.testing.assert_allclose(target.translation(), palm.translation() + motion.translation(), atol=1e-6)
        np.testing.assert_allclose((motion.rotation() @ palm.rotation()).as_matrix(),
                                   target.rotation().as_matrix(), atol=1e-6)


class IsaacSideTests(unittest.TestCase):
    def test_release_disables_only_that_pairs_attachments_until_close(self):
        import builtins
        from pxr import Sdf, Usd, UsdPhysics
        isaac = runpy.run_path(str(EXAMPLES / "helios_teleop_isaac.py"))
        stage = Usd.Stage.CreateInMemory()
        joints = {arm: UsdPhysics.Joint.Define(stage, f"/D6Joint_{arm}") for arm in ("LU", "RL", "LL")}
        for joint in joints.values():
            joint.CreateJointEnabledAttr(True)
        layer = Sdf.Layer.CreateAnonymous()
        stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
        with patch.object(builtins, "helios_teleop", (stage, layer), create=True):
            isaac["release"](["RL", "LL"])
            self.assertEqual({arm: j.GetJointEnabledAttr().Get() for arm, j in joints.items()},
                             {"LU": True, "RL": False, "LL": False})
            isaac["close"]()  # As when the controller exits.
            self.assertTrue(all(j.GetJointEnabledAttr().Get() for j in joints.values()))
            self.assertIsNone(builtins.helios_teleop)


class FakeBridge:
    def __init__(self, cfg):
        self.cfg, self.inputs, self.published, self.names, self.released = cfg, {}, [], [], []

    def read(self):
        return self.cfg

    def release_attachments(self, arms):
        self.released.append(tuple(arms))

    def publish(self, names, positions):
        self.names.append(list(names))
        self.published.append(np.asarray(positions, dtype=float))
        return True


class FakeStatus:
    def __init__(self):
        self.queue = []

    def recv_json(self, _flags):
        if not self.queue:
            raise zmq.Again()
        return self.queue.pop(0)


@unittest.skipUnless(TELEOP["DEFAULT_URDF"].exists() and TELEOP["DEFAULT_CONSTANTS"].exists(),
                     "Helios URDF or ORBIT_Teleop is not available")
class TeleopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax.numpy as jnp
        cls.jnp = jnp
        cls.m = TELEOP["build_model"](TELEOP["DEFAULT_URDF"], TELEOP["DEFAULT_CONSTANTS"])
        cls.cfg = jnp.zeros(cls.m.robot.joints.num_actuated_joints)

    def test_fingers_map_onto_orbit_teleop_command_slots(self):
        self.assertEqual(len(self.m.fingers), 64)
        order = runpy.run_path(str(TELEOP["DEFAULT_CONSTANTS"]))["HELIOS_JOINT_ORDER"]
        slots = dict(zip(self.m.fingers, self.m.finger_slots))
        self.assertEqual(order[slots["right_m1_mcp"]], "right_ring_mcp")
        self.assertEqual(order[slots["or4l_left_m2_pip"]], "or4l_left_middle_pip")
        self.assertEqual(order[slots["or4r_right_thumb_abd"]], "or4r_right_thumb_abd")

    def test_free_base_moves_while_holding_hands_stay_in_place(self):
        m, jnp = self.m, self.jnp
        base = jaxlie.SE3.identity()
        start = np.array(TELEOP["world_palms"](m, self.cfg, base).wxyz_xyz)
        targets = start.copy()
        targets[0, 4:] += [0.25, 0.0, 0.1]  # RU far beyond a fixed-base reach step.
        active = jnp.array([1.0, 1.0, 0.0, 0.0])  # Upper pair driven, lower pair holds.
        cfg = self.cfg
        for _ in range(30):  # Control ticks; each warm-starts from the last.
            cfg, base = TELEOP["solve_teleop"](m.robot, m.palm_indices, jnp.asarray(targets), active, cfg, base)
        palms = np.asarray(TELEOP["world_palms"](m, cfg, base).translation())
        np.testing.assert_allclose(palms[2:], start[2:, 4:], atol=1e-3)  # RL/LL hold the robot.
        self.assertLess(np.linalg.norm(palms[0] - targets[0, 4:]), 0.002)
        self.assertGreater(np.linalg.norm(np.asarray(base.translation())), 0.01)  # The body moved.

    def test_loop_releases_pairs_once_and_drives_only_their_fingers(self):
        m = self.m
        bridge, status = FakeBridge(self.cfg), FakeStatus()
        operator_cmd = np.arange(m.command_size, dtype=float) / 100.0
        upper, lower = pose([0.3, -0.2, -0.1], (0.3, 0.0, 0.0)), pose([0.3, -0.2, -0.6])
        up = jaxlie.SE3.from_translation(np.array([0.0, 0.0, 0.03]))
        steps = iter([
            ({"active_pair": "upper"}, upper), (None, upper), (None, up @ upper),
            ({"active_pair": "lower"}, lower), (None, lower),
            ({"active_pair": "upper"}, upper),  # Upper again: re-anchored, not released again.
        ])

        class Keys:
            def read(self):
                step = next(steps, None)
                if step is None:
                    return "q"
                if step[0]:
                    status.queue.append(step[0])
                now = time.monotonic()
                bridge.inputs["right"], bridge.inputs["command"] = (message(step[1]), now), (operator_cmd, now)
                return ""

        with patch.object(TELEOP["run_teleop"].__globals__["time"], "sleep"):
            TELEOP["run_teleop"](m, bridge, Keys(), status)
        self.assertEqual(bridge.released, [("RU", "LU"), ("RL", "LL")])
        n = len(m.command_names)
        np.testing.assert_allclose(bridge.published[0][:n], 0.0, atol=1e-5)  # Activation does not jump.
        self.assertGreater(np.abs(bridge.published[2][:n]).max(), 1e-3)  # The upper motion moved joints.
        np.testing.assert_allclose(bridge.published[3][:n], bridge.published[2][:n], atol=1e-4)  # Lower re-anchors.
        upper_fingers = [f for f, p in zip(m.fingers, m.finger_pairs) if p == "upper"]
        self.assertEqual(bridge.names[0][n:], upper_fingers)  # Lower fingers untouched before lower is driven.
        self.assertEqual(bridge.names[3][n:], m.fingers)
        commanded = dict(zip(bridge.names[3], bridge.published[3]))
        for name, slot in zip(m.fingers, m.finger_slots):
            self.assertAlmostEqual(commanded[name], operator_cmd[slot])


if __name__ == "__main__":
    unittest.main()
