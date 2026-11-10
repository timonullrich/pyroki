"""Offline checks that helios.yaml actuators resolve and author like Isaac Lab."""

from pathlib import Path
import runpy
import unittest

import numpy as np
import yaml

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SCENE = Path(__file__).resolve().parents[3] / "Thesis/assests/scene_helios_v0.usd"
GRASP = runpy.run_path(str(EXAMPLES / "helios_grasp_isaac.py"))
resolve = GRASP["resolve_actuators"]
ACTUATORS = yaml.safe_load((EXAMPLES / "helios.yaml").read_text())["actuators"]
FINGERS = [f"{f}_{j}" for f in ("pinky", "m1", "m2", "index") for j in ("abd", "mcp", "pip")]
FINGERS += [f"thumb_{j}" for j in ("mcp", "abd", "pip", "dip")]
JOINTS = [f"{j}_{arm}" for arm in ("RU", "LU", "RL", "LL")
          for j in ("Shoulder1", "Shoulder2", "Shoulder3", "Ua", "Rcj1", "Rcj2")]
JOINTS += ["Yaw_HIP", "Roll_HIP"]
JOINTS += [f"{prefix}{side}_{name}" for prefix, side in (("", "right"), ("", "left"), ("or4r_", "right"), ("or4l_", "left"))
           for name in ("wrist", *FINGERS)]


class ActuatorTests(unittest.TestCase):
    def test_grasp_uses_only_ru_closing_joints_with_configured_pd_drives(self):
        values = resolve(JOINTS, ACTUATORS)
        names = list(GRASP["RU_CLOSE_DIRECTIONS"])
        self.assertEqual(len(names), 12)
        self.assertTrue(all(name.startswith("right_") for name in names))
        self.assertNotIn("right_wrist", names)
        self.assertNotIn("right_index_abd", names)
        GRASP["check_grasp_drives"](values)  # helios.yaml configures every closing drive.
        for param in ("stiffness", "damping", "joint_effort_limit"):
            for bad in (None, 0, -1, np.inf, np.nan):
                broken = {name: dict(params) for name, params in values.items()}
                broken[names[0]][param] = bad
                with self.subTest(param=param, value=bad), self.assertRaisesRegex(ValueError, f"positive finite {param}"):
                    GRASP["check_grasp_drives"](broken)

    def test_closing_targets_are_closing_side_joint_limits_in_radians(self):
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.CreateInMemory()
        driven = {}
        for i, name in enumerate(GRASP["RU_CLOSE_DIRECTIONS"]):
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/{name}")
            joint.CreateLowerLimitAttr(-15.0)
            joint.CreateUpperLimitAttr(30.0 + i)
            driven[name] = joint.GetPrim()
        targets = GRASP["closed_targets"](driven)
        np.testing.assert_allclose(targets, np.radians(30.0 + np.arange(len(driven))))
        UsdPhysics.RevoluteJoint(driven["right_index_pip"]).GetUpperLimitAttr().Set(float("inf"))
        with self.assertRaisesRegex(ValueError, "right_index_pip.*finite closing-side"):
            GRASP["closed_targets"](driven)

    def test_legacy_ru_joint_is_optional_and_masking_preserves_supports(self):
        from pxr import Sdf, Usd, UsdPhysics
        stage = Usd.Stage.CreateInMemory()
        joints = {arm: UsdPhysics.Joint.Define(stage, f"/D6Joint_{arm}").GetPrim()
                  for arm in ("RU", "LU", "RL", "LL")}
        layer = Sdf.Layer.CreateAnonymous()
        stage.GetSessionLayer().subLayerPaths.insert(0, layer.identifier)
        with Usd.EditContext(stage, layer):
            GRASP["remove_ru_attachment"](stage)
            # Repeating this with no active RU joint is supported.
            GRASP["remove_ru_attachment"](stage)
        self.assertFalse(joints["RU"].IsActive())
        self.assertTrue(all(joints[arm].IsActive() for arm in ("LU", "RL", "LL")))
        stage.GetSessionLayer().subLayerPaths.remove(layer.identifier)
        self.assertTrue(joints["RU"].IsActive())

    def test_every_scene_joint_gets_a_complete_configuration(self):
        names = JOINTS
        if SCENE.exists():
            from pxr import Usd, UsdPhysics
            stage = Usd.Stage.Open(str(SCENE))
            names = [p.GetName() for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
            self.assertCountEqual(names, JOINTS)
        values = resolve(names, ACTUATORS)
        self.assertCountEqual(values, names)
        body = {"stiffness": 71.1, "damping": 7.54, "joint_effort_limit": 100.0, "joint_velocity_limit": 5.0}
        self.assertEqual(values["Shoulder3_LL"], body)
        self.assertEqual(values["Roll_HIP"], body)
        hand_fields = {"stiffness", "damping", "friction", "armature", "joint_effort_limit"}
        for name in JOINTS[26:]:  # Wrists and fingers.
            self.assertEqual(set(values[name]), hand_fields, name)
        self.assertEqual(values["or4r_right_m2_pip"]["stiffness"], 14.4)
        self.assertEqual(values["left_thumb_abd"]["joint_effort_limit"], 0.58)
        self.assertEqual(values["or4l_left_wrist"]["friction"], 0.105)
        self.assertEqual(values["right_thumb_dip"]["damping"], 0.42)

    def test_invalid_configurations_are_rejected(self):
        cases = {
            "matches no joint": {"a": {"joint_names_expr": ["typo"]}},
            "more than one actuator group": {"a": {"joint_names_expr": ["x.*"]}, "b": {"joint_names_expr": ["x1"]}},
            "more than one pattern": {"a": {"joint_names_expr": ["x.*"], "damping": {"x.*": 1, "x1": 2}}},
            "unknown fields": {"a": {"joint_names_expr": ["x1"], "stifness": 1}},
            "nonnegative": {"a": {"joint_names_expr": ["x1"], "stiffness": -1}},
        }
        for message, actuators in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                resolve(["x1", "x2"], actuators)

    def test_usd_values_use_degree_units_for_revolute_joints(self):
        from pxr import Usd, UsdPhysics
        stage = Usd.Stage.CreateInMemory()
        prim = UsdPhysics.RevoluteJoint.Define(stage, "/j").GetPrim()
        params = {"stiffness": 14.4, "damping": 2.05, "friction": 0.039, "armature": 0.0,
                  "joint_effort_limit": 1.29, "joint_velocity_limit": 5.0}
        GRASP["apply_actuator"](prim, params)
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        self.assertAlmostEqual(drive.GetStiffnessAttr().Get(), 14.4 * np.pi / 180, places=6)
        self.assertAlmostEqual(drive.GetDampingAttr().Get(), 2.05 * np.pi / 180, places=6)
        self.assertAlmostEqual(drive.GetMaxForceAttr().Get(), 1.29, places=6)
        self.assertAlmostEqual(prim.GetAttribute("physxJoint:maxJointVelocity").Get(), 5.0 * 180 / np.pi, places=3)
        self.assertAlmostEqual(prim.GetAttribute("physxJoint:jointFriction").Get(), 0.039, places=6)
        self.assertEqual(prim.GetAttribute("physxJoint:armature").Get(), 0.0)
        self.assertIn("PhysxJointAPI", prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems())


if __name__ == "__main__":
    unittest.main()
