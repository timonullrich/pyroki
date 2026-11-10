"""Offline checks that the applied initial pose opens the support thumbs clear of the bars."""

import contextlib
import json
import math
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
APPLY = runpy.run_path(str(ROOT / "examples/helios_apply_initial_pose.py"))
POSE = json.loads((ROOT / "helios_initial_pose.json").read_text())
URDF = (ROOT.parent / "Thesis/third_party/ORBIT_urdf_stack/helios_description/urdf"
        / "helios_quadmanual_flat_inertias.urdf")
BAR_AXES_XY = [np.array([0.3 * i, 0.265]) for i in range(4)]  # scene_helios_v0.usd
BAR_RADIUS_M = 0.015
HAND_PREFIXES = {"LU": "or2l_", "RL": "or4r_", "LL": "or4l_"}


def fake_physx_schema():
    """usd-core lacks PhysxSchema; author the same JointStateAPI attribute names."""
    from pxr import Sdf

    class State:
        def __init__(self, prim):
            self.prim = prim

        def CreatePositionAttr(self, value):
            self.prim.CreateAttribute("state:angular:physics:position", Sdf.ValueTypeNames.Float).Set(value)

        def CreateVelocityAttr(self, value):
            self.prim.CreateAttribute("state:angular:physics:velocity", Sdf.ValueTypeNames.Float).Set(value)

    return SimpleNamespace(JointStateAPI=SimpleNamespace(Apply=lambda prim, _axis: State(prim)))


@contextlib.contextmanager
def physx_schema():
    import pxr
    with patch.object(pxr, "PhysxSchema", fake_physx_schema(), create=True):
        yield


class ApplyPoseTests(unittest.TestCase):
    def make_stage(self, directory, pose):
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
        stage = Usd.Stage.CreateNew(str(Path(directory) / "scene.usda"))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)  # Like scene_helios_v0.usd.
        UsdGeom.Xform.Define(stage, "/World/robot/Geometry/world")
        names = [*pose["joint_positions_rad"], *APPLY["support_thumb_positions"]()]
        for name in names:
            joint = UsdPhysics.RevoluteJoint.Define(stage, f"/World/robot/Physics/{name}").GetPrim()
            UsdPhysics.DriveAPI.Apply(joint, "angular")
        for arm in ("LU", "RL", "LL"):
            joint = UsdPhysics.Joint.Define(stage, f"/World/robot/Physics/D6Joint_{arm}")
            joint.CreateLocalPos0Attr(Gf.Vec3f(*pose["attachments"][arm]["target_world_m"]))
        stage.GetRootLayer().Save()
        return stage

    def test_support_thumbs_get_opened_state_and_matching_drive_target(self):
        from pxr import UsdPhysics
        with tempfile.TemporaryDirectory() as directory:
            pose = {**POSE, "base_prim": "/World/robot/Geometry/world"}
            stage = self.make_stage(directory, pose)
            pose["scene"] = stage.GetRootLayer().realPath
            with physx_schema():
                APPLY["apply_pose"](stage, pose)
            for prefix in APPLY["SUPPORT_THUMB_PREFIXES"].values():
                for joint, degrees in APPLY["SUPPORT_THUMB_DEG"].items():
                    prim = stage.GetPrimAtPath(f"/World/robot/Physics/{prefix}{joint}")
                    self.assertAlmostEqual(prim.GetAttribute("state:angular:physics:position").Get(), degrees, places=4)
                    target = UsdPhysics.DriveAPI.Get(prim, "angular").GetTargetPositionAttr().Get()
                    self.assertAlmostEqual(target, degrees, places=4)
            # The arm joints from the JSON are still applied as before.
            name, radians = next(iter(POSE["joint_positions_rad"].items()))
            prim = stage.GetPrimAtPath(f"/World/robot/Physics/{name}")
            self.assertAlmostEqual(prim.GetAttribute("state:angular:physics:position").Get(), math.degrees(radians), places=4)

    def test_pose_that_already_sets_a_support_thumb_is_rejected(self):
        pose = {**POSE, "joint_positions_rad": {**POSE["joint_positions_rad"], "left_thumb_abd": 0.0}}
        stage = SimpleNamespace(GetRootLayer=lambda: SimpleNamespace(realPath=POSE["scene"]),
                                GetPrimAtPath=lambda _path: SimpleNamespace(IsA=lambda _type: True))
        with physx_schema(), self.assertRaisesRegex(ValueError, "support thumb"):
            APPLY["apply_pose"](stage, pose)


@unittest.skipUnless(URDF.exists(), "Helios URDF and meshes are not available")
class ThumbClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import trimesh
        import yourdfpy
        from scipy.spatial.transform import Rotation
        cls.trimesh = trimesh
        cls.urdf = yourdfpy.URDF.load(str(URDF), load_meshes=False)
        cls.base = np.eye(4)
        cls.base[:3, 3] = POSE["base_position_world_m"]
        w, x, y, z = POSE["base_orientation_world_wxyz"]
        cls.base[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        cls.points = {}
        for link in cls.urdf.robot.links:
            if not link.name.startswith(tuple(HAND_PREFIXES.values())):
                continue
            cls.points[link.name] = []
            for geometry in link.collisions or link.visuals:
                if geometry.geometry.mesh is None:
                    continue
                path = URDF.parent.parent / geometry.geometry.mesh.filename.replace("package://helios_description/", "")
                mesh = trimesh.load(path, force="mesh")
                if geometry.geometry.mesh.scale is not None:
                    mesh.apply_scale(geometry.geometry.mesh.scale)
                hull = mesh.convex_hull  # PhysX colliders are convex hulls of these meshes.
                origin = geometry.origin if geometry.origin is not None else np.eye(4)
                cls.points[link.name].append((np.vstack([hull.vertices, hull.sample(3000, seed=0)]), origin))

    def clearance_mm(self, extra):
        cfg = {name: 0.0 for name in self.urdf.actuated_joint_names}
        cfg.update({k: v for k, v in POSE["joint_positions_rad"].items() if k in cfg})
        cfg.update(extra)
        self.urdf.update_cfg(cfg)
        result = {}
        for arm, prefix in HAND_PREFIXES.items():
            nearest = np.inf
            for link, parts in self.points.items():
                if not link.startswith(prefix):
                    continue
                for points, origin in parts:
                    world = self.trimesh.transform_points(points, self.base @ self.urdf.get_transform(link) @ origin)
                    nearest = min(nearest, min(np.linalg.norm(world[:, :2] - axis, axis=1).min() for axis in BAR_AXES_XY))
            result[arm] = (nearest - BAR_RADIUS_M) * 1000
        return result

    def test_zero_thumbs_penetrate_and_opened_thumbs_clear_the_bars(self):
        closed = self.clearance_mm({})
        self.assertTrue(all(value < -4.0 for value in closed.values()), closed)
        opened = self.clearance_mm(APPLY["support_thumb_positions"]())
        self.assertTrue(all(value > 5.0 for value in opened.values()), opened)


if __name__ == "__main__":
    unittest.main()
