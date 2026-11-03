"""Isaac-side attachment control and measured telemetry, installed by the launcher.

All overrides live in an anonymous session layer removed on Stop/process exit.
No edits are written to the saved robot or scene.
"""

import builtins
import json
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

ATTACH_DISTANCE = 0.02


def anchor_position(joint, side, transform_point):
    """Resolve localPos against its body, or world when that relationship is empty."""
    body = getattr(joint, f"GetBody{side}Rel")().GetTargets()
    local = np.array(getattr(joint, f"GetLocalPos{side}Attr")().Get(), dtype=float)
    return transform_point(str(body[0]), local) if body else local


class HeliosAttachment:
    def __init__(self, owner_pid, drive_gains, effort_names, target_m=None):
        import carb
        import omni.kit.app
        import omni.timeline
        import omni.usd
        import rclpy
        from isaacsim.core.experimental.prims import XformPrim
        from isaacsim.core.experimental.utils.backend import use_backend
        from std_msgs.msg import String

        self.timeline = omni.timeline.get_timeline_interface()
        self.stage = omni.usd.get_context().get_stage()
        self.usd_context = omni.usd.get_context()
        self.owner_pid = owner_pid
        self.closed = False
        self.has_played = False
        joints = (UsdPhysics.Joint(p) for p in self.stage.Traverse() if p.GetTypeName() == "PhysicsJoint")
        self.ru, = [j for j in joints if any("or2r_R_Carpals" in str(p) for p in j.GetBody1Rel().GetTargets())]
        self.effort_names = effort_names
        self.articulation = None
        paths = {str(p) for rel in (self.ru.GetBody0Rel(), self.ru.GetBody1Rel()) for p in rel.GetTargets()}
        self.ru_forearm_path = str(self.ru.GetBody1Rel().GetTargets()[0].GetParentPath())
        paths.add(self.ru_forearm_path)
        self.prims = {p: XformPrim(p, reset_xform_op_properties=False) for p in paths}
        self.use_backend = use_backend
        settings = carb.settings.get_settings()
        self.backend = "fabric" if settings.get_as_bool("/app/useFabricSceneDelegate") else "usd"
        if self.backend == "usd" and settings.get("/physics/updateToUsd") is False:
            raise RuntimeError("Enable physics updateToUsd or Fabric Scene Delegate to read live hand poses")
        self.metres = UsdGeom.GetStageMetersPerUnit(self.stage)
        if target_m is not None:
            target_m = np.asarray(target_m, dtype=float)
            if target_m.shape != (3,) or not np.isfinite(target_m).all():
                raise ValueError("target_m must contain three finite body0-local coordinates in metres")
        if not rclpy.ok():
            rclpy.init(args=[])
        self.node = rclpy.create_node("helios_ru_attachment")
        self.message_type = String
        self.publisher = self.node.create_publisher(String, "/helios/ru_attachment", 1)
        self.wait_clear = False
        self.layer = Sdf.Layer.CreateAnonymous("helios_auto_attach")
        self.stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)
        # A switchable external attachment must not become part of the articulation tree.
        with Usd.EditContext(self.stage, self.layer):
            self.ru.CreateExcludeFromArticulationAttr(True)
            if target_m is not None:
                self.ru.CreateLocalPos0Attr(Gf.Vec3f(*(target_m / self.metres)))
            for prim in self.stage.Traverse():
                if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                    for drive in UsdPhysics.DriveAPI.GetAll(prim):
                        drive.CreateStiffnessAttr(drive_gains["stiffness"])
                        drive.CreateDampingAttr(drive_gains["damping"])
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
        print(f"RU auto-attachment ready: distance < {ATTACH_DISTANCE} m")

    def release(self):
        with Usd.EditContext(self.stage, self.layer):
            self.ru.GetJointEnabledAttr().Set(False)
        self.wait_clear = True  # Leave the radius before allowing reattachment.

    def transform_point(self, path, local):
        # Fabric supplies the current simulated pose even when USD transforms are stale.
        with self.use_backend(self.backend, raise_on_fallback=True):
            p, q = self.prims[path].get_world_poses()
            scale = self.prims[path].get_world_scales()
        p, q, v = p.numpy()[0], q.numpy()[0], local * scale.numpy()[0]
        return p + v + 2.0 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)

    def read_efforts(self):
        """Initialize after Play; report missing measurements as missing, never zero."""
        try:
            if self.articulation is None:
                from isaacsim.core.prims import SingleArticulation
                body = self.ru.GetBody1Rel().GetTargets()[0]
                root = next(p.GetPath() for p in self.stage.Traverse()
                            if p.HasAPI(UsdPhysics.ArticulationRootAPI) and body.HasPrefix(p.GetPath()))
                articulation = SingleArticulation(str(root), reset_xform_properties=False)
                articulation.initialize()
                self.effort_indices = np.array([articulation.get_dof_index(name) for name in self.effort_names], dtype=int)
                self.articulation = articulation
            efforts = self.articulation.get_measured_joint_efforts(joint_indices=self.effort_indices)
            if efforts is None:
                raise ValueError("Isaac returned no measured joint efforts")
            values = np.asarray(efforts).reshape(-1)
            if len(values) != len(self.effort_names):
                raise ValueError("Unexpected measured-effort count")
            return {name: float(value) for name, value in zip(self.effort_names, values) if np.isfinite(value)}
        except Exception:
            return {}

    def read_wrist_wrench(self, joint):
        """Direct transmitted wrist wrench, not a D6 force estimate."""
        try:
            # Force rows use JOINT indices + 1, not DOF indices.
            index = self.articulation._articulation_view._metadata.joint_indices[joint] + 1
            reading = self.articulation.get_measured_joint_forces(
                joint_indices=np.array([index], dtype=int)
            )
            wrench = np.asarray(reading, dtype=float).reshape(-1)
            if wrench.shape != (6,) or not np.isfinite(wrench).all():
                raise ValueError("Invalid wrist wrench")
            return wrench.tolist()
        except Exception:
            return None

    def read_ru_roll(self):
        try:
            with self.use_backend(self.backend, raise_on_fallback=True):
                _, orientation = self.prims[self.ru_forearm_path].get_world_poses()
            w, x, y, z = orientation.numpy()[0]
            # World Y/Z components of the forearm's +Z direction, before wrist rotation.
            roll = np.arctan2(1 - 2 * (x * x + y * y), abs(2 * (y * z - w * x)) + 1e-8)
            return float(roll) if np.isfinite(roll) else None
        except Exception:
            return None

    def read_root_pose(self):
        try:
            position, orientation = self.articulation.get_world_pose()
            pose = np.concatenate((orientation, np.asarray(position) * self.metres))
            return pose.tolist() if pose.shape == (7,) and np.isfinite(pose).all() else None
        except Exception:
            return None

    def update(self, _event):
        if self.closed:
            return
        try:
            os.kill(self.owner_pid, 0)
        except ProcessLookupError:
            self.close()
            return
        if self.usd_context.get_stage() != self.stage or (self.has_played and self.timeline.is_stopped()):
            self.close()
            return
        if not self.timeline.is_playing():
            return
        self.has_played = True
        try:
            target = anchor_position(self.ru, 0, self.transform_point) * self.metres
            hand = anchor_position(self.ru, 1, self.transform_point) * self.metres
            distance = float(np.linalg.norm(hand - target))
            if not np.isfinite(distance):
                raise ValueError("Nonfinite attachment positions")
            attached = bool(self.ru.GetJointEnabledAttr().Get())
            if distance >= ATTACH_DISTANCE:
                self.wait_clear = False
            if not self.wait_clear and not attached and distance < ATTACH_DISTANCE:
                with Usd.EditContext(self.stage, self.layer):
                    self.ru.GetJointEnabledAttr().Set(True)
                attached = True
                print(f"RU attached at {distance * 100:.2f} cm")
            self.publisher.publish(self.message_type(data=json.dumps({
                "attached": attached, "target_m": target.tolist(),
                "hand_m": hand.tolist(), "sim_time_s": self.timeline.get_current_time(),
                "efforts_nm": self.read_efforts(),
                "rl_wrist_wrench": self.read_wrist_wrench("or4r_right_wrist"),
                "lu_wrist_wrench": self.read_wrist_wrench("left_wrist"),
                "ll_wrist_wrench": self.read_wrist_wrench("or4l_left_wrist"),
                "root_pose_wxyz_xyz": self.read_root_pose(),
                "ru_roll_rad": self.read_ru_roll(),
            })))
        except Exception as error:
            print(f"RU auto-attach stopped: {error}")
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.subscription.unsubscribe()
        self.node.destroy_node()
        layers = self.stage.GetSessionLayer().subLayerPaths
        if self.layer.identifier in layers:
            layers.remove(self.layer.identifier)
        print("RU auto-attach stopped; temporary joint overrides removed")


def start_attachment(owner_pid, drive_gains, effort_names, target_m=None):
    previous = getattr(builtins, "helios_attachment", None)
    if previous is not None:
        previous.close()
    builtins.helios_attachment = HeliosAttachment(owner_pid, drive_gains, effort_names, target_m)
    return builtins.helios_attachment
