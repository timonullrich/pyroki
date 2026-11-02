"""Started from the terminal by helios_retargetiing.py --ros --auto-attach.

Enable isaacsim.code_editor.python_server in Isaac first; start while STOPPED,
then press Play. Alternatively, run this file in Isaac's Script Editor.

Requires the ROS 2 Bridge and PhysX. Uses the live scene's D6 joint anchors,
including mesh scale. Keeps attachment angular settings unchanged (rotation must
be free). RU is enabled only after the controller arms docking and its two
anchors are strictly less than 0.02 m apart.
The launcher can temporarily override RU localPos0 with target_m (metres).
With no body0, localPos0 is in world coordinates. Joint changes use a private
temporary layer that is removed on shutdown; the saved asset is not modified.
The terminal launcher also supplies helios_gains.yaml for all existing revolute
and prismatic motor drives in the scene, using the same temporary layer.

Publishes /helios/ru_attachment (std_msgs/String JSON): attached, distance_m,
anchors_m (world-space body0 anchors in RU, LU, RL, LL order), hand_m,
sim_time_s, armed, command_id, and efforts_nm from get_measured_joint_efforts().
rl_wrist_wrench contains [Fx, Fy, Fz, Tx, Ty, Tz] from the lower-right wrist's
get_measured_joint_forces() reading, in its child joint frame (N and N m).
root_pose_wxyz_xyz is the measured articulation root pose in world metres.
ru_roll_rad is the RU hand pointing tilt in the global Y-Z plane (0 = horizontal).
The launcher supplies the non-finger motor names to measure. Receives commands on
/helios/ru_attachment_command; release must be acknowledged before motion.
The IK subscribes automatically; no additional OmniGraph nodes are needed.
Stop, close(), or exit of the launching terminal process removes the helper and
its temporary edits. Restart with --auto-attach after stopping the simulation.
"""

import builtins
import json
import os
import time

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

PALMS = ("or2r_R_Carpals", "or2l_L_Carpals", "or4r_R_Carpals", "or4l_L_Carpals")
ATTACH_DISTANCE = 0.02


def find_attachment_joints(stage):
    joints = [UsdPhysics.Joint(p) for p in stage.Traverse() if p.GetTypeName() == "PhysicsJoint"]
    result = []
    for palm in PALMS:
        matches = [j for j in joints if any(palm in str(p) for p in j.GetBody1Rel().GetTargets())]
        if len(matches) != 1:
            raise ValueError(f"Expected one D6 attachment for {palm}, found {len(matches)}")
        result.append(matches[0])
    return result


def anchor_position(joint, side, transform_point):
    """Resolve localPos against its body, or world when that relationship is empty."""
    body = getattr(joint, f"GetBody{side}Rel")().GetTargets()
    local = np.array(getattr(joint, f"GetLocalPos{side}Attr")().Get(), dtype=float)
    return transform_point(str(body[0]), local) if body else local


class HeliosAttachment:
    def __init__(self, owner_pid=None, drive_gains=None, effort_names=None, target_m=None):
        import carb
        import omni.kit.app
        import omni.timeline
        import omni.usd
        import rclpy
        from isaacsim.core.experimental.prims import XformPrim
        from isaacsim.core.experimental.utils.backend import use_backend
        from std_msgs.msg import String

        self.timeline = omni.timeline.get_timeline_interface()
        if not self.timeline.is_stopped():
            raise RuntimeError("Stop Isaac before running this script, then press Play")
        self.stage = omni.usd.get_context().get_stage()
        self.usd_context = omni.usd.get_context()
        self.owner_pid = owner_pid
        self.closed = False
        self.has_played = False
        self.joints = find_attachment_joints(self.stage)
        self.ru = self.joints[0]
        self.effort_names = effort_names or []
        self.articulation = None
        self.effort_warning = False
        self.wrist_force_index = None
        self.wrist_warning = False
        paths = {str(p) for j in self.joints for rel in (j.GetBody0Rel(), j.GetBody1Rel()) for p in rel.GetTargets()}
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
        self.ros = rclpy
        self.message_type = String
        self.publisher = self.node.create_publisher(String, "/helios/ru_attachment", 1)
        self.node.create_subscription(String, "/helios/ru_attachment_command", self.receive_command, 1)
        self.armed = False
        self.command_id = None
        self.last_update = 0.0
        self.layer = Sdf.Layer.CreateAnonymous("helios_auto_attach")
        self.stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)
        # A switchable external attachment must not become part of the articulation tree.
        with Usd.EditContext(self.stage, self.layer):
            self.ru.CreateExcludeFromArticulationAttr(True)
            if target_m is not None:
                self.ru.CreateLocalPos0Attr(Gf.Vec3f(*(target_m / self.metres)))
            if drive_gains is not None:
                count = 0
                for prim in self.stage.Traverse():
                    if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                        for drive in UsdPhysics.DriveAPI.GetAll(prim):
                            drive.CreateStiffnessAttr(drive_gains["stiffness"])
                            drive.CreateDampingAttr(drive_gains["damping"])
                            count += 1
                print(f"Applied motor gains to {count} drives: {drive_gains}")
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
        print(f"RU attachment ready: {self.ru.GetPath()}; waiting for controller commands")

    def receive_command(self, msg):
        command = json.loads(msg.data)
        command_id = command["id"]
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("Attachment command needs a nonempty id")
        if command["action"] == "release" and command_id != self.command_id:
            self.armed = False
            with Usd.EditContext(self.stage, self.layer):
                self.ru.GetJointEnabledAttr().Set(False)
            self.command_id = command_id
        elif command["action"] == "arm" and command_id == self.command_id:
            self.armed = True

    def transform_point(self, path, local):
        # Fabric supplies the current simulated pose even when USD transforms are stale.
        with self.use_backend(self.backend, raise_on_fallback=True):
            p, q = self.prims[path].get_world_poses()
            scale = self.prims[path].get_world_scales()
        p, q, v = p.numpy()[0], q.numpy()[0], local * scale.numpy()[0]
        return p + v + 2.0 * np.cross(q[1:], np.cross(q[1:], v) + q[0] * v)

    def read_efforts(self):
        """Initialize after Play; report missing measurements as missing, never zero."""
        if not self.effort_names:
            return {}
        try:
            if self.articulation is None:
                from isaacsim.core.prims import SingleArticulation

                roots = [p.GetPath() for p in self.stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
                body = self.ru.GetBody1Rel().GetTargets()[0]
                # Find the nearest enclosing robot subtree with one articulation.
                matches = []
                for path in reversed(body.GetPrefixes()):
                    matches = [root for root in roots if root.HasPrefix(path)]
                    if matches:
                        break
                if len(matches) != 1:
                    raise ValueError(f"Cannot identify Helios articulation: {matches}")
                articulation = SingleArticulation(str(matches[0]), reset_xform_properties=False)
                articulation.initialize()
                self.effort_indices = np.array([
                    i for i, name in enumerate(articulation.dof_names) if name in self.effort_names
                ], dtype=int)
                self.measured_names = [articulation.dof_names[i] for i in self.effort_indices]
                if not self.measured_names:
                    raise ValueError("No requested motor names match the Isaac articulation")
                self.articulation = articulation
            efforts = self.articulation.get_measured_joint_efforts(joint_indices=self.effort_indices)
            if efforts is None:
                raise ValueError("Isaac returned no measured joint efforts")
            values = np.asarray(efforts).reshape(-1)
            if len(values) != len(self.measured_names):
                raise ValueError("Unexpected measured-effort count")
            return {name: float(value) for name, value in zip(self.measured_names, values) if np.isfinite(value)}
        except Exception as error:
            if not self.effort_warning:
                print(f"Effort measurements unavailable (will retry): {error}")
                self.effort_warning = True
            return {}

    def read_rl_wrist_wrench(self):
        """Direct transmitted wrench at or4r_right_wrist, not a D6 force estimate."""
        if self.articulation is None:
            return None  # read_efforts initializes the shared articulation after Play.
        try:
            if self.wrist_force_index is None:
                metadata = self.articulation._articulation_view._metadata
                # Force rows use articulation JOINT indices + 1 (row 0 is the
                # base). A DOF index would select the wrong row with fixed joints.
                self.wrist_force_index = metadata.joint_indices["or4r_right_wrist"] + 1
            reading = self.articulation.get_measured_joint_forces(
                joint_indices=np.array([self.wrist_force_index], dtype=int)
            )
            if reading is None:
                raise ValueError("Isaac returned no wrist wrench")
            wrench = np.asarray(reading, dtype=float).reshape(-1)
            if wrench.shape != (6,) or not np.isfinite(wrench).all():
                raise ValueError("Invalid wrist wrench")
            return wrench.tolist()
        except Exception as error:
            if not self.wrist_warning:
                print(f"Lower-right wrist wrench unavailable (will retry): {error}")
                self.wrist_warning = True
            return None

    def read_ru_roll(self):
        try:
            palm = str(self.ru.GetBody1Rel().GetTargets()[0])
            with self.use_backend(self.backend, raise_on_fallback=True):
                _, orientation = self.prims[palm].get_world_poses()
            w, x, y, z = orientation.numpy()[0]
            # World Y/Z components of the hand's local +Z (finger) direction.
            roll = np.arctan2(1 - 2 * (x * x + y * y), abs(2 * (y * z - w * x)) + 1e-8)
            return float(roll) if np.isfinite(roll) else None
        except Exception:
            return None

    def read_root_pose(self):
        if self.articulation is None:
            return None
        try:
            position, orientation = self.articulation.get_world_pose()
            pose = np.concatenate((orientation, np.asarray(position) * self.metres))
            return pose.tolist() if pose.shape == (7,) and np.isfinite(pose).all() else None
        except Exception:
            return None

    def update(self, _event):
        if self.closed:
            return
        if self.owner_pid is not None:
            try:
                os.kill(self.owner_pid, 0)
            except ProcessLookupError:
                self.close()
                return
        if self.usd_context.get_stage() != self.stage:
            self.close()
            return
        if self.timeline.is_stopped():
            if self.has_played:
                self.close()
            return
        if not self.timeline.is_playing():
            return
        self.has_played = True
        if time.monotonic() - self.last_update < 0.02:
            return
        self.last_update = time.monotonic()
        try:
            for _ in range(4):
                self.ros.spin_once(self.node, timeout_sec=0.0)
            anchors = np.array([anchor_position(j, 0, self.transform_point) for j in self.joints]) * self.metres
            hand = anchor_position(self.ru, 1, self.transform_point) * self.metres
            distance = float(np.linalg.norm(hand - anchors[0]))
            if not np.isfinite(anchors).all() or not np.isfinite(distance):
                raise ValueError("Nonfinite attachment positions")
            attached = bool(self.ru.GetJointEnabledAttr().Get())
            if self.armed and not attached and distance < ATTACH_DISTANCE:
                with Usd.EditContext(self.stage, self.layer):
                    self.ru.GetJointEnabledAttr().Set(True)
                attached = True
                print(f"RU attached at {distance * 100:.2f} cm")
            self.publisher.publish(self.message_type(data=json.dumps({
                "attached": attached, "distance_m": distance, "anchors_m": anchors.tolist(),
                "hand_m": hand.tolist(), "sim_time_s": self.timeline.get_current_time(),
                "armed": self.armed, "command_id": self.command_id,
                "efforts_nm": self.read_efforts(),
                "rl_wrist_wrench": self.read_rl_wrist_wrench(),
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
        self.subscription = None
        self.node.destroy_node()
        layers = self.stage.GetSessionLayer().subLayerPaths
        if self.layer.identifier in layers:
            layers.remove(self.layer.identifier)
        print("RU auto-attach stopped; temporary joint overrides removed")


def start_attachment(owner_pid=None, drive_gains=None, effort_names=None, target_m=None):
    previous = getattr(builtins, "helios_attachment", None)
    if previous is not None:
        previous.close()
    builtins.helios_attachment = HeliosAttachment(owner_pid, drive_gains, effort_names, target_m)
    return builtins.helios_attachment


if __name__ == "__main__":
    if "helios_attachment" in globals():
        helios_attachment.close()
    helios_attachment = start_attachment()
