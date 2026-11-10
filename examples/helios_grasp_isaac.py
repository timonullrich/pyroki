"""Isaac-side manual RU grasp control and measured telemetry, installed by the launcher.

USD overrides live in an anonymous session layer removed on Stop/process exit.
Closing ramps PD position targets; the drives' maxForce caps the grasp torque.
Targets reopen on exit. No edits are written to the saved scene.
"""

import builtins
import json
import os
import re

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics

# Positive flexion closes the RU fingers in the supplied Helios model.
# Thumb basal/opposition joints also close; finger abduction and wrist keep their targets.
RU_CLOSE_DIRECTIONS = {
    **{f"right_{finger}_{joint}": 1.0
       for finger in ("pinky", "m1", "m2", "index") for joint in ("mcp", "pip")},
    **{f"right_thumb_{joint}": 1.0 for joint in ("mcp", "abd", "pip", "dip")},
}
ACTUATOR_PARAMS = ("stiffness", "damping", "friction", "armature", "joint_effort_limit", "joint_velocity_limit")


def resolve_actuators(joint_names, actuators):
    """Per-joint SI values with Isaac Lab ImplicitActuatorCfg matching rules.

    Patterns full-match joint names; every pattern must match something, and no
    joint may belong to two groups or match two patterns of one parameter.
    """
    resolved = {}
    for group, cfg in actuators.items():
        unknown = set(cfg) - {"joint_names_expr", *ACTUATOR_PARAMS}
        if unknown:
            raise ValueError(f"Actuator group {group}: unknown fields {sorted(unknown)}")
        members = []
        for pattern in cfg["joint_names_expr"]:
            matched = [name for name in joint_names if re.fullmatch(pattern, name)]
            if not matched:
                raise ValueError(f"Actuator group {group}: {pattern!r} matches no joint")
            members += [name for name in matched if name not in members]
        for name in members:
            if name in resolved:
                raise ValueError(f"Joint {name} is in more than one actuator group")
            resolved[name] = {}
        for param in ACTUATOR_PARAMS:
            values = cfg.get(param)
            if values is None:
                continue
            for pattern, value in (values if isinstance(values, dict) else {".*": values}).items():
                value = float(value)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"Actuator group {group}: {param} must be finite and nonnegative")
                matched = [name for name in members if re.fullmatch(pattern, name)]
                if not matched:
                    raise ValueError(f"Actuator group {group}: {param} pattern {pattern!r} matches no joint")
                for name in matched:
                    if param in resolved[name]:
                        raise ValueError(f"Joint {name}: {param} matches more than one pattern")
                    resolved[name][param] = value
    return resolved


def apply_actuator(prim, params):
    """Author SI actuator values on a joint; USD angular drives use degrees."""
    angular = prim.IsA(UsdPhysics.RevoluteJoint)
    per_rad = np.pi / 180.0 if angular else 1.0
    drive = UsdPhysics.DriveAPI.Apply(prim, "angular" if angular else "linear")
    physx = {"friction": "jointFriction", "armature": "armature", "joint_velocity_limit": "maxJointVelocity"}
    if physx.keys() & params.keys():
        prim.AddAppliedSchema("PhysxJointAPI")
    for param, value in params.items():
        if param == "stiffness":
            drive.CreateStiffnessAttr(value * per_rad)  # N·m/rad -> N·m/deg
        elif param == "damping":
            drive.CreateDampingAttr(value * per_rad)  # N·m·s/rad -> N·m·s/deg
        elif param == "joint_effort_limit":
            drive.CreateMaxForceAttr(value)  # N·m, no angle unit
        else:
            # rad/s -> deg/s; friction coefficient and armature are not per angle.
            value = value / per_rad if param == "joint_velocity_limit" else value
            prim.CreateAttribute(f"physxJoint:{physx[param]}", Sdf.ValueTypeNames.Float).Set(value)


def remove_ru_attachment(stage):
    """Mask legacy RU D6 joints in the caller's session layer; none is required."""
    for prim in list(stage.Traverse()):
        if prim.GetTypeName() == "PhysicsJoint" and (
            prim.GetName() == "D6Joint_RU" or any(
                "or2r_R_Carpals" in str(path)
                for path in UsdPhysics.Joint(prim).GetBody1Rel().GetTargets()
            )
        ):
            prim.SetActive(False)


def check_grasp_drives(actuator_values):
    """Closing uses the configured implicit PD drives, capped by joint_effort_limit."""
    for name in RU_CLOSE_DIRECTIONS:
        params = actuator_values.get(name, {})
        for param in ("stiffness", "damping", "joint_effort_limit"):
            value = params.get(param)
            if value is None or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name}: grasp requires a positive finite {param}")


def closed_targets(driven):
    """Closing-side joint limit of each RU closing joint, in radians."""
    targets = []
    for name, direction in RU_CLOSE_DIRECTIONS.items():
        joint = UsdPhysics.RevoluteJoint(driven[name])
        limit = (joint.GetUpperLimitAttr() if direction > 0 else joint.GetLowerLimitAttr()).Get()
        if limit is None or not np.isfinite(limit):
            raise ValueError(f"{name}: grasp requires a finite closing-side joint limit")
        targets.append(np.radians(limit))
    return np.asarray(targets)


class HeliosGrasp:
    def __init__(self, owner_pid, actuators, effort_names, target_m, hand_offset_m, closing_speed_rad_s=0.35):
        if not np.isfinite(closing_speed_rad_s) or closing_speed_rad_s <= 0:
            raise ValueError("closing_speed_rad_s must be positive and finite")
        self.closing_speed = float(closing_speed_rad_s)
        self.closing_elapsed_s = 0.0
        self.closing_start = None
        import carb
        import omni.kit.app
        import omni.timeline
        import omni.physx
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
        wrist, = [UsdPhysics.Joint(p) for p in self.stage.Traverse()
                  if p.GetName() == "right_wrist" and p.IsA(UsdPhysics.RevoluteJoint)]
        self.ru_palm_path = str(wrist.GetBody1Rel().GetTargets()[0])
        self.ru_forearm_path = str(wrist.GetBody0Rel().GetTargets()[0])
        self.effort_names = effort_names
        self.articulation = None
        self.closing = False
        self.prims = {p: XformPrim(p, reset_xform_op_properties=False)
                      for p in (self.ru_palm_path, self.ru_forearm_path)}
        self.use_backend = use_backend
        settings = carb.settings.get_settings()
        self.backend = "fabric" if settings.get_as_bool("/app/useFabricSceneDelegate") else "usd"
        if self.backend == "usd" and settings.get("/physics/updateToUsd") is False:
            raise RuntimeError("Enable physics updateToUsd or Fabric Scene Delegate to read live hand poses")
        self.metres = UsdGeom.GetStageMetersPerUnit(self.stage)
        driven = [p for p in self.stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint) or p.IsA(UsdPhysics.PrismaticJoint)]
        if len({p.GetName() for p in driven}) != len(driven):
            raise ValueError("Actuator patterns need unique joint names in the stage")
        driven = {p.GetName(): p for p in driven}
        actuator_values = resolve_actuators(list(driven), actuators)
        unconfigured = sorted(set(driven) - set(actuator_values))
        if unconfigured:
            print(f"Keeping scene drive settings for {len(unconfigured)} joints: {', '.join(unconfigured)}")
        check_grasp_drives(actuator_values)
        self.closed_targets = closed_targets(driven)
        self.target_m = np.asarray(target_m, dtype=float)
        self.hand_offset = np.asarray(hand_offset_m, dtype=float) / self.metres
        for label, value in (("target_m", self.target_m), ("hand_offset_m", self.hand_offset)):
            if value.shape != (3,) or not np.isfinite(value).all():
                raise ValueError(f"{label} must contain three finite coordinates")
        if not rclpy.ok():
            rclpy.init(args=[])
        self.node = rclpy.create_node("helios_ru_grasp")
        self.message_type = String
        self.publisher = self.node.create_publisher(String, "/helios/ru_status", 1)
        self.layer = Sdf.Layer.CreateAnonymous("helios_grasp")
        self.stage.GetSessionLayer().subLayerPaths.insert(0, self.layer.identifier)
        with Usd.EditContext(self.stage, self.layer):
            remove_ru_attachment(self.stage)
            for name, params in actuator_values.items():
                apply_actuator(driven[name], params)
            # Limit the light fingers' physical speed throughout closing and contact.
            for name in RU_CLOSE_DIRECTIONS:
                apply_actuator(driven[name], {"joint_velocity_limit": closing_speed_rad_s})
            # Start the RU fingers open, including any old scene drive targets.
            for name, prim in driven.items():
                if name.startswith("right_") and name != "right_wrist":
                    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                    drive.CreateTargetPositionAttr(0.0)
                    drive.CreateTargetVelocityAttr(0.0)
        self.subscription = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
        self.physics_subscription = omni.physx.get_physx_interface().subscribe_physics_step_events(self.physics_step)
        print(f"Actuator settings applied to {len(actuator_values)} joints")
        print("RU grasp ready; fingers remain open until the terminal closing command")

    def start_closing(self):
        if self.closed or not self.timeline.is_playing():
            raise RuntimeError("Press Play before starting the RU grasp")
        if self.closing:
            return
        self.initialize_articulation()
        positions = self.articulation.get_joint_positions(joint_indices=self.grasp_indices)
        self.closing_start = np.asarray(positions, dtype=float).reshape(-1)
        if self.closing_start.shape != self.closed_targets.shape or not np.isfinite(self.closing_start).all():
            raise RuntimeError("Could not read RU closing joint positions")
        self.closing = True
        self.closing_elapsed_s = 0.0
        self.physics_step(0.0)
        print(f"RU closing at {self.closing_speed:g} rad/s with the configured PD drives; "
              "torque is capped at each joint_effort_limit")

    def set_grasp_targets(self, targets):
        from isaacsim.core.utils.types import ArticulationAction
        self.articulation.apply_action(ArticulationAction(joint_positions=targets, joint_indices=self.grasp_indices))

    def physics_step(self, dt):
        if self.closed or not self.closing:
            return
        try:
            # Targets move toward the closed limits; in contact the implicit drive
            # pushes with stiffness * lag (damped), saturating at maxForce.
            self.closing_elapsed_s += max(float(dt), 0.0)
            travel = min(self.closing_speed * self.closing_elapsed_s, 2 * np.pi)
            step = np.clip(self.closed_targets - self.closing_start, -travel, travel)
            self.set_grasp_targets(self.closing_start + step)
        except Exception as error:
            print(f"RU grasp stopped: {error}")
            self.close()

    def initialize_articulation(self):
        if self.articulation is not None:
            return
        from isaacsim.core.prims import SingleArticulation
        body = Sdf.Path(self.ru_palm_path)
        root = next(p.GetPath() for p in self.stage.Traverse()
                    if p.HasAPI(UsdPhysics.ArticulationRootAPI) and body.HasPrefix(p.GetPath()))
        articulation = SingleArticulation(str(root), reset_xform_properties=False)
        articulation.initialize()
        self.effort_indices = np.array([articulation.get_dof_index(name) for name in self.effort_names], dtype=int)
        self.grasp_indices = np.array([articulation.get_dof_index(name) for name in RU_CLOSE_DIRECTIONS], dtype=int)
        self.articulation = articulation

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
            self.initialize_articulation()
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
            hand = self.transform_point(self.ru_palm_path, self.hand_offset) * self.metres
            if not np.isfinite(hand).all():
                raise ValueError("Nonfinite hand position")
            self.publisher.publish(self.message_type(data=json.dumps({
                "closing": self.closing, "target_m": self.target_m.tolist(),
                "hand_m": hand.tolist(), "sim_time_s": self.timeline.get_current_time(),
                "efforts_nm": self.read_efforts(),
                "rl_wrist_wrench": self.read_wrist_wrench("or4r_right_wrist"),
                "lu_wrist_wrench": self.read_wrist_wrench("left_wrist"),
                "ll_wrist_wrench": self.read_wrist_wrench("or4l_left_wrist"),
                "root_pose_wxyz_xyz": self.read_root_pose(),
                "ru_roll_rad": self.read_ru_roll(),
            })))
        except Exception as error:
            print(f"RU grasp helper stopped: {error}")
            self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.physics_subscription = None  # Release the PhysX callback subscription.
        if (self.closing and self.usd_context.get_stage() == self.stage
                and not self.timeline.is_stopped()):
            try:
                self.set_grasp_targets(np.zeros_like(self.closed_targets))  # Reopen, as at startup.
            except Exception as error:
                print(f"Could not reopen the RU fingers: {error}")
        self.closing = False
        self.subscription.unsubscribe()
        self.node.destroy_node()
        layers = self.stage.GetSessionLayer().subLayerPaths
        if self.layer.identifier in layers:
            layers.remove(self.layer.identifier)
        print("RU grasp stopped; temporary joint overrides removed")


def start_grasp(owner_pid, actuators, effort_names, target_m, hand_offset_m, **grasp):
    previous = getattr(builtins, "helios_grasp", None)
    if previous is not None:
        previous.close()
    builtins.helios_grasp = HeliosGrasp(owner_pid, actuators, effort_names, target_m, hand_offset_m, **grasp)
    return builtins.helios_grasp
