"""Helios ROS IK. Start with Isaac stopped, then Play; Space transfers RU, q exits.

Joint feedback initializes IK once; measured hand motion gates the clearance path.
Attachments, gains and target overrides use a removable Isaac session layer.
Run instructions and configuration: README.md.
"""

import json
import os
import select
import socket
import sys
import termios
import time
import tty
import uuid
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import tyro
import yourdfpy
import yaml

from helios_metrics import RolloutMetrics
from helios_rviz import RvizPoses

DEFAULT_URDF = (
    Path(__file__).resolve().parents[2]
    / "Thesis/third_party/ORBIT_urdf_stack/helios_description/urdf/helios_quadmanual_flat.urdf"
)
HANDS = {
    "RU": ("right_wrist", "or2r_R_Carpals_8d1f1041"),
    "LU": ("left_wrist", "or2l_L_Carpals_719fff8c"),
    "RL": ("or4r_right_wrist", "or4r_R_Carpals_8d1f1041"),
    "LL": ("or4l_left_wrist", "or4l_L_Carpals_719fff8c"),
}
# scene_helios_v0.usd: each D6's body1 anchor, transformed from its mesh
# frame into its URDF palm frame, including the mesh's 0.001 scale (metres).
D6_OFFSET = (-0.005, 0.018122922877, 0.04950000067)
# RU body0-local target in metres (world coordinates when body0 is empty).
# Edit between runs; None uses the scene target.
RU_TARGET_M = (0.28, 0.265, 1.408)
# Residual multipliers: each soft cost is sum((weight * error)**2).
COST_WEIGHTS = {
    "position": 50.0,  # Also scales the hard attached-position constraints.
    "fixed_orientations": 25.0,
    "ru_roll": 0.0,  # Forearm tilt in the global Y-Z plane; 0 disables it.
    "upright": 0.0,
    "motion": 0.05,
}


def isaac_request(code, port, timeout=30.0):
    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as connection:
        connection.settimeout(timeout)
        connection.sendall(code.encode())
        connection.shutdown(socket.SHUT_WR)
        with connection.makefile("r", encoding="utf-8") as stream:
            response = json.load(stream)
    if response.get("status") != "ok":
        raise RuntimeError(f"Isaac request failed: {response.get('evalue', response)}")
    return response


def start_auto_attachment(port, effort_names, gains):
    source = Path(__file__).with_name("helios_attach_isaac.py").read_text()
    code = (
        "import omni.timeline\n"
        "if not omni.timeline.get_timeline_interface().is_stopped():\n"
        "    raise RuntimeError('Stop Isaac before starting auto-attach, then press Play')\n"
        "_helios = {'__name__': '_helios_attachment'}\n"
        f"exec(compile({source!r}, 'helios_attach_isaac.py', 'exec'), _helios)\n"
        f"_helios['start_attachment']({os.getpid()}, {gains!r}, {effort_names!r}, {RU_TARGET_M!r})\n"
    )
    response = isaac_request(code, port)
    print(response.get("output", ""), end="")
    print("Attachment helper is running in Isaac. Press Play, then wait for the Space-key prompt.")


class IsaacBridge:
    """Joint commands, measured status and a direct Isaac release call."""

    def __init__(self, names, isaac_port):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String

        rclpy.init(args=[])
        self.ros, self.message_type, self.names = rclpy, JointState, names
        self.node = rclpy.create_node("helios_ik")
        self.state = None
        self.attachment = None
        self.attachment_received = 0.0
        self.isaac_port = isaac_port
        self.publisher = self.node.create_publisher(JointState, "/joint_command", 1)
        self.node.create_subscription(
            JointState, "/joint_states",
            lambda msg: setattr(self, "state", (msg, time.monotonic())),
            qos_profile_sensor_data,
        )
        self.node.create_subscription(String, "/helios/ru_attachment", self.receive_attachment, 1)

    def receive_attachment(self, msg):
        self.attachment = json.loads(msg.data)
        self.attachment_received = time.monotonic()

    def attachment_status(self):
        if time.monotonic() - self.attachment_received > 1.0:
            return None
        return self.attachment

    def release(self):
        isaac_request("import builtins\nbuiltins.helios_attachment.release()", self.isaac_port)

    def read(self):
        # Drain queued feedback without blocking the control loop.
        for _ in range(6):
            self.ros.spin_once(self.node, timeout_sec=0.0)
        if self.state is None or time.monotonic() - self.state[1] > 1.0:
            return None
        joints = self.state[0]
        positions = dict(zip(joints.name, joints.position))
        if not all(name in positions for name in self.names):
            return None
        cfg = jnp.array([positions[name] for name in self.names])
        if not np.isfinite(cfg).all():
            raise ValueError("Nonfinite joint feedback")
        return cfg

    def publish(self, names, positions):
        # A long initial JIT compile or paused simulation must not send stale commands.
        if self.state is None or time.monotonic() - self.state[1] > 1.0 or self.attachment_status() is None:
            return False
        msg = self.message_type()
        msg.header.stamp = self.state[0].header.stamp
        msg.name, msg.position = list(names), np.asarray(positions, dtype=float).tolist()
        self.publisher.publish(msg)
        return True

    def close(self):
        self.node.destroy_node()
        self.ros.try_shutdown()


class TerminalKeys:
    """Single-key input with terminal settings restored even on Ctrl+C/errors."""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Run this script in an interactive terminal for Space/q controls")
        self.fd = sys.stdin.fileno()
        self.settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def read(self):
        if select.select([self.fd], [], [], 0.0)[0]:
            return os.read(self.fd, 32).decode(errors="ignore")
        return ""

    def __exit__(self, *_exc):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)


class ClearanceTrajectory:
    phases = ("retreat (+Y)", "travel", "approach (-Y)")

    def __init__(self, start, goal, offset, durations_s, waypoint_tolerance_m, tracking_tolerance_m):
        start, goal, offset = np.asarray([start, goal, offset], dtype=float)
        self.waypoints = np.array([start, start + offset, goal + offset, goal])
        self.durations = np.asarray(durations_s, dtype=float)
        self.waypoint_tolerance = float(waypoint_tolerance_m)
        self.tracking_tolerance = float(tracking_tolerance_m)
        if self.waypoints.shape != (4, 3) or not np.isfinite(self.waypoints).all():
            raise ValueError("Trajectory points must be finite 3D positions")
        if self.durations.shape != (3,) or not np.isfinite(self.durations).all() or np.any(self.durations <= 0):
            raise ValueError("Provide three positive, finite phase durations")
        if not all(np.isfinite(t) and t > 0 for t in (self.waypoint_tolerance, self.tracking_tolerance)):
            raise ValueError("Trajectory tolerances must be positive and finite")
        self.phase, self.elapsed = 0, 0.0

    @property
    def done(self):
        return self.phase == 3

    def sample(self, elapsed):
        if self.done:
            return self.waypoints[-1].copy()
        u = np.clip(elapsed / self.durations[self.phase], 0.0, 1.0)
        blend = u**3 * (10 - 15 * u + 6 * u**2)  # Zero endpoint velocity and acceleration.
        return self.waypoints[self.phase] + blend * (self.waypoints[self.phase + 1] - self.waypoints[self.phase])

    def propose(self, dt, hand):
        elapsed = self.elapsed
        if not self.done and np.linalg.norm(hand - self.sample(elapsed)) <= self.tracking_tolerance:
            elapsed = min(elapsed + np.clip(dt, 0.0, 0.05), self.durations[self.phase])
        return self.sample(elapsed), elapsed

    def commit(self, elapsed, hand):
        if self.done:
            return
        self.elapsed = elapsed
        if elapsed >= self.durations[self.phase] and np.linalg.norm(hand - self.waypoints[self.phase + 1]) <= self.waypoint_tolerance:
            self.phase += 1
            self.elapsed = 0.0


def load_helios(urdf_path):
    model = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=False, load_meshes=False).robot
    active = (
        {
            f"{joint}_{arm}"
            for arm in HANDS
            for joint in ("Shoulder1", "Shoulder2", "Shoulder3", "Rcj1", "Rcj2", "Ua")
        }
        | {wrist for wrist, _ in HANDS.values()}
        | {"Yaw_HIP", "Roll_HIP"}
    )
    for joint in model.joints:
        if joint.name not in active:
            joint.type = "fixed"  # Keep fingers at their zero pose.
        elif joint.name.startswith("Rcj2_"):
            joint.mimic = yourdfpy.Mimic(
                joint=joint.name.replace("Rcj2_", "Rcj1_"),
                multiplier=1.0 if joint.name.endswith(("_RU", "_LL")) else -1.0,
                offset=0.0,
            )
    for arm, (_, palm) in HANDS.items():
        origin = np.eye(4)
        origin[:3, 3] = D6_OFFSET
        model.links.append(yourdfpy.Link(name=f"ee_{arm}"))
        model.joints.append(
            yourdfpy.Joint(
                name=f"ee_{arm}_fixed",
                type="fixed",
                parent=palm,
                child=f"ee_{arm}",
                origin=origin,
            )
        )
    return yourdfpy.URDF(model, load_meshes=False)


@jdc.jit
def solve_ik(
    robot: pk.Robot, link_indices, targets, orientation_targets, previous_cfg, previous_base,
    ru_attached: jdc.Static[bool] = False, ik_to_world=None,
):
    joint_var = robot.joint_var_cls(0)
    base_var = jaxls.SE3Var(0)

    def position_residual(vals, var, base, indices, targets):
        positions = vals[base] @ robot.forward_kinematics(vals[var])[indices, 4:]
        return ((positions - targets) * COST_WEIGHTS["position"]).flatten()

    @jaxls.Cost.factory
    def orientation_cost(vals, var, base):
        # Ordinary least-squares cost, never an equality constraint. Include the
        # floating-base rotation so these targets stay fixed in the IK frame.
        fixed = slice(None) if ru_attached else slice(1, None)
        local = jaxlie.SO3(robot.forward_kinematics(vals[var])[link_indices[fixed], :4])
        actual = vals[base].rotation() @ local
        error = (jaxlie.SO3(orientation_targets[fixed]).inverse() @ actual).log()
        return (COST_WEIGHTS["fixed_orientations"] * error).flatten()

    @jaxls.Cost.factory
    def upright_cost(vals, base):
        # Align torso Z with world Z; leave yaw and translation free.
        return (vals[base].rotation().as_matrix()[:, 2] - jnp.array([0.0, 0.0, 1.0])) * COST_WEIGHTS["upright"]

    @jaxls.Cost.factory
    def ru_roll_cost(vals, var, base):
        forearm = robot.links.names.index("or2r_TopTower_Model_4a80d30e")  # Parent of the RU wrist joint.
        rotation = vals[base].rotation() @ jaxlie.SO3(robot.forward_kinematics(vals[var])[forearm, :4])
        if ik_to_world is not None:
            rotation = ik_to_world @ rotation
        matrix = rotation.as_matrix()
        # This frame's +Z points along the forearm towards the wrist, before wrist rotation.
        # Both +/-Y are horizontal. Epsilon handles a vanishing Y-Z projection.
        return jnp.atleast_1d(COST_WEIGHTS["ru_roll"] * jnp.arctan2(matrix[2, 2], jnp.abs(matrix[1, 2]) + 1e-8))

    @jaxls.Cost.factory
    def motion_cost(vals, var, base):
        motion = jnp.concatenate([vals[var] - previous_cfg, (previous_base.inverse() @ vals[base]).log()])
        return motion * COST_WEIGHTS["motion"]

    terms = {
        "ru_position": jaxls.Cost.factory(position_residual)(joint_var, base_var, link_indices[:1], targets[:1]),
        "fixed_positions_violation": jaxls.Cost.factory(kind="constraint_eq_zero")(position_residual)(
            joint_var, base_var, link_indices if ru_attached else link_indices[1:],
            targets if ru_attached else targets[1:],
        ),
        "fixed_orientations": orientation_cost(joint_var, base_var),
        "ru_roll": ru_roll_cost(joint_var, base_var),
        "joint_limits_violation": pk.costs.limit_constraint(robot, joint_var),
        "upright": upright_cost(base_var),
        "motion": motion_cost(joint_var, base_var),
    }
    solution, summary = jaxls.LeastSquaresProblem(list(terms.values()), [joint_var, base_var]).analyze().solve(
        initial_vals=jaxls.VarValues.make([joint_var.with_value(previous_cfg), base_var.with_value(previous_base)]),
        linear_solver="dense_cholesky", verbose=False, return_summary=True,
    )
    # Evaluate the original terms, not the solver's changing AL penalties.
    costs = {}
    for name, term in terms.items():
        residual = term.compute_residual(solution, *term.args)
        if term.kind == "constraint_leq_zero":
            residual = jnp.maximum(residual, 0.0)
        costs[name] = jnp.sum(residual**2)
    costs["total_objective"] = sum(costs[name] for name, term in terms.items() if term.kind == "l2_squared")
    costs["initial_objective"] = summary.cost_history[0]
    return solution[joint_var], solution[base_var], summary.iterations, costs


def main(
    urdf_path: Path = DEFAULT_URDF,
    isaac_port: int = 8226,
    config_path: Path = Path(__file__).with_name("helios.yaml"),
    metrics_dir: Path = Path("helios_runs"),
    rviz: bool = True,
):
    config = yaml.safe_load(config_path.read_text())
    motion, gains = config["trajectory"], config["gains"]
    clearance = float(motion.pop("clearance_m"))
    if not np.isfinite(clearance) or clearance <= 0:
        raise ValueError("clearance_m must be positive and finite (Isaac world +Y)")
    if not all(np.isfinite(value) and value >= 0 for value in COST_WEIGHTS.values()) or COST_WEIGHTS["position"] == 0:
        raise ValueError("Cost weights must be finite and nonnegative; position must be positive")
    if not all(np.isfinite(value) and value >= 0 for value in gains.values()):
        raise ValueError("Motor gains must be finite and nonnegative")
    # Validate settings before starting the helper or releasing anything.
    ClearanceTrajectory(np.zeros(3), np.zeros(3), [0, clearance, 0], **motion)
    urdf = load_helios(urdf_path)
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=urdf.zero_cfg)
    indices = jnp.array([robot.links.names.index(f"ee_{arm}") for arm in HANDS])
    command_names = [j.name for j in urdf.robot.joints if j.type != "fixed"]
    command_indices = jnp.array([robot.joints.names.index(name) for name in command_names])
    run_settings = {
        "urdf": str(urdf_path), "trajectory": {"clearance_m": clearance, **motion},
        "gains": gains, "residual_weights": dict(COST_WEIGHTS),
    }
    with TerminalKeys() as keys:
        start_auto_attachment(isaac_port, command_names, gains)
        bridge = IsaacBridge(robot.joints.actuated_names, isaac_port)
        try:
            viewer = RvizPoses(bridge.node, urdf_path, command_names) if rviz else None
            run_controller(robot, indices, command_names, command_indices, bridge, keys, clearance, motion,
                           metrics_dir=metrics_dir, run_settings=run_settings, viewer=viewer)
        finally:
            bridge.close()


def run_controller(robot, indices, command_names, command_indices, bridge, keys, clearance, motion,
                   metrics_dir=Path("helios_runs"), run_settings=None, viewer=None):
    print("Waiting for joint states and attachment helper; LU/RL/LL must be attached and settled. q: exit")
    deadline = time.monotonic() + 60.0
    while ((state := bridge.read()) is None or (status := bridge.attachment_status()) is None
           or status["root_pose_wxyz_xyz"] is None):
        if "q" in keys.read().lower():
            return
        if time.monotonic() > deadline:
            raise TimeoutError("Missing joint/attachment feedback; start Isaac and the updated attachment helper")
        time.sleep(0.02)

    cfg, base = state, jaxlie.SE3.identity()
    initial_poses = robot.forward_kinematics(cfg)[indices]
    targets = np.array(initial_poses[:, 4:])
    orientation_targets = initial_poses[:, :4]
    was_attached = status["attached"]
    world_to_ik = jaxlie.SE3(jnp.asarray(status["root_pose_wxyz_xyz"])).inverse()
    ik_to_world = world_to_ik.rotation().inverse()
    offset = np.asarray(world_to_ik.rotation() @ jnp.array([0.0, clearance, 0.0]))
    trajectory = None
    metrics = None
    waiting_for_release = False
    last_sim_time = status["sim_time_s"]
    last_message = None
    print("Compiling IK before enabling motion...")
    # Both attachment modes are static JIT specializations; compile before release.
    for attached in (False, True):
        warm_result = solve_ik(
            robot, indices, jnp.asarray(targets), orientation_targets, cfg, base, attached, ik_to_world
        )
        jax.block_until_ready(warm_result)
    keys.read()  # Discard keys pressed before the controller was ready.
    print(f"Ready. Space: release → +Y {clearance:.2f} m → travel → -Y approach → attach. q: exit")

    while True:
        pressed = keys.read().lower()
        if "q" in pressed:
            if metrics is not None:
                metrics.save("interrupted")
            return
        state = bridge.read()
        status = bridge.attachment_status()
        if state is None or status is None:
            if last_message != "Waiting for fresh Isaac feedback; motion held.":
                last_message = "Waiting for fresh Isaac feedback; motion held."
                print(last_message)
            last_sim_time = None
            time.sleep(0.02)
            continue
        sim_time = status["sim_time_s"]
        if last_sim_time is not None and sim_time < last_sim_time:
            raise RuntimeError("Isaac timeline reset; restart the controller")
        dt = 0.0 if last_sim_time is None else sim_time - last_sim_time
        last_sim_time = sim_time
        hand = np.asarray(world_to_ik @ jnp.array(status["hand_m"]))
        dock = np.asarray(world_to_ik @ jnp.array(status["target_m"]))
        if status["attached"] and not was_attached:
            # Latch the last commanded RU orientation once, without reseeding IK
            # from measured joints or pulling RU back to its pre-transfer pose.
            ru_rotation = base.rotation() @ jaxlie.SO3(robot.forward_kinematics(cfg)[indices[0], :4])
            orientation_targets = orientation_targets.at[0].set(ru_rotation.wxyz)
        was_attached = status["attached"]
        # This feedback precedes the new command: compare it with the last target.
        measured_target = targets[0].copy()
        completed = trajectory is not None and not waiting_for_release and status["attached"]
        if completed:
            trajectory = None
            print("RU attached. Space: repeat.")
        if waiting_for_release:
            waiting_for_release = status["attached"]
            dt = 0.0

        if " " in pressed and trajectory is None and not completed:
            bridge.release()
            start = np.asarray(base @ robot.forward_kinematics(cfg)[indices[:1], 4:])[0]
            trajectory = ClearanceTrajectory(start, dock, offset, **motion)
            settings = {**(run_settings or {}), "start_m": start.tolist(), "goal_m": dock.tolist()}
            metrics = RolloutMetrics(metrics_dir, uuid.uuid4().hex, sim_time, command_names, settings)
            waiting_for_release = True

        candidate_targets = targets.copy()
        proposed_elapsed = 0.0
        if trajectory is not None and not waiting_for_release:
            candidate_targets[0], proposed_elapsed = trajectory.propose(dt, hand)
        elif trajectory is None and status["attached"]:
            candidate_targets[0] = dock

        # Keep the previous solution as seed/reference; feedback only gates motion.
        solve_start = time.perf_counter()
        result = solve_ik(
            robot, indices, jnp.asarray(candidate_targets), orientation_targets, cfg, base,
            status["attached"], ik_to_world,
        )
        candidate_cfg, candidate_base, steps, costs = jax.block_until_ready(result)
        solve_ms = (time.perf_counter() - solve_start) * 1000
        actual = np.asarray(candidate_base @ robot.forward_kinematics(candidate_cfg)[indices, 4:])
        errors = np.linalg.norm(actual - candidate_targets, axis=1)
        valid = (
            np.isfinite(candidate_cfg).all() and np.isfinite(actual).all()
            and errors[1:].max() < 0.001
            and errors[0] < (0.001 if status["attached"] else 0.005)
        )
        accepted = valid and bridge.publish(command_names, robot.joints.get_full_config(candidate_cfg)[command_indices])
        if accepted:
            cfg, base, targets = candidate_cfg, candidate_base, candidate_targets
            if trajectory is not None and not waiting_for_release:
                trajectory.commit(proposed_elapsed, hand)
        else:
            # Never feed a failed solve back into the next one or advance the path.
            bridge.publish(command_names, robot.joints.get_full_config(cfg)[command_indices])

        if viewer is not None:
            commanded = dict(zip(robot.joints.names, np.asarray(robot.joints.get_full_config(cfg))))
            viewer.publish(commanded, world_to_ik.inverse() @ base, bridge.state[0], status.get("root_pose_wxyz_xyz"))

        phase = "idle" if trajectory is None else trajectory.phases[min(trajectory.phase, 2)]
        if metrics is not None:
            metrics.add(sim_time, hand, measured_target, status.get("efforts_nm", {}), solve_ms, steps, costs, accepted, phase,
                        rl_wrist_wrench=status.get("rl_wrist_wrench"), ru_roll_rad=status.get("ru_roll_rad"),
                        lu_wrist_wrench=status.get("lu_wrist_wrench"), ll_wrist_wrench=status.get("ll_wrist_wrench"))
            if completed:
                metrics.save()
                metrics = None

        message = phase if valid else "IK cannot track this step; holding"
        if message != last_message:
            print(message)
            last_message = message
        time.sleep(0.02)


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        pass
