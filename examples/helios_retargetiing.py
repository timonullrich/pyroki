"""Headless Helios IK: press Space to transfer RU to its scene attachment target.

Run from a terminal: python examples/helios_retargetiing.py
Uses the neighbouring Thesis checkout's URDF/meshes; override with --urdf-path.

Isaac: source /opt/ros/jazzy/setup.bash and source .venv/bin/activate before running.
The project .venv must use Python 3.12 for ROS Jazzy.
Install dependencies: python -m pip install -e . PyYAML matplotlib
This is a standalone ROS node; no colcon build or workspace overlay is needed.
Input: /joint_states (JointState). Start after LU, RL, LL are attached and settled.
Their startup FK positions are latched in the initial root frame (assumed upright),
using D6_OFFSET. IK coordinates are relative to the initial root. Joint feedback
initializes IK once; subsequent solves use the previous solution.
Attached-hand positions are hard constraints; their orientations have a strong
soft penalty. LU/RL/LL retain startup orientations; RU retains its orientation
at attachment. RU's hand pointing direction has a soft horizontal preference
in the global Y-Z plane.
Restart this script after scene resets or changing attachments.
Output: /joint_command (JointState, radians), including the coupled Rcj2 joints.
Use --no-command-mimics if Isaac implements that coupling itself. The IK assumes
the fingers are held at their zero pose in Isaac.
For RU auto-attachment, enable isaacsim.code_editor.python_server in Isaac once,
then run while Isaac is stopped, and press Play (--ros --auto-attach are defaults).
This starts helios_attach_isaac.py remotely; no Script Editor step is needed.
It also applies helios_gains.yaml to all existing motor drives until shutdown.
Space releases RU, retreats along world +Y, travels with clearance, approaches
along -Y, then arms attachment. Edit helios_trajectory.yaml for clearance/timing.
RU_TARGET_M temporarily overrides RU's body0 anchor through the helper's session
layer. Stop Isaac or exit this script to remove it before editing the scene target.
Hand feedback gates trajectory progress, but does not reseed IK.
RViz overlay: rviz2 -d examples/helios.rviz (blue: commanded, orange: measured).
Use --no-rviz to disable its publishers; no extra ROS nodes are needed.
Press q or Ctrl+C to exit. Use --no-auto-attach if the helper is already running.
Each transfer saves summary.json, rollout.csv, and metrics.png under helios_runs/.
"""

import atexit
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

from helios_trajectory import ClearanceTrajectory
from helios_metrics import RolloutMetrics

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
# Edit between runs; None uses the scene target. Applied with --auto-attach.
RU_TARGET_M = (0.28, 0.265, 1.208)
# Residual multipliers: each soft cost is sum((weight * error)**2).
COST_WEIGHTS = {
    "position": 50.0,  # Also scales the hard attached-position constraints.
    "fixed_orientations": 25.0,
    "ru_roll": 0.0,  # Hand pointing tilt in the global Y-Z plane; 0 disables it.
    "upright": 5.0,
    "motion": 0.05,
}


def isaac_request(code, port, timeout=30.0):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5.0) as connection:
            connection.settimeout(timeout)
            connection.sendall(code.encode())
            connection.shutdown(socket.SHUT_WR)
            with connection.makefile("r", encoding="utf-8") as stream:
                response = json.load(stream)
    except OSError as error:
        raise RuntimeError(
            f"Cannot reach Isaac's Python server on port {port}. Enable "
            "isaacsim.code_editor.python_server in Isaac's Extensions window. "
            f"Connection details: {error}"
        ) from error
    if response.get("status") != "ok":
        raise RuntimeError(f"Isaac request failed: {response.get('evalue', response)}")
    return response


def stop_auto_attachment(port, token):
    # Only stop this process's helper, never a replacement started by another controller.
    code = (
        "import builtins\n"
        "_helper = getattr(builtins, 'helios_attachment', None)\n"
        f"if _helper is not None and getattr(_helper, 'owner_token', None) == {token!r}:\n"
        "    _helper.close()\n"
    )
    try:
        isaac_request(code, port, timeout=3.0)
    except (RuntimeError, ValueError) as error:
        print(f"Could not send helper shutdown; it will also detect process exit: {error}")


def start_auto_attachment(port: int = 8226, effort_names=None):
    """Install the helper in Isaac, tied to this terminal process's lifetime."""
    import yaml

    source = Path(__file__).with_name("helios_attach_isaac.py").read_text()
    gains = yaml.safe_load(Path(__file__).with_name("helios_gains.yaml").read_text())
    gains = {name: float(gains[name]) for name in ("stiffness", "damping")}
    if not all(np.isfinite(value) and value >= 0 for value in gains.values()):
        raise ValueError("Motor gains must be finite and nonnegative")
    token = uuid.uuid4().hex
    # Use a separate namespace so later remote commands cannot overwrite helper globals.
    code = (
        "import omni.timeline\n"
        "if not omni.timeline.get_timeline_interface().is_stopped():\n"
        "    raise RuntimeError('Stop Isaac before starting auto-attach, then press Play')\n"
        "if '_helios_attachment_context' not in globals():\n"
        "    _helios_attachment_context = {}\n"
        "_previous = _helios_attachment_context.get('helios_attachment')\n"
        "if _previous is not None:\n"
        "    _previous.close()\n"
        "_helios_attachment_context['__name__'] = '_helios_attachment'\n"
        f"exec(compile({source!r}, 'helios_attach_isaac.py', 'exec'), _helios_attachment_context)\n"
        f"_helper = _helios_attachment_context['start_attachment'](owner_pid={os.getpid()}, drive_gains={gains!r}, effort_names={effort_names!r}, target_m={RU_TARGET_M!r})\n"
        f"_helper.owner_token = {token!r}\n"
        "_helios_attachment_context['helios_attachment'] = _helper\n"
    )
    response = isaac_request(code, port)
    atexit.register(stop_auto_attachment, port, token)
    print(response.get("output", ""), end="")
    print("Attachment helper is running in Isaac. Press Play, then wait for the Space-key prompt.")


class IsaacBridge:
    """Joint commands plus fresh attachment status and acknowledged release/arm."""

    def __init__(self, names, joint_states_topic, joint_command_topic):
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
        self.closed = False
        self.string_type = String
        self.publisher = self.node.create_publisher(JointState, joint_command_topic, 1)
        self.attachment_publisher = self.node.create_publisher(String, "/helios/ru_attachment_command", 1)
        self.node.create_subscription(
            JointState, joint_states_topic,
            lambda msg: setattr(self, "state", (msg, time.monotonic())),
            qos_profile_sensor_data,
        )
        self.node.create_subscription(String, "/helios/ru_attachment", self.receive_attachment, 1)
        atexit.register(self.close)

    def receive_attachment(self, msg):
        status = json.loads(msg.data)
        anchors = np.asarray(status["anchors_m"], dtype=float)
        hand = np.asarray(status["hand_m"], dtype=float)
        if (
            anchors.shape != (4, 3) or not np.isfinite(anchors).all()
            or hand.shape != (3,) or not np.isfinite(hand).all()
            or type(status["attached"]) is not bool or type(status["armed"]) is not bool
            or not np.isfinite(status["sim_time_s"]) or not np.isfinite(status["distance_m"])
        ):
            raise ValueError("Invalid RU attachment status")
        self.attachment = status
        self.attachment_received = time.monotonic()

    def attachment_status(self):
        if time.monotonic() - self.attachment_received > 1.0:
            return None
        return self.attachment

    def command_attachment(self, action, command_id):
        self.attachment_publisher.publish(self.string_type(data=json.dumps({"action": action, "id": command_id})))

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
        if self.closed:
            return
        self.closed = True
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


def estimate_base(local_points, fixed_points):
    """Fit the root pose from three noncollinear, stationary attachment points."""
    local, fixed = np.asarray(local_points), np.asarray(fixed_points)
    x, y = local - local.mean(axis=0), fixed - fixed.mean(axis=0)
    u, singular_values, vt = np.linalg.svd(x.T @ y)
    if singular_values[1] < 1e-8:
        raise ValueError("Cannot estimate base: attachment points are collinear")
    rotation = vt.T @ np.diag([1.0, 1.0, np.linalg.det(vt.T @ u.T)]) @ u.T
    return jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3.from_matrix(jnp.asarray(rotation)),
        jnp.asarray(fixed.mean(axis=0) - rotation @ local.mean(axis=0)),
    )


def load_helios(urdf_path: Path, load_meshes: bool = True) -> yourdfpy.URDF:
    model = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=False).robot
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
    return yourdfpy.URDF(
        model,
        load_meshes=load_meshes,
        filename_handler=lambda fname: str(
            urdf_path.resolve().parents[2] / fname.removeprefix("package://")
        ),
    )


@jdc.jit
def solve_ik(
    robot: pk.Robot, link_indices, targets, orientation_targets, previous_cfg, previous_base,
    ru_attached: jdc.Static[bool] = False, orientation_weight=COST_WEIGHTS["fixed_orientations"], ik_to_world=None,
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
        return (orientation_weight * error).flatten()

    @jaxls.Cost.factory
    def upright_cost(vals, base):
        # Align torso Z with world Z; leave yaw and translation free.
        return (vals[base].rotation().as_matrix()[:, 2] - jnp.array([0.0, 0.0, 1.0])) * COST_WEIGHTS["upright"]

    @jaxls.Cost.factory
    def ru_roll_cost(vals, var, base):
        rotation = vals[base].rotation() @ jaxlie.SO3(robot.forward_kinematics(vals[var])[link_indices[0], :4])
        if ik_to_world is not None:
            rotation = ik_to_world @ rotation
        matrix = rotation.as_matrix()
        # Fingers point along local +Z: column 2 is their world direction.
        # Both +/-Y are horizontal. Epsilon handles a vanishing Y-Z projection.
        return jnp.atleast_1d(COST_WEIGHTS["ru_roll"] * jnp.arctan2(matrix[2, 2], jnp.abs(matrix[1, 2]) + 1e-8))

    @jaxls.Cost.factory
    def motion_cost(vals, var, base):
        return (
            jnp.concatenate(
                [
                    vals[var] - previous_cfg,
                    (previous_base.inverse() @ vals[base]).log(),
                ]
            )
            * COST_WEIGHTS["motion"]
        )

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
    solution, summary = (
        jaxls.LeastSquaresProblem(
            costs=list(terms.values()),
            variables=[joint_var, base_var],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(previous_cfg), base_var.with_value(previous_base)]
            ),
            linear_solver="dense_cholesky",
            verbose=False,
            return_summary=True,
        )
    )
    # Evaluate the original terms, not the solver's changing AL penalties.
    costs = {}
    for name, term in terms.items():
        residual = term.compute_residual(solution, *term.args)
        if term.kind == "constraint_leq_zero":
            residual = jnp.maximum(residual, 0.0)
        costs[name] = jnp.sum(residual**2)
    costs["total_objective"] = sum(costs[name] for name, term in terms.items() if term.kind == "l2_squared")
    # JAXLS records the same non-augmented objective at the initial guess.
    # Keep the final value evaluated at `solution`: the last history entry can
    # describe a rejected trial, or be truncated at the iteration limit.
    costs["initial_objective"] = summary.cost_history[0]
    return solution[joint_var], solution[base_var], summary.iterations, costs


def main(
    urdf_path: Path = DEFAULT_URDF,
    ros: bool = True,
    joint_states_topic: str = "/joint_states",
    joint_command_topic: str = "/joint_command",
    command_mimics: bool = True,
    auto_attach: bool = True,
    isaac_port: int = 8226,
    trajectory_path: Path = Path(__file__).with_name("helios_trajectory.yaml"),
    metrics_dir: Path = Path("helios_runs"),
    rviz: bool = True,
):
    import yaml

    if not ros:
        raise ValueError("The headless trajectory controller requires ROS and the Isaac attachment helper")
    motion = yaml.safe_load(trajectory_path.read_text())
    clearance = float(motion.pop("clearance_m"))
    orientation_weight = COST_WEIGHTS["fixed_orientations"]
    if not np.isfinite(clearance) or clearance <= 0:
        raise ValueError("clearance_m must be positive and finite (Isaac world +Y)")
    if not all(np.isfinite(value) and value >= 0 for value in COST_WEIGHTS.values()) or COST_WEIGHTS["position"] == 0:
        raise ValueError("Cost weights must be finite and nonnegative; position must be positive")
    # Validate settings before starting the helper or releasing anything.
    ClearanceTrajectory(np.zeros(3), np.zeros(3), [0, clearance, 0], **motion)
    urdf = load_helios(urdf_path, load_meshes=False)
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=urdf.zero_cfg)
    indices = jnp.array([robot.links.names.index(f"ee_{arm}") for arm in HANDS])
    command_names = list(robot.joints.actuated_names)
    if command_mimics:
        command_names += [j.name for j in urdf.robot.joints if j.mimic is not None and j.type != "fixed"]
    command_indices = jnp.array([robot.joints.names.index(name) for name in command_names])
    # Include coupled joints even when Isaac supplies their commands itself.
    effort_names = [j.name for j in urdf.robot.joints if j.type != "fixed"]
    run_settings = {
        "urdf": str(urdf_path), "trajectory": {"clearance_m": clearance, **motion},
        "gains": yaml.safe_load(Path(__file__).with_name("helios_gains.yaml").read_text()) if auto_attach else "external helper",
        "residual_weights": dict(COST_WEIGHTS),
    }
    with TerminalKeys() as keys:
        if auto_attach:
            start_auto_attachment(isaac_port, effort_names)
        bridge = IsaacBridge(robot.joints.actuated_names, joint_states_topic, joint_command_topic)
        try:
            from helios_rviz import RvizPoses

            viewer = RvizPoses(bridge.node, urdf_path, effort_names) if rviz else None
            run_controller(robot, indices, command_names, command_indices, bridge, keys, clearance, motion,
                           orientation_weight, metrics_dir, effort_names, run_settings, viewer)
        finally:
            bridge.close()


def run_controller(robot, indices, command_names, command_indices, bridge, keys, clearance, motion, orientation_weight=COST_WEIGHTS["fixed_orientations"],
                   metrics_dir=Path("helios_runs"), effort_names=None, run_settings=None, viewer=None):
    print("Waiting for joint states and attachment helper; LU/RL/LL must be attached and settled. q: exit")
    deadline = time.monotonic() + 60.0
    while True:
        if "q" in keys.read().lower():
            return
        state = bridge.read()
        status = bridge.attachment_status()
        if state is not None and status is not None:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("Missing joint/attachment feedback; start Isaac and the updated attachment helper")
        time.sleep(0.02)

    cfg, base = state, jaxlie.SE3.identity()
    initial_poses = robot.forward_kinematics(cfg)[indices]
    targets = np.array(initial_poses[:, 4:])
    orientation_targets = initial_poses[:, :4]
    was_attached = status["attached"]
    # Calibrate once; do not estimate the base from measured joints each iteration.
    world_to_ik = estimate_base(jnp.array(status["anchors_m"])[1:], targets[1:])
    ik_to_world = world_to_ik.rotation().inverse()
    offset = np.asarray(world_to_ik.rotation() @ jnp.array([0.0, clearance, 0.0]))
    trajectory = None
    metrics = None
    mode = "idle"
    command_id = None
    last_command = 0.0
    last_sim_time = status["sim_time_s"]
    last_message = None
    last_report = 0.0
    print("Compiling IK before enabling motion...")
    # Both attachment modes are static JIT specializations; compile before release.
    for attached in (False, True):
        warm_result = solve_ik(
            robot, indices, jnp.asarray(targets), orientation_targets, cfg, base, attached, orientation_weight, ik_to_world
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
        dock = np.asarray(world_to_ik @ jnp.array(status["anchors_m"])[0])
        if status["attached"] and not was_attached:
            # Latch the last commanded RU orientation once, without reseeding IK
            # from measured joints or pulling RU back to its pre-transfer pose.
            ru_rotation = base.rotation() @ jaxlie.SO3(robot.forward_kinematics(cfg)[indices[0], :4])
            orientation_targets = orientation_targets.at[0].set(ru_rotation.wxyz)
        was_attached = status["attached"]
        # This feedback precedes the new command: compare it with the last target.
        measured_target = targets[0].copy()
        completed = False

        if " " in pressed:
            if mode == "idle":
                start = np.asarray(base @ robot.forward_kinematics(cfg)[indices[:1], 4:])[0]
                trajectory = ClearanceTrajectory(start, dock, offset, **motion)
                command_id = uuid.uuid4().hex
                settings = {**(run_settings or {}), "start_m": start.tolist(), "goal_m": dock.tolist()}
                metrics = RolloutMetrics(metrics_dir, command_id, sim_time, effort_names or command_names, settings)
                mode, last_command = "releasing", 0.0
            else:
                print("Transfer already running; Space ignored.")

        if mode == "releasing":
            if status["command_id"] == command_id and not status["attached"] and not status["armed"]:
                mode = "moving"
                dt = 0.0
            elif time.monotonic() - last_command >= 0.2:
                bridge.command_attachment("release", command_id)
                last_command = time.monotonic()
        if mode == "docking":
            if status["command_id"] == command_id and status["attached"]:
                completed = True
                mode, trajectory = "idle", None
                print("RU attached. Space: repeat using the scene's current RU target.")
            elif time.monotonic() - last_command >= 0.2:
                bridge.command_attachment("arm", command_id)
                last_command = time.monotonic()

        candidate_targets = targets.copy()
        proposed_elapsed = 0.0
        if mode == "moving":
            if status["attached"]:
                raise RuntimeError("RU unexpectedly attached during clearance motion")
            candidate_targets[0], proposed_elapsed = trajectory.propose(dt, hand)
        elif mode == "idle" and status["attached"]:
            candidate_targets[0] = dock

        # Keep the previous solution as seed/reference; feedback only gates motion.
        target_array = jnp.asarray(candidate_targets)
        solve_start = time.perf_counter()
        result = solve_ik(
            robot, indices, target_array, orientation_targets, cfg, base,
            status["attached"], orientation_weight, ik_to_world,
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
            if mode == "moving":
                trajectory.commit(proposed_elapsed, hand)
                if trajectory.done:
                    mode, last_command = "docking", 0.0
        else:
            # Never feed a failed solve back into the next one or advance the path.
            bridge.publish(command_names, robot.joints.get_full_config(cfg)[command_indices])

        if viewer is not None:
            commanded = dict(zip(robot.joints.names, np.asarray(robot.joints.get_full_config(cfg))))
            viewer.publish(commanded, world_to_ik.inverse() @ base, bridge.state[0], status.get("root_pose_wxyz_xyz"))

        if metrics is not None:
            phase = mode if mode != "moving" else trajectory.phases[trajectory.phase]
            metrics.add(sim_time, hand, measured_target, status.get("efforts_nm", {}), solve_ms, steps, costs, accepted, phase,
                        rl_wrist_wrench=status.get("rl_wrist_wrench"), ru_roll_rad=status.get("ru_roll_rad"))
            if completed:
                metrics.save()
                metrics = None

        message = mode if mode != "moving" else trajectory.phases[trajectory.phase]
        if not valid:
            message += f": IK cannot track this step ({errors.max() * 1000:.1f} mm); holding"
        elif mode == "moving":
            message += f"; hand error {np.linalg.norm(hand - targets[0]) * 100:.1f} cm"
        if message != last_message and time.monotonic() - last_report > 1.0:
            print(message)
            last_message, last_report = message, time.monotonic()
        time.sleep(0.02)


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        pass
