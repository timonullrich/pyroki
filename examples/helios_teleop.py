"""Teleoperate Helios in Isaac from ORBIT_Teleop's wrist poses and finger commands with Pyroki IK.

Start with Isaac stopped, then Play, then ORBIT_Teleop with --relative (it publishes the
active pair on its status socket). The base is free: the inactive pair's hands hold the
robot where they are, so moving the active pair can move the whole body. Each active palm
follows the operator's wrist motion since its pair became active (no jump on activation).
The first time a pair is driven, its D6 bar attachments are disabled and its fingers follow
the operator's commands from then on. Pure teleop: no scripted motion or grasp. q: exit.
"""

import re
import runpy
import time
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import tyro
import yaml
import zmq

from helios_retargeting import DEFAULT_URDF, HANDS, TerminalKeys, isaac_request, load_helios

PAIRS = {"upper": ("RU", "LU"), "lower": ("RL", "LL")}  # (right, left) arms, as in ORBIT_Teleop.
# [x, y, z, qx, qy, qz, qw] of the active pair's wrist frames in the robot root frame.
WRIST_TOPIC = "/helios/high_level/targets/{}_wrist_pose"
COMMAND_TOPIC = "/helios/high_level/commands"  # All joints in ORBIT_Teleop's HELIOS_JOINT_ORDER.
STATUS_ENDPOINT = "tcp://127.0.0.1:8106"  # ORBIT_Teleop arm_status_port (JSON with active_pair).
DEFAULT_CONSTANTS = Path(__file__).resolve().parents[2] / "ORBIT_Teleop/openteach/constants.py"
FINGER = re.compile(r".*_(pinky|m1|m2|index|thumb)_(abd|mcp|pip|dip)")
STALE_S = 0.5
MOTION_WEIGHT = 0.1


def isaac_call(port, call):
    """Run a helios_teleop_isaac.py function in Isaac, e.g. call="close()"."""
    helpers, isaac = (Path(__file__).with_name(name).read_text()
                      for name in ("helios_grasp_isaac.py", "helios_teleop_isaac.py"))
    response = isaac_request(
        "_teleop = {'__name__': '_helios_teleop'}\n"
        f"exec(compile({helpers!r}, 'helios_grasp_isaac.py', 'exec'), _teleop)\n"  # Actuator helpers only.
        f"exec(compile({isaac!r}, 'helios_teleop_isaac.py', 'exec'), _teleop)\n"
        f"exec({call!r}, _teleop)\n", port)
    print(response.get("output", ""), end="")


class TeleopBridge:
    """Isaac /joint_states and /joint_command, plus the latest ORBIT_Teleop messages in inputs."""

    def __init__(self, names, isaac_port):
        import rclpy
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Float64MultiArray

        rclpy.init(args=[])
        self.rclpy, self.message_type, self.names, self.isaac_port = rclpy, JointState, names, isaac_port
        self.node = rclpy.create_node("helios_teleop")
        self.state, self.inputs = None, {}
        self.publisher = self.node.create_publisher(JointState, "/joint_command", 1)
        self.node.create_subscription(JointState, "/joint_states",
                                      lambda msg: setattr(self, "state", (msg, time.monotonic())), qos_profile_sensor_data)
        for key, topic in (("right", WRIST_TOPIC.format("right")), ("left", WRIST_TOPIC.format("left")),
                           ("command", COMMAND_TOPIC)):
            self.node.create_subscription(
                Float64MultiArray, topic, lambda msg, key=key: self.inputs.__setitem__(key, (msg.data, time.monotonic())), 10)

    def read(self):
        for _ in range(6):  # Drain queued messages without blocking.
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
        if self.state is None or time.monotonic() - self.state[1] > 1.0:
            return None  # Paused, stopped or not started: hold.
        positions = dict(zip(self.state[0].name, self.state[0].position))
        if not all(name in positions for name in self.names):
            return None
        cfg = jnp.array([positions[name] for name in self.names])
        return cfg if np.isfinite(cfg).all() else None

    def publish(self, names, positions):
        msg = self.message_type()
        msg.header.stamp = self.state[0].header.stamp
        msg.name, msg.position = list(names), np.asarray(positions, dtype=float).tolist()
        self.publisher.publish(msg)

    def release_attachments(self, arms):
        isaac_call(self.isaac_port, f"release({list(arms)!r})")

    def close(self):
        self.node.destroy_node()
        self.rclpy.try_shutdown()


def build_model(urdf_path, constants_path):
    urdf = load_helios(urdf_path)
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=urdf.zero_cfg)
    command_names = [j.name for j in urdf.robot.joints if j.type != "fixed"]
    order = list(runpy.run_path(str(constants_path))["HELIOS_JOINT_ORDER"])
    fingers = [j.name for j in urdf.robot.joints if FINGER.fullmatch(j.name)]
    return SimpleNamespace(
        robot=robot, arms=list(HANDS),
        palm_indices=jnp.array([robot.links.names.index(palm) for _, palm in HANDS.values()], dtype=jnp.int32),
        command_names=command_names,
        command_indices=jnp.array([robot.joints.names.index(name) for name in command_names]),
        fingers=fingers,
        finger_pairs=["lower" if name.startswith("or4") else "upper" for name in fingers],
        # ORBIT_Teleop calls the m1/m2 fingers ring/middle.
        finger_slots=[order.index(name.replace("_m1_", "_ring_").replace("_m2_", "_middle_")) for name in fingers],
        command_size=len(order),
    )


def teleop_target(palm_start, operator_start, operator_now):
    """Palm pose at activation, moved by the operator's motion since then."""
    rotation = operator_now.rotation() @ operator_start.rotation().inverse() @ palm_start.rotation()
    translation = palm_start.translation() + operator_now.translation() - operator_start.translation()
    return jaxlie.SE3.from_rotation_and_translation(rotation, translation)


def world_palms(m, cfg, base):
    return base @ jaxlie.SE3(m.robot.forward_kinematics(cfg)[m.palm_indices])


@jdc.jit
def solve_teleop(robot, palm_indices, targets, active, previous_cfg, previous_base):
    """Active palms track their targets; the other palms hold the robot. The base is free."""
    joints, base = robot.joint_var_cls(0), jaxls.SE3Var(0)

    @jaxls.Cost.factory(kind="constraint_eq_zero")
    def holding_positions(vals, joints, base):
        palms = vals[base] @ robot.forward_kinematics(vals[joints])[palm_indices, 4:]
        return ((palms - targets[:, 4:]) * (1.0 - active)[:, None] * 1000.0).flatten()  # mm

    @jaxls.Cost.factory
    def base_motion(vals, base):
        return (previous_base.inverse() @ vals[base]).log() * MOTION_WEIGHT

    costs = [
        holding_positions(joints, base),
        # Soft orientations for every palm; positions only for the active ones.
        pk.costs.pose_cost_with_base(robot, joints, base, jaxlie.SE3(targets), palm_indices,
                                     50.0 * active[:, None], 10.0),
        pk.costs.rest_cost(joints, previous_cfg, MOTION_WEIGHT),
        base_motion(base),
        pk.costs.limit_constraint(robot, joints),
    ]
    solution = jaxls.LeastSquaresProblem(costs, [joints, base]).analyze().solve(
        initial_vals=jaxls.VarValues.make([joints.with_value(previous_cfg), base.with_value(previous_base)]),
        linear_solver="dense_cholesky", verbose=False,
        # Bounded time per tick; each tick warm-starts from the last one (uncapped: ~0.5 s).
        termination=jaxls.TerminationConfig(max_iterations=10),
    )
    return solution[joints], solution[base]


def latest_pair(status, pair):
    while True:
        try:
            message = status.recv_json(zmq.NOBLOCK)
        except zmq.Again:
            return pair
        if isinstance(message, dict) and message.get("active_pair") in PAIRS:
            pair = message["active_pair"]


def fresh(inputs, key):
    sample = inputs.get(key)
    if sample is None or time.monotonic() - sample[1] > STALE_S:
        return None
    data = np.asarray(sample[0], dtype=float)
    return data if np.isfinite(data).all() else None


def run_teleop(m, bridge, keys, status):
    inputs = bridge.inputs
    print("Waiting for Isaac joint states; press Play. q: exit")
    while (state := bridge.read()) is None:
        if "q" in keys.read().lower():
            return
        time.sleep(0.02)
    # IK frame: the robot root at start, fixed in the world.
    cfg, base = state, jaxlie.SE3.identity()
    holds = np.array(world_palms(m, cfg, base).wxyz_xyz)
    print("Compiling IK...")
    jax.block_until_ready(solve_teleop(m.robot, m.palm_indices, jnp.asarray(holds), jnp.zeros(len(m.arms)), cfg, base))
    print("Ready; start ORBIT_Teleop with --relative. q: exit")
    pair, anchors, released = None, {}, set()
    while True:
        if "q" in keys.read().lower():
            return
        if bridge.read() is None:
            time.sleep(0.02)
            continue
        if (new_pair := latest_pair(status, pair)) != pair:
            # Every hand holds where it is; the new pair then follows the operator from here.
            pair, anchors = new_pair, {}
            holds = np.array(world_palms(m, cfg, base).wxyz_xyz)
            print(f"Active pair: {pair}")

        targets, active, driven = holds.copy(), np.zeros(len(m.arms)), False
        for side, arm in zip(("right", "left"), PAIRS.get(pair, ())):
            i = m.arms.index(arm)
            active[i] = 1.0  # Without a sample it softly keeps its pose from the pair switch.
            pose = fresh(inputs, side)
            if pose is None or pose.shape != (7,):
                continue
            operator = jaxlie.SE3.from_rotation_and_translation(jaxlie.SO3.from_quaternion_xyzw(pose[3:]), pose[:3])
            palm_start, operator_start = anchors.setdefault(arm, (jaxlie.SE3(holds[i]), operator))
            targets[i] = teleop_target(palm_start, operator_start, operator).wxyz_xyz
            driven = True
        if driven:
            if pair not in released:
                bridge.release_attachments(PAIRS[pair])
                released.add(pair)
                print(f"{pair} pair driven for the first time: D6 attachments of {', '.join(PAIRS[pair])} "
                      "disabled; its fingers follow the operator.")
            new_cfg, new_base = solve_teleop(m.robot, m.palm_indices, jnp.asarray(targets), jnp.asarray(active), cfg, base)
            held = np.linalg.norm(np.asarray(world_palms(m, new_cfg, new_base).translation()) - targets[:, 4:], axis=1)
            if np.isfinite(new_cfg).all() and np.isfinite(held).all() and held[active == 0].max(initial=0.0) < 1e-3:
                cfg, base = new_cfg, new_base  # Otherwise keep the last command.
        else:
            time.sleep(0.01)  # Nothing to solve; the IK paces the loop otherwise.

        names = list(m.command_names)
        positions = list(np.asarray(m.robot.joints.get_full_config(cfg)[m.command_indices]))
        operator_cmd = fresh(inputs, "command")
        if operator_cmd is not None and operator_cmd.shape == (m.command_size,):
            for name, finger_pair, slot in zip(m.fingers, m.finger_pairs, m.finger_slots):
                if finger_pair in released:  # Fingers of a pair not yet driven keep their targets.
                    names.append(name)
                    positions.append(operator_cmd[slot])
        bridge.publish(names, positions)


def main(
    urdf_path: Path = DEFAULT_URDF,
    isaac_port: int = 8226,
    config_path: Path = Path(__file__).with_name("helios.yaml"),
    status_endpoint: str = STATUS_ENDPOINT,
    teleop_constants: Path = DEFAULT_CONSTANTS,
):
    actuators = yaml.safe_load(config_path.read_text())["actuators"]
    m = build_model(urdf_path, teleop_constants)
    context = zmq.Context()
    status = context.socket(zmq.SUB)
    status.setsockopt(zmq.LINGER, 0)
    status.setsockopt(zmq.CONFLATE, 1)
    status.setsockopt(zmq.SUBSCRIBE, b"")
    status.connect(status_endpoint)
    try:
        with TerminalKeys() as keys:
            isaac_call(isaac_port, f"start({actuators!r})")
            bridge = TeleopBridge(m.robot.joints.actuated_names, isaac_port)
            try:
                run_teleop(m, bridge, keys, status)
            finally:
                bridge.close()
                isaac_call(isaac_port, "close()")  # Restores attachments and scene gains.
    finally:
        status.close()
        context.term()


if __name__ == "__main__":
    tyro.cli(main)
