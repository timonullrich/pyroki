"""Helios whole-body IK: drag RU while the other hand positions stay fixed.

Run: python examples/helios_retargetiing.py
Uses the neighbouring Thesis checkout's URDF/meshes; override with --urdf-path.

Isaac: source /opt/ros/jazzy/setup.bash, source .venv/bin/activate, then run
with --ros. The project .venv must use Python 3.12 for ROS Jazzy.
Install dependencies once in that environment: python -m pip install -e . PyYAML
This is a standalone ROS node; no colcon build or workspace overlay is needed.
Input: /joint_states (JointState). Start after LU, RL, LL are attached and settled.
Their startup FK positions are latched in the initial root frame (assumed upright),
using D6_OFFSET. Base motion is estimated from these three stationary points;
Viser coordinates are relative to the initial root, not Isaac's world frame.
Restart this script after scene resets or changing attachments.
Output: /joint_command (JointState, radians), including the coupled Rcj2 joints.
Use --no-command-mimics if Isaac implements that coupling itself. The IK assumes
the fingers are held at their zero pose in Isaac.
For RU auto-attachment, enable isaacsim.code_editor.python_server in Isaac once,
then run with --ros --auto-attach while Isaac is stopped, and press Play.
This starts helios_attach_isaac.py remotely; no Script Editor step is needed.
Its optional /helios/ru_attachment status shows the docking target in Viser and
locks RU in the IK after attachment; startup still needs only /joint_states.
"""

import atexit
import json
import os
import socket
import time
import uuid
from pathlib import Path

import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
import pyroki as pk
import tyro
import viser
import yourdfpy
from viser.extras import ViserUrdf

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


def start_auto_attachment(port: int = 8226):
    """Install the helper in Isaac, tied to this terminal process's lifetime."""
    source = Path(__file__).with_name("helios_attach_isaac.py").read_text()
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
        f"_helper = _helios_attachment_context['start_attachment'](owner_pid={os.getpid()})\n"
        f"_helper.owner_token = {token!r}\n"
        "_helios_attachment_context['helios_attachment'] = _helper\n"
    )
    response = isaac_request(code, port)
    atexit.register(stop_auto_attachment, port, token)
    print(response.get("output", ""), end="")
    print("Auto-attach is running in Isaac. Press Play, then move RU toward the target.")


class IsaacBridge:
    """ROS stays optional so the standalone Viser example still works."""

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
        self.publisher = self.node.create_publisher(JointState, joint_command_topic, 1)
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
        if anchors.shape != (4, 3) or not np.isfinite(anchors).all() or type(status["attached"]) is not bool:
            raise ValueError("Invalid RU attachment status")
        self.attachment = status

    def read(self):
        # Drain queued feedback without blocking the Viser loop.
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
        if self.state is None or time.monotonic() - self.state[1] > 1.0:
            return
        msg = self.message_type()
        msg.header.stamp = self.state[0].header.stamp
        msg.name, msg.position = list(names), np.asarray(positions, dtype=float).tolist()
        self.publisher.publish(msg)

    def close(self):
        self.node.destroy_node()
        self.ros.try_shutdown()


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
def solve_ik(robot: pk.Robot, link_indices, targets, previous_cfg, previous_base, ru_attached: jdc.Static[bool] = False):
    joint_var = robot.joint_var_cls(0)
    base_var = jaxls.SE3Var(0)

    def position_residual(vals, var, base, indices, targets):
        positions = vals[base] @ robot.forward_kinematics(vals[var])[indices, 4:]
        return ((positions - targets) * 50.0).flatten()

    @jaxls.Cost.factory
    def upright_cost(vals, base):
        # Align torso Z with world Z; leave yaw and translation free.
        return (vals[base].rotation().as_matrix()[:, 2] - jnp.array([0.0, 0.0, 1.0])) * 5.0

    @jaxls.Cost.factory
    def motion_cost(vals, var, base):
        return (
            jnp.concatenate(
                [
                    vals[var] - previous_cfg,
                    (previous_base.inverse() @ vals[base]).log(),
                ]
            )
            * 0.05
        )

    solution = (
        jaxls.LeastSquaresProblem(
            costs=[
                jaxls.Cost.factory(position_residual)(
                    joint_var, base_var, link_indices[:1], targets[:1]
                ),
                jaxls.Cost.factory(kind="constraint_eq_zero")(position_residual)(
                    joint_var, base_var,
                    link_indices if ru_attached else link_indices[1:],
                    targets if ru_attached else targets[1:],
                ),
                pk.costs.limit_constraint(robot, joint_var),
                upright_cost(base_var),
                # Resolve redundant motion smoothly; no hand orientation target.
                motion_cost(joint_var, base_var),
            ],
            variables=[joint_var, base_var],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [joint_var.with_value(previous_cfg), base_var.with_value(previous_base)]
            ),
            linear_solver="dense_cholesky",
            verbose=False,
        )
    )
    return solution[joint_var], solution[base_var]


def main(
    urdf_path: Path = DEFAULT_URDF,
    ros: bool = False,
    joint_states_topic: str = "/joint_states",
    joint_command_topic: str = "/joint_command",
    command_mimics: bool = True,
    auto_attach: bool = False,
    isaac_port: int = 8226,
):
    if auto_attach:
        if not ros:
            raise ValueError("--auto-attach requires --ros")
        start_auto_attachment(isaac_port)
    urdf = load_helios(urdf_path)
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=urdf.zero_cfg)
    indices = jnp.array([robot.links.names.index(f"ee_{arm}") for arm in HANDS])
    cfg = robot.joint_var_cls.default_factory()
    base = jaxlie.SE3.identity()
    bridge = None
    if ros:
        bridge = IsaacBridge(robot.joints.actuated_names, joint_states_topic, joint_command_topic)
        print(f"Waiting for {joint_states_topic}; start after the three hands are attached and settled in Isaac.")
        deadline = time.monotonic() + 60.0
        while (state := bridge.read()) is None:
            if time.monotonic() > deadline:
                raise TimeoutError("Missing/incomplete ROS state; check topics, joint names, and start Isaac")
            time.sleep(0.02)
        cfg = state
    positions = np.array(base @ robot.forward_kinematics(cfg)[indices, 4:])
    command_names = list(robot.joints.actuated_names)
    if command_mimics:
        command_names += [j.name for j in urdf.robot.joints if j.mimic is not None and j.type != "fixed"]
    command_indices = jnp.array([robot.joints.names.index(name) for name in command_names])

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=3, height=3, position=(0, 0, -0.75))
    base_frame = server.scene.add_frame("/robot", show_axes=False)
    visual = ViserUrdf(server, urdf, root_node_name="/robot")
    visual.update_cfg(np.asarray(cfg))
    controls = []
    for arm, position in zip(HANDS, positions):
        controls.append(
            server.scene.add_transform_controls(
                f"/targets/{arm}",
                position=position,
                scale=0.12,
                disable_rotations=True,
                disable_axes=arm != "RU",
                disable_sliders=arm != "RU",
            )
        )
        server.scene.add_label(f"/targets/{arm}/label", text=arm)
    error = server.gui.add_number("Max position error (mm)", 0.0, disabled=True)
    attachment_status = server.gui.add_text("RU attachment", "No Isaac helper", disabled=True)
    dock = server.scene.add_icosphere("/ru_dock", radius=0.05, color=(255, 180, 30), opacity=0.25, visible=False)

    targets = positions.copy()
    ru_attached = False
    while True:
        if bridge:
            state = bridge.read()
            if state is None:
                time.sleep(0.02)
                continue
            cfg = state
            base = estimate_base(robot.forward_kinematics(cfg)[indices[1:], 4:], targets[1:])
            if bridge.attachment is not None:
                status = bridge.attachment
                anchors = jnp.array(status["anchors_m"])
                # Align Isaac world coordinates with Viser's initial-root frame.
                world_to_viser = estimate_base(anchors[1:], targets[1:])
                dock.position = np.asarray(world_to_viser @ anchors[0])
                dock.visible = True
                ru_attached = status["attached"]
                attachment_status.value = "Attached" if ru_attached else f"Free ({status['distance_m'] * 100:.1f} cm)"
                controls[0].disable_axes = controls[0].disable_sliders = ru_attached
                if ru_attached:
                    controls[0].position = dock.position
        targets[0] = controls[0].position
        cfg, base = solve_ik(robot, indices, jnp.asarray(targets), cfg, base, ru_attached)
        actual = np.asarray(base @ robot.forward_kinematics(cfg)[indices, 4:])
        # Equality constraints are numerical: reject failed/nonfinite solutions.
        if (
            bridge
            and np.isfinite(cfg).all()
            and np.isfinite(actual).all()
            and np.linalg.norm((actual - targets)[0 if ru_attached else 1:], axis=1).max() < 0.001
        ):
            bridge.publish(command_names, robot.joints.get_full_config(cfg)[command_indices])
        with server.atomic():
            base_frame.position = np.asarray(base.translation())
            base_frame.wxyz = np.asarray(base.rotation().wxyz)
            visual.update_cfg(np.asarray(cfg))
            error.value = float(np.linalg.norm(actual - targets, axis=1).max() * 1000)
        time.sleep(0.02)


if __name__ == "__main__":
    tyro.cli(main)
