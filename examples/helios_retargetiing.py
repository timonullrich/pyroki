"""Helios whole-body IK: drag RU while the other hand positions stay fixed.

Run: python examples/helios_retargetiing.py
Uses the neighbouring Thesis checkout's URDF/meshes; override with --urdf-path.
"""

import time
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
def solve_ik(robot: pk.Robot, link_indices, targets, previous_cfg, previous_base):
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
                    joint_var, base_var, link_indices[1:], targets[1:]
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


def main(urdf_path: Path = DEFAULT_URDF):
    urdf = load_helios(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    indices = jnp.array([robot.links.names.index(f"ee_{arm}") for arm in HANDS])
    cfg = robot.joint_var_cls.default_factory()
    base = jaxlie.SE3.identity()
    positions = np.asarray(robot.forward_kinematics(cfg)[indices, 4:])

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

    targets = positions.copy()
    while True:
        targets[0] = controls[0].position
        cfg, base = solve_ik(robot, indices, jnp.asarray(targets), cfg, base)
        actual = np.asarray(base @ robot.forward_kinematics(cfg)[indices, 4:])
        with server.atomic():
            base_frame.position = np.asarray(base.translation())
            base_frame.wxyz = np.asarray(base.rotation().wxyz)
            visual.update_cfg(np.asarray(cfg))
            error.value = float(np.linalg.norm(actual - targets, axis=1).max() * 1000)
        time.sleep(0.02)


if __name__ == "__main__":
    tyro.cli(main)
