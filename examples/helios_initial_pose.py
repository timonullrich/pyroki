"""Solve a starting pose for the saved scene's LU/RL/LL attachment points and RU_TARGET_M.

LU/RL/LL palms keep their scene orientation so the opened thumbs stay clear of the bars.

Reads USD only; does not connect to or modify Isaac. Fingers are fixed at zero.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics
import pyroki as pk
from scipy.spatial.transform import Rotation
import tyro
import yourdfpy

from helios_retargeting import DEFAULT_URDF, HANDS, load_helios


DEFAULT_SCENE = Path(__file__).resolve().parents[2] / "Thesis/assests/scene_helios_v0.usd"
SUPPORTS = ("LU", "RL", "LL")
# RU end-effector target in world metres, set by hand (not read from the scene).
RU_TARGET_M = (0.5799999833106995, 0.26499998569488525, 1.2086482048034668)
ARMS = (*SUPPORTS, "RU")


def read_attachments(scene_path, urdf_path):
    stage = Usd.Stage.Open(str(scene_path))
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    urdf = load_helios(urdf_path)
    targets, root_paths = [], set()
    for arm in SUPPORTS:
        joint_prim, = [p for p in stage.Traverse() if p.GetName() == f"D6Joint_{arm}"]
        joint = UsdPhysics.Joint(joint_prim)
        target = Gf.Vec3d(joint.GetLocalPos0Attr().Get())
        body0 = joint.GetBody0Rel().GetTargets()
        if body0:
            target = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(body0[0])).Transform(target)
        targets.append(np.asarray(target) * metres)

        body = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
        palm = body
        while palm and palm.GetName() != HANDS[arm][1]:
            palm = palm.GetParent()
        if not palm:
            raise ValueError(f"Cannot find the URDF palm frame for {arm}")
        relative, _ = cache.ComputeRelativeTransform(body, palm)
        offset = relative.Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        urdf.joint_map[f"ee_{arm}_fixed"].origin[:3, 3] = np.asarray(offset) * metres

        root = palm
        while root and root.GetName() != urdf.base_link:
            root = root.GetParent()
        if not root:
            raise ValueError(f"Cannot find the URDF root frame for {arm}")
        root_paths.add(root.GetPath())

    root_path, = root_paths
    root_matrix = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root_path))).T.copy()
    root_matrix[:3, 3] *= metres
    if not np.allclose(root_matrix[:3, :3].T @ root_matrix[:3, :3], np.eye(3), atol=1e-6):
        raise ValueError("The robot root must have unit scale")
    # Rebuild yourdfpy's scene graph after replacing the attachment offsets.
    return yourdfpy.URDF(urdf.robot, load_meshes=False), np.array(targets), root_matrix, str(root_path)


def support_orientations(robot, indices, joints, base):
    """World orientations of the LU/RL/LL palms."""
    return base.rotation() @ jaxlie.SO3(robot.forward_kinematics(joints)[indices[:len(SUPPORTS)], :4])


@jdc.jit
def solve_initial_pose(robot, indices, targets, seed_base):
    joints, base = robot.joint_var_cls(0), jaxls.SE3Var(0)
    seed_joints = robot.joint_var_cls.default_factory()
    # Keep the support palms as oriented in the scene, where the opened thumbs clear the bars.
    seed_orientations = support_orientations(robot, indices, seed_joints, seed_base)

    @jaxls.Cost.factory(kind="constraint_eq_zero")
    def attachments(vals, joints, base):
        positions = vals[base] @ robot.forward_kinematics(vals[joints])[indices, 4:]
        orientations = support_orientations(robot, indices, vals[joints], vals[base])
        return jnp.concatenate([((positions - targets) * 1000.0).flatten(),
                                (seed_orientations.inverse() @ orientations).log().flatten()])

    @jaxls.Cost.factory
    def stay_near_seed(vals, joints, base):
        return jnp.concatenate([vals[joints] - seed_joints, (seed_base.inverse() @ vals[base]).log()])

    return jaxls.LeastSquaresProblem(
        [attachments(joints, base), pk.costs.limit_constraint(robot, joints), stay_near_seed(joints, base)],
        [joints, base],
    ).analyze().solve(
        initial_vals=jaxls.VarValues.make([joints.with_value(seed_joints), base.with_value(seed_base)]),
        linear_solver="dense_cholesky", verbose=False,
    )


def main(
    scene_path: Path = DEFAULT_SCENE,
    urdf_path: Path = DEFAULT_URDF,
    output: Path = Path("helios_initial_pose.json"),
):
    """Print and save base pose, non-finger joint positions, and attachment errors."""
    jax.config.update("jax_enable_x64", True)
    urdf, targets, seed_matrix, root_path = read_attachments(scene_path, urdf_path)
    targets = np.vstack([targets, RU_TARGET_M])
    robot = pk.Robot.from_urdf(urdf, default_joint_cfg=urdf.zero_cfg)
    indices = jnp.array([robot.links.names.index(f"ee_{arm}") for arm in ARMS])
    seed_base = jaxlie.SE3.from_matrix(jnp.asarray(seed_matrix))
    initial_positions = np.asarray(seed_base @ robot.forward_kinematics(robot.joint_var_cls.default_factory())[indices, 4:])
    solution = solve_initial_pose(robot, indices, jnp.asarray(targets), seed_base)
    cfg, base = solution[robot.joint_var_cls(0)], solution[jaxls.SE3Var(0)]
    full = dict(zip(robot.joints.names, np.asarray(robot.joints.get_full_config(cfg))))
    positions = {j.name: float(full[j.name]) for j in urdf.robot.joints if j.type != "fixed"}

    # Independently verify the solution with yourdfpy FK, including mimic joints.
    urdf.update_cfg(dict(zip(robot.joints.actuated_names, np.asarray(cfg))))
    base_matrix = np.asarray(base.as_matrix())
    verified = np.array([(base_matrix @ urdf.get_transform(f"ee_{arm}"))[:3, 3] for arm in ARMS])
    errors = np.linalg.norm(verified - targets, axis=1)
    seed_rotations = Rotation.from_matrix(np.asarray(support_orientations(
        robot, indices, robot.joint_var_cls.default_factory(), seed_base).as_matrix()))
    turned = np.degrees((seed_rotations.inv() * Rotation.from_matrix(np.array(
        [(base_matrix @ urdf.get_transform(f"ee_{arm}"))[:3, :3] for arm in SUPPORTS]))).magnitude())
    limits_ok = all(j.limit.lower - 1e-6 <= positions[j.name] <= j.limit.upper + 1e-6
                    for j in urdf.robot.joints if j.type == "revolute")
    if (not np.isfinite(base_matrix).all() or not np.isfinite(errors).all() or errors.max() > 1e-4
            or not np.isfinite(turned).all() or turned.max() > 0.01 or not limits_ok):
        raise RuntimeError(f"No validated pose: attachment errors {errors * 1000} mm, "
                           f"support palms turned {turned} deg, joint limits OK: {limits_ok}")

    report = {
        "scene": str(scene_path.resolve()),
        "base_frame": urdf.base_link,
        "base_prim": root_path,
        "base_position_world_m": base_matrix[:3, 3].tolist(),
        "base_orientation_world_wxyz": np.asarray(base.rotation().wxyz).tolist(),
        "base_rotation_world_xyz_deg": Rotation.from_matrix(base_matrix[:3, :3]).as_euler("xyz", degrees=True).tolist(),
        "joint_positions_rad": positions,
        "joint_positions_deg": {name: float(np.degrees(value)) for name, value in positions.items()},
        "attachments": {
            arm: {"target_world_m": targets[i].tolist(), "initial_error_mm": float(np.linalg.norm(initial_positions[i] - targets[i]) * 1000),
                  "solved_error_mm": float(errors[i] * 1000)}
            for i, arm in enumerate(ARMS)
        },
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    tyro.cli(main)
