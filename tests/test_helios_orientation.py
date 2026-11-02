"""Orientation preferences must remain soft even when exact rotation is impossible."""

from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import jax.numpy as jnp
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy


class OrientationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        examples = Path(__file__).resolve().parents[1] / "examples"
        with patch.object(sys, "path", [str(examples), *sys.path]):
            module = runpy.run_path(str(examples / "helios_retargetiing.py"))
        cls.solve = staticmethod(module["solve_ik"])
        cls.weights = module["COST_WEIGHTS"]
        cls.weights["ru_roll"] = 1.0  # Exercise the penalty even if disabled in the user's settings.
        # Three noncollinear support points fix the base. The RU slider can move,
        # but no joint can rotate a hand: conflicting orientations are infeasible.
        joints = [yourdfpy.Joint(
            name="slider", type="prismatic", parent="root", child="RU", axis=np.array([1., 0, 0]),
            origin=np.eye(4), limit=yourdfpy.Limit(lower=-1, upper=1, effort=1, velocity=1),
        )]
        for name, position in zip(("LU", "RL", "LL"), np.eye(3)):
            origin = np.eye(4)
            origin[:3, 3] = position
            joints.append(yourdfpy.Joint(name=name, type="fixed", parent="root", child=name, origin=origin))
        urdf = yourdfpy.URDF(yourdfpy.Robot(
            name="orientation_test", links=[yourdfpy.Link(name=n) for n in ("root", "RU", "LU", "RL", "LL")],
            joints=joints,
        ), load_meshes=False)
        cls.robot = pk.Robot.from_urdf(urdf)
        cls.indices = jnp.array([cls.robot.links.names.index(n) for n in ("RU", "LU", "RL", "LL")])
        cls.cfg = cls.robot.joint_var_cls.default_factory()
        cls.base = jaxlie.SE3.identity()
        poses = cls.robot.forward_kinematics(cls.cfg)[cls.indices]
        cls.targets = poses[:, 4:].at[0, 0].set(0.2)
        cls.rotations = poses[:, :4]

    def test_impossible_orientation_does_not_override_hard_positions(self):
        rotations = jnp.tile(jaxlie.SO3.from_z_radians(0.3).wxyz, (4, 1))
        for attached in (False, True):
            with self.subTest(ru_attached=attached):
                # Once RU is attached the seed is already at its docking position.
                seed = self.cfg.at[0].set(0.2) if attached else self.cfg
                cfg, base, steps, costs = self.solve(self.robot, self.indices, self.targets, rotations,
                                                    seed, self.base, attached, 25.0)
                self.assertGreater(int(steps), 0)
                self.assertTrue(all(np.isfinite(value) and value >= 0 for value in costs.values()))
                expected_total = sum(costs[name] for name in ("ru_position", "fixed_orientations", "ru_roll", "upright", "motion"))
                self.assertAlmostEqual(float(costs["total_objective"]), float(expected_total), places=4)
                self.assertGreater(float(costs["fixed_orientations"]), 100)
                self.assertLess(float(costs["fixed_positions_violation"]), 0.01)
                poses = self.robot.forward_kinematics(cfg)[self.indices]
                positions = np.array(base @ poses[:, 4:])
                errors = np.linalg.norm(positions - self.targets, axis=1)
                self.assertTrue(np.isfinite(cfg).all())
                self.assertLess(errors.max(), 0.001)
                rotation_error = (jaxlie.SO3(rotations).inverse() @ (base.rotation() @ jaxlie.SO3(poses[:, :4]))).log()
                self.assertGreater(np.linalg.norm(np.array(rotation_error)[1:], axis=1).min(), 0.29)

    def test_free_ru_orientation_has_no_penalty(self):
        changed = self.rotations.at[0].set(jaxlie.SO3.from_z_radians(1.0).wxyz)
        solutions = [self.solve(self.robot, self.indices, self.targets, rotations,
                               self.cfg, self.base, False, 25.0) for rotations in (self.rotations, changed)]
        np.testing.assert_allclose(solutions[0][0], solutions[1][0], atol=1e-6)
        np.testing.assert_allclose(solutions[0][1].wxyz_xyz, solutions[1][1].wxyz_xyz, atol=1e-6)

    def test_objective_matches_explicit_ik_formula(self):
        cfg, base, _, costs = self.solve(self.robot, self.indices, self.targets, self.rotations,
                                         self.cfg, self.base, False, 25.0)
        poses = self.robot.forward_kinematics(cfg)[self.indices]
        ru_error = base @ poses[0, 4:] - self.targets[0]
        rotation_error = (jaxlie.SO3(self.rotations[1:]).inverse() @ (base.rotation() @ jaxlie.SO3(poses[1:, :4]))).log()
        upright_error = base.rotation().as_matrix()[:, 2] - jnp.array([0, 0, 1])
        motion = jnp.concatenate([cfg - self.cfg, (self.base.inverse() @ base).log()])
        ru_matrix = (base.rotation() @ jaxlie.SO3(poses[0, :4])).as_matrix()
        roll = jnp.arctan2(ru_matrix[2, 2], jnp.abs(ru_matrix[1, 2]) + 1e-8)
        objective = (jnp.sum((self.weights["position"] * ru_error)**2) + jnp.sum((25 * rotation_error)**2)
                     + (self.weights["ru_roll"] * roll)**2 + jnp.sum((self.weights["upright"] * upright_error)**2)
                     + jnp.sum((self.weights["motion"] * motion)**2))
        np.testing.assert_allclose(costs["total_objective"], objective, rtol=1e-5)
        # Initially the hand points straight up, in addition to the 20 cm displacement.
        expected_initial = (self.weights["position"] * 0.2)**2 + (self.weights["ru_roll"] * np.pi / 2)**2
        self.assertAlmostEqual(float(costs["initial_objective"]), expected_initial, places=4)
        self.assertLess(float(costs["total_objective"]), float(costs["initial_objective"]))

    def test_ru_roll_uses_world_frame_and_stays_soft(self):
        for roll, direction in ((0.0, 1), (0.4, 1), (-0.4, 1), (0.0, -1)):
            # Local +Z points along +/-world Y, tilted upwards/downwards. Twisting
            # about local Z must not change the pointing-direction penalty.
            ik_to_world = (jaxlie.SO3.from_x_radians(direction * (-np.pi / 2 + roll))
                           @ jaxlie.SO3.from_z_radians(0.6))
            cfg, base, _, costs = self.solve(self.robot, self.indices, self.targets, self.rotations,
                                             self.cfg, self.base, False, 25.0, ik_to_world)
            self.assertAlmostEqual(float(costs["initial_objective"]),
                                   (self.weights["position"] * 0.2)**2 + (self.weights["ru_roll"] * roll)**2, places=3)
            positions = base @ self.robot.forward_kinematics(cfg)[self.indices, 4:]
            self.assertLess(np.linalg.norm(np.asarray(positions - self.targets), axis=1).max(), 0.001)
            if roll and self.weights["ru_roll"]:
                # This robot cannot rotate RU independently of the fixed supports.
                self.assertGreater(float(costs["ru_roll"]), 0.1)
            else:
                self.assertLess(float(costs["ru_roll"]), 1e-6)


if __name__ == "__main__":
    unittest.main()
