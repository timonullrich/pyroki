# Helios IK

ROS 2 controller using the included Pyroki library. Requires Python 3.12,
ROS Jazzy and Isaac Sim with its ROS 2 Bridge and
`isaacsim.code_editor.python_server` enabled.

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m pip install -e '.[helios]'
python examples/helios_retargeting.py
```

Start the controller with Isaac **stopped**, then press Play. LU/RL/LL must
be attached and settled. RU automatically retreats along world +Y, travels with
clearance, approaches along -Y, then holds its target with the hand open.
Press **c** in the controller terminal (no Enter needed) whenever you want RU
to start closing slowly. Torque ramps smoothly from zero to full force over three
simulation seconds, with closing speed capped at 0.35 rad/s (about 20°/s).
Full torque then continues until you exit or stop Isaac. Press **q** or Ctrl+C to exit. Restart after resetting
the scene.

RU uses physical finger contact only. Its old D6 joint is masked in the temporary
session layer if present; it can also be deleted from the scene. There is no
proximity attachment or automatic closing. A closing command does not confirm
that an object has been grasped. LU/RL/LL retain their existing attachments.

Configuration:

- `examples/helios_retargeting.py`: `RU_TARGET_M`, `HAND_OFFSET_M`, `COST_WEIGHTS`.
- `examples/helios.yaml`: actuators, clearance, durations and tracking tolerances.
  `actuators` uses Isaac Lab `ImplicitActuatorCfg` fields and SI units (per radian),
  matching `helios_rl`'s `helios.py`. Each controller start writes them, converted to
  USD's per-degree drive units, into the temporary session layer for every matched joint.
  `grasp.closing_speed_rad_s` sets how fast the RU closing targets move and limits
  those joints' speed. Lower it for a gentler grasp. It must be positive and finite;
  restart the controller with Isaac stopped after changing it.

The hardcoded closing directions are in `examples/helios_grasp_isaac.py`.
Closing keeps the configured implicit PD drives (`stiffness`, `damping` in
`helios.yaml`) and moves each closing joint's position target toward its closing
joint limit, reapplied every physics step. In contact, the drive torque grows with
the target lag and is capped by the drive's `maxForce`, i.e. the joint's
`joint_effort_limit`: 0.78 N·m at finger MCP joints, 1.29 N·m at finger PIP joints,
and 1.09/0.58/1.95/1.95 N·m at the thumb MCP/ABD/PIP/DIP joints. Full torque needs
contact at least `joint_effort_limit / stiffness` before the limit (about 4–5° for
fingers, 32° for thumb PIP/DIP). Finger abduction stays at zero and the wrist follows
IK. Exit sets the closing targets back to zero (open). Joint limits remain enforced by physics.

Gain and legacy-joint overrides are temporary. Stop Isaac or exit the controller
to remove them. `RU_TARGET_M` is an independent world-space target in metres.
Joint feedback and Isaac's root pose initialize IK once; subsequent solves use the previous solution.
LU/RL/LL positions are hard constraints and their orientations have soft penalties.
RU remains a soft position target, including while grasping.

Run RViz in another sourced terminal:

```bash
rviz2 -d examples/helios.rviz
```

Blue is commanded; orange is measured. Use Move Camera to orbit/pan/zoom.
`--no-rviz` disables publishing. `--help` lists the remaining path/port options.
The ROS topics are `/joint_states` and `/joint_command`; coupled joints are commanded.

Each completed approach saves `summary.json`, `rollout.csv` and `metrics.png` under
`helios_runs/` (`--metrics-dir` overrides this). CSV uses semicolons; exported
floats use four significant figures. The summary includes the destination in world metres (`settings.target_world_m`). Only plots are smoothed. Missing sensor
readings are NaN. Costs before/after solving are the model's soft objective;
hard-constraint violations are shown separately.

To compare IK approaches offline (no ROS or Isaac needed), run:

```bash
python examples/helios_compare_runs.py --approaches testsv3 testsv3.1 testsv4 testsv5
```

Each approach is a `helios_runs/config_<name>` folder; every run inside it is
included, so new runs appear automatically. It writes tracking error, wrist force and
torque, actuator effort, computation time and solver step charts plus `summary.csv` to
`helios_runs/comparison/` (`--out-dir` overrides this). Add `--completed-only` to skip interrupted runs.

Offline checks: `.venv/bin/python -m unittest discover -s tests -p 'test_helios_*.py'`.

To calculate an initial pose aligned with the saved scene's LU/RL/LL attachments:

```bash
python examples/helios_initial_pose.py
```

This reads `scene_helios_v0.usd` and writes `helios_initial_pose.json` with the
world base pose, joint positions in radians/degrees, and attachment errors.
Use `--scene-path` for another saved scene. It does not edit USD or contact Isaac.
Fingers are excluded; RU is unconstrained. Initialize with RU detached, zero
velocities, and joint drive targets matching the solved joint positions.
To apply these values through Isaac's Python server, stop Isaac and exit the
retargeting controller, then run:

```bash
python examples/helios_apply_initial_pose.py
```

The script applies the base pose, non-finger initial joint states, matching drive
targets, and zero velocities. It also opens the LU/RL/LL thumbs (`SUPPORT_THUMB_DEG`:
abduction −15°, MCP −25°) as both initial state and drive target: at 0° they start
4–8 mm inside the bars, which makes the contacts fight every step. LU/RL/LL are enabled and any legacy RU D6 is masked. Start the
retargeting controller afterward, then press Play. Joint states take effect when
physics initializes, using Isaac's [JointStateAPI](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#jointstateapi-for-set-and-get-of-joint-position-and-velocity).
The setup remains in a temporary session layer across Stop/Play; rerunning the
script replaces it, and reloading the scene discards it. No USD files are saved.
Use `--pose-path` for another pose JSON or `--isaac-port` for another server port.

For this scene, the reported world base pose also applies to
`/World/helios_quadmanual_flat_inertias` (the old `/World/helios_quadmanual_flat` is deactivated), whose parent has an identity transform.
The solution removes attachment-position mismatch; it is not a static force-balance solve.

The Pyroki library and its original license are retained; see `LICENSE`.
