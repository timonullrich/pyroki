# Helios IK

ROS 2 controller using the included Pyroki library. Requires Python 3.12,
ROS Jazzy and Isaac Sim with its ROS 2 Bridge and
`isaacsim.code_editor.python_server` enabled.

```bash
source /opt/ros/jazzy/setup.bash
source .venv/bin/activate
python -m pip install -e '.[helios]'
python examples/helios_retargetiing.py
```

Start the controller with Isaac **stopped**, then press Play. LU/RL/LL must
be attached and the fingers held at zero. After initialization, RU automatically releases, retreats along
world +Y, travels with clearance, approaches along -Y, and automatically
reattaches within 2 cm. A released hand must first leave that radius.
Press q or Ctrl+C to exit. Restart after resetting the scene.

Configuration:

- `examples/helios_retargetiing.py`: `RU_TARGET_M`, `D6_OFFSET`, `COST_WEIGHTS`.
- `examples/helios.yaml`: motor gains, clearance, durations and tracking tolerances.

Target, gain and attachment overrides are temporary. Stop Isaac or exit the
controller to remove them before editing the underlying scene target.
Joint feedback and Isaac's root pose initialize IK once; subsequent solves use the previous solution.
Attached positions are hard constraints and orientations have soft penalties.

Run RViz in another sourced terminal:

```bash
rviz2 -d examples/helios.rviz
```

Blue is commanded; orange is measured. Use Move Camera to orbit/pan/zoom.
`--no-rviz` disables publishing. `--help` lists the remaining path/port options.
The ROS topics are `/joint_states` and `/joint_command`; coupled joints are commanded.

Each transfer saves `summary.json`, `rollout.csv` and `metrics.png` under
`helios_runs/` (`--metrics-dir` overrides this). CSV uses semicolons; exported
floats use four significant figures. The summary includes the destination in world metres (`settings.target_world_m`). Only plots are smoothed. Missing sensor
readings are NaN. Costs before/after solving are the model's soft objective;
hard-constraint violations are shown separately.

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
targets, and zero velocities. LU/RL/LL are enabled and RU is disabled. Start the
retargeting controller afterward, then press Play. Joint states take effect when
physics initializes, using Isaac's [JointStateAPI](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#jointstateapi-for-set-and-get-of-joint-position-and-velocity).
The setup remains in a temporary session layer across Stop/Play; rerunning the
script replaces it, and reloading the scene discards it. No USD files are saved.
Use `--pose-path` for another pose JSON or `--isaac-port` for another server port.

For this scene, the reported world base pose also applies to
`/World/helios_quadmanual_flat`, whose parent has an identity transform.
The solution removes attachment-position mismatch; it is not a static force-balance solve.

The Pyroki library and its original license are retained; see `LICENSE`.
