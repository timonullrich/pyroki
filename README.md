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
be attached and the fingers held at zero. Space releases RU, retreats along
world +Y, travels with clearance, approaches along -Y, and reattaches.
Press q or Ctrl+C to exit. Restart after resetting the scene.

Configuration:

- `examples/helios_retargetiing.py`: `RU_TARGET_M`, `D6_OFFSET`, `COST_WEIGHTS`.
  `ru_roll` penalizes RU forearm tilt in global Y-Z; 0 means horizontal.
- `examples/helios.yaml`: motor gains, clearance, durations and tracking tolerances.

Target, gain and attachment overrides are temporary. Stop Isaac or exit the
controller to remove them before editing the underlying scene target.
Joint feedback initializes IK once; subsequent solves use the previous solution.
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
floats use four significant figures. Only plots are smoothed. Missing sensor
readings are NaN. Costs before/after solving are the model's soft objective;
hard-constraint violations are shown separately.

Offline checks: `.venv/bin/python -m unittest discover -s tests -p 'test_helios_*.py'`.
The Pyroki library and its original license are retained; see `LICENSE`.
