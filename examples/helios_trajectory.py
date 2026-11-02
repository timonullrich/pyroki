"""Smooth RU clearance motion, independent of ROS and the IK implementation.

This prescribes a clearance path; it is not a collision planner.
"""

import numpy as np


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
        tolerances = np.array([self.waypoint_tolerance, self.tracking_tolerance])
        if not np.isfinite(tolerances).all() or np.any(tolerances <= 0):
            raise ValueError("Trajectory tolerances must be positive and finite")
        self.phase = 0
        self.elapsed = 0.0

    @property
    def done(self):
        return self.phase == 3

    def sample(self, elapsed):
        if self.done:
            return self.waypoints[-1].copy()
        u = np.clip(elapsed / self.durations[self.phase], 0.0, 1.0)
        # Quintic time scaling: zero velocity and acceleration at both ends.
        blend = u**3 * (10.0 - 15.0 * u + 6.0 * u**2)
        return self.waypoints[self.phase] + blend * (self.waypoints[self.phase + 1] - self.waypoints[self.phase])

    def propose(self, dt, hand):
        """Do not commit progress until the corresponding IK command is accepted."""
        if self.done:
            return self.sample(0.0), self.elapsed
        elapsed = self.elapsed
        if np.linalg.norm(np.asarray(hand) - self.sample(elapsed)) <= self.tracking_tolerance:
            # Slow frames/JIT compilation must not skip large sections of the path.
            elapsed = min(elapsed + np.clip(dt, 0.0, 0.05), self.durations[self.phase])
        return self.sample(elapsed), elapsed

    def commit(self, elapsed, hand):
        if self.done:
            return
        self.elapsed = elapsed
        if (
            elapsed >= self.durations[self.phase]
            and np.linalg.norm(np.asarray(hand) - self.waypoints[self.phase + 1]) <= self.waypoint_tolerance
        ):
            self.phase += 1
            self.elapsed = 0.0
