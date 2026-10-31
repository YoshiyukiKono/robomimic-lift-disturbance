"""Task-specific metrics, independent of the policy and rendering libraries."""

import math


class HoldMetric:
    """Success requires consecutive samples at/above a relative lift threshold.

    This measures cube height, not finger contacts or proof of a secure grasp.
    Every update represents one completed control interval, in simulation time.
    """

    def __init__(self, initial_z, lift_height_m, hold_seconds, control_freq):
        values = (initial_z, lift_height_m, hold_seconds, control_freq)
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("Metric inputs must be finite")
        if min(lift_height_m, hold_seconds, control_freq) <= 0:
            raise ValueError("Height, hold time, and control frequency must be positive")
        self.initial_z = float(initial_z)
        self.threshold = float(lift_height_m)
        self.control_freq = float(control_freq)
        self.required_steps = max(1, math.ceil(hold_seconds * control_freq - 1e-9))
        self.consecutive_steps = 0
        self.max_consecutive_steps = 0
        self.steps = 0
        self.max_height_m = 0.0
        self.time_to_success_s = None

    def update(self, cube_z):
        if not math.isfinite(float(cube_z)):
            raise ValueError("Non-finite cube position")
        self.steps += 1
        relative_height = float(cube_z) - self.initial_z
        self.max_height_m = max(self.max_height_m, relative_height)
        if relative_height >= self.threshold:
            self.consecutive_steps += 1
        else:
            self.consecutive_steps = 0
        self.max_consecutive_steps = max(self.max_consecutive_steps, self.consecutive_steps)
        if self.time_to_success_s is None and self.consecutive_steps >= self.required_steps:
            self.time_to_success_s = self.steps / self.control_freq
        return self.success

    @property
    def success(self):
        return self.time_to_success_s is not None

    @property
    def max_hold_s(self):
        return self.max_consecutive_steps / self.control_freq
