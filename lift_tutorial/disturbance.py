"""World-frame horizontal force pulse, scheduled in integer control intervals."""
import math

FORCE_FIELDS = ["push_fx_n", "push_fy_n", "push_fz_n"]


def validate_disturbance(value):
    if not isinstance(value, dict) or set(value) != {"start_s", "duration_s", "force_n"}:
        raise ValueError("disturbance requires start_s, duration_s, force_n")
    for key in ("start_s", "duration_s"):
        v = value[key]
        if type(v) not in (int, float) or not math.isfinite(v) or v < 0:
            raise ValueError(f"disturbance.{key} must be finite and nonnegative")
    if value["duration_s"] <= 0:
        raise ValueError("disturbance.duration_s must be positive")
    force = value["force_n"]
    if not isinstance(force, list) or len(force) != 3 or any(
            type(v) not in (int, float) or not math.isfinite(v) for v in force):
        raise ValueError("disturbance.force_n requires three finite numbers")
    if force[2] != 0:
        raise ValueError("Only horizontal forces are supported (Fz must be zero)")
    if math.hypot(*force[:2]) > 5:
        raise ValueError("Tutorial force magnitude must be at most 5 N")
    return value


def pulse_steps(value, frequency, horizon):
    validate_disturbance(value)
    start = value["start_s"] * frequency
    duration = value["duration_s"] * frequency
    if not math.isclose(start, round(start), abs_tol=1e-8) or not math.isclose(duration, round(duration), abs_tol=1e-8):
        raise ValueError("Pulse times must align with control intervals (0.05 s at 20 Hz)")
    start, duration = round(start), round(duration)
    if duration < 1 or start + duration > horizon:
        raise ValueError("Pulse must fit inside episode horizon")
    return start, duration


def force_for_step(value, step, frequency, horizon):
    if value is None:
        return [0.0, 0.0, 0.0]
    start, duration = pulse_steps(value, frequency, horizon)
    # Step 1 integrates [0, dt); a pulse at 0.5 s starts at step 11 (20 Hz).
    return list(value["force_n"]) if start <= step - 1 < start + duration else [0.0, 0.0, 0.0]
