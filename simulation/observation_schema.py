"""
Canonical Observation Schema

Defines a platform-agnostic observation vector and helpers to build it from
raw state, video-derived pose, or simulator observations.
"""

from typing import Iterable, Optional, Tuple

import numpy as np

DEFAULT_SPEED_OF_SOUND_M_S = 343.0

CANONICAL_OBS_FEATURES = [
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "roll", "pitch", "yaw",
    "p", "q", "r",
    "airspeed", "alpha", "beta", "mach",
    "energy_fraction",
    "aileron", "elevator", "rudder", "flaps",
]
CANONICAL_OBS_DIM = len(CANONICAL_OBS_FEATURES)


def _as_vector(
    value: Optional[Iterable[float]],
    length: int,
    default: float = 0.0,
) -> np.ndarray:
    if value is None:
        return np.full(length, default, dtype=np.float32)
    arr = np.array(value, dtype=np.float32).flatten()
    if arr.size == length:
        return arr
    if arr.size > length:
        return arr[:length]
    padded = np.full(length, default, dtype=np.float32)
    padded[:arr.size] = arr
    return padded


def _looks_like_quaternion(q: np.ndarray) -> bool:
    q = np.array(q, dtype=np.float32).flatten()
    if q.size < 4 or not np.all(np.isfinite(q[:4])):
        return False
    norm = np.linalg.norm(q[:4])
    return 0.5 < norm < 1.5


def quaternion_to_rpy(q: np.ndarray, fmt: str = "xyzw") -> np.ndarray:
    q = np.array(q, dtype=np.float32).flatten()
    if q.size < 4:
        return np.zeros(3, dtype=np.float32)
    fmt = fmt.lower()
    if fmt == "wxyz":
        w, x, y, z = q[:4]
    else:
        x, y, z, w = q[:4]

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x ** 2 + y ** 2))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y ** 2 + z ** 2))
    return np.array([roll, pitch, yaw], dtype=np.float32)


def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def estimate_aero_from_velocity(
    velocity_world: np.ndarray,
    rpy: np.ndarray,
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S,
) -> Tuple[float, float, float, float]:
    vel = _as_vector(velocity_world, 3, default=0.0)
    rpy = _as_vector(rpy, 3, default=0.0)

    roll, pitch, yaw = rpy
    rotation = rpy_to_rotation_matrix(roll, pitch, yaw)
    v_body = rotation.T @ vel

    u, v, w = float(v_body[0]), float(v_body[1]), float(v_body[2])
    airspeed = float(np.linalg.norm(v_body))
    if airspeed > 1e-6:
        alpha = float(np.arctan2(w, max(u, 1e-6)))
        beta = float(np.arcsin(np.clip(v / airspeed, -1.0, 1.0)))
    else:
        alpha = 0.0
        beta = 0.0

    speed_of_sound = max(float(speed_of_sound_m_s), 1e-3)
    mach = airspeed / speed_of_sound
    return airspeed, alpha, beta, mach


def build_canonical_observation(
    position: Optional[Iterable[float]] = None,
    velocity: Optional[Iterable[float]] = None,
    attitude: Optional[Iterable[float]] = None,
    angular_velocity: Optional[Iterable[float]] = None,
    airspeed: Optional[float] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    mach: Optional[float] = None,
    energy_fraction: Optional[float] = None,
    control_surfaces: Optional[Iterable[float]] = None,
    attitude_format: Optional[str] = None,
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S,
    energy_default: float = 1.0,
) -> np.ndarray:
    pos = _as_vector(position, 3, default=0.0)
    vel = _as_vector(velocity, 3, default=0.0)
    ang_vel = _as_vector(angular_velocity, 3, default=0.0)
    ctrl = _as_vector(control_surfaces, 4, default=0.0)

    rpy = None
    if attitude is not None:
        att = np.array(attitude, dtype=np.float32).flatten()
        if att.size >= 4 and (attitude_format or _looks_like_quaternion(att)):
            fmt = attitude_format or "xyzw"
            rpy = quaternion_to_rpy(att[:4], fmt=fmt)
        else:
            rpy = _as_vector(att, 3, default=0.0)
    if rpy is None:
        rpy = np.zeros(3, dtype=np.float32)

    if airspeed is None or alpha is None or beta is None or mach is None:
        est_airspeed, est_alpha, est_beta, est_mach = estimate_aero_from_velocity(
            vel, rpy, speed_of_sound_m_s=speed_of_sound_m_s
        )
        if airspeed is None:
            airspeed = est_airspeed
        if alpha is None:
            alpha = est_alpha
        if beta is None:
            beta = est_beta
        if mach is None:
            mach = est_mach

    if energy_fraction is None:
        energy_fraction = energy_default

    obs = np.concatenate([
        pos,
        vel,
        rpy,
        ang_vel,
        np.array([float(airspeed)], dtype=np.float32),
        np.array([float(alpha)], dtype=np.float32),
        np.array([float(beta)], dtype=np.float32),
        np.array([float(mach)], dtype=np.float32),
        np.array([float(energy_fraction)], dtype=np.float32),
        ctrl,
    ]).astype(np.float32)

    if obs.shape[0] != CANONICAL_OBS_DIM:
        raise ValueError(f"Canonical observation has shape {obs.shape}, expected {CANONICAL_OBS_DIM}")
    return obs


def canonicalize_observation(
    obs: Iterable[float],
    quaternion_format: str = "xyzw",
    speed_of_sound_m_s: float = DEFAULT_SPEED_OF_SOUND_M_S,
    energy_default: float = 1.0,
    pass_through_if_canonical: bool = True,
) -> np.ndarray:
    obs_arr = np.array(obs, dtype=np.float32).flatten()

    if pass_through_if_canonical and obs_arr.size == CANONICAL_OBS_DIM:
        return obs_arr.astype(np.float32)

    position = obs_arr[0:3] if obs_arr.size >= 3 else None
    velocity = obs_arr[3:6] if obs_arr.size >= 6 else None

    attitude = None
    attitude_format = None
    if obs_arr.size >= 10:
        maybe_quat = obs_arr[6:10]
        if _looks_like_quaternion(maybe_quat):
            attitude = maybe_quat
            attitude_format = quaternion_format
        else:
            attitude = obs_arr[6:9]
            attitude_format = None
    elif obs_arr.size >= 9:
        attitude = obs_arr[6:9]

    angular_velocity = obs_arr[10:13] if obs_arr.size >= 13 else None

    return build_canonical_observation(
        position=position,
        velocity=velocity,
        attitude=attitude,
        angular_velocity=angular_velocity,
        attitude_format=attitude_format,
        speed_of_sound_m_s=speed_of_sound_m_s,
        energy_default=energy_default,
    )
