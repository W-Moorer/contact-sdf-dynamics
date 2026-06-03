from __future__ import annotations
import time
import numpy as np
from .projection import angular_error_deg, normalize


def rmse(a, b):
    a = np.asarray(a); b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def percentile(a, q):
    return float(np.percentile(np.asarray(a), q))


def best_candidate_angle_deg(candidate_normals: list[np.ndarray], ref_normals: np.ndarray) -> np.ndarray:
    out = np.empty(len(candidate_normals))
    for i, ns in enumerate(candidate_normals):
        ns = normalize(np.asarray(ns))
        ref = normalize(ref_normals[i][None, :])[0]
        dots = np.clip(ns @ ref, -1.0, 1.0)
        out[i] = np.degrees(np.arccos(np.max(dots)))
    return out


def cone_hit_rate(candidate_normals: list[np.ndarray], ref_normals: np.ndarray, tol_deg: float = 5.0) -> float:
    ang = best_candidate_angle_deg(candidate_normals, ref_normals)
    return float(np.mean(ang <= tol_deg))


def time_call(fn, *args, repeat: int = 3, **kwargs):
    # warm-up
    fn(*args, **kwargs)
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        times.append(time.perf_counter() - t0)
    return out, float(min(times))
