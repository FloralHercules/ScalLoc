"""Geometry from the validated fast ScalLoc pipeline."""
from __future__ import annotations
import glob
import importlib.util
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from contextlib import redirect_stdout
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
os.environ.setdefault("PYTHONHASHSEED", "42")
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRotation
from tqdm.auto import tqdm
from .configuration import *


def wrap_deg(value: float) -> float:
    return float((float(value) + 180.0) % 360.0 - 180.0)


def apply_sim2(matrix: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    mat = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    return pts @ mat[:, :2].T + mat[:, 2]


def compose_sim2(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Return outer(inner(x)) for two 2x3 similarities."""
    a = np.asarray(outer, dtype=np.float64).reshape(2, 3)
    b = np.asarray(inner, dtype=np.float64).reshape(2, 3)
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = a[:, :2] @ b[:, :2]
    out[:, 2] = a[:, :2] @ b[:, 2] + a[:, 2]
    return out


def invert_sim2(matrix: np.ndarray) -> np.ndarray:
    """Return the inverse of a non-degenerate 2x3 similarity."""
    mat = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    linear_inv = np.linalg.inv(mat[:, :2])
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = linear_inv
    out[:, 2] = -linear_inv @ mat[:, 2]
    return out


def sim2_from_pivot(
    pivot_local_xy: np.ndarray,
    pivot_utm_xy: np.ndarray,
    scale: float,
    yaw_deg: float,
) -> np.ndarray:
    """Build local->UTM Sim(2) from its mapping at a shared local pivot."""
    theta = math.radians(float(yaw_deg))
    linear = float(scale) * np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ], dtype=np.float64)
    pivot_local = np.asarray(pivot_local_xy, dtype=np.float64).reshape(2)
    pivot_utm = np.asarray(pivot_utm_xy, dtype=np.float64).reshape(2)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = linear
    matrix[:, 2] = pivot_utm - linear @ pivot_local
    return matrix


def blend_sim2_at_local_pivot(
    start_matrix: np.ndarray,
    end_matrix: np.ndarray,
    pivot_local_xy: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Blend two similarities in position/yaw/log-scale without matrix collapse."""
    weight = float(np.clip(alpha, 0.0, 1.0))
    pivot = np.asarray(pivot_local_xy, dtype=np.float64).reshape(1, 2)
    start_position = apply_sim2(start_matrix, pivot)[0]
    end_position = apply_sim2(end_matrix, pivot)[0]
    start_scale, start_yaw = sim2_scale_yaw(start_matrix)
    end_scale, end_yaw = sim2_scale_yaw(end_matrix)
    position = (1.0 - weight) * start_position + weight * end_position
    log_scale = (1.0 - weight) * math.log(max(start_scale, 1e-12)) + weight * math.log(max(end_scale, 1e-12))
    yaw = circular_weighted_mean([start_yaw, end_yaw], [1.0 - weight, weight])
    return sim2_from_pivot(pivot_local_xy, position, math.exp(log_scale), yaw)


def sim2_scale_yaw(matrix: np.ndarray) -> tuple[float, float]:
    mat = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    scale = float(math.sqrt(max(abs(np.linalg.det(mat[:, :2])), 0.0)))
    theta = float(math.degrees(math.atan2(mat[1, 0], mat[0, 0])))
    return scale, wrap_deg(theta)


def satellite_residual_for_pose(
    residual: np.ndarray,
    pivot_utm: np.ndarray,
    apply_yaw: bool | None = None,
    apply_scale: bool | None = None,
) -> np.ndarray:
    """Keep enabled Sim(2) components while preserving the raw mapping at a UTM pivot."""
    raw_scale, raw_yaw = sim2_scale_yaw(residual)
    use_yaw = ANCHOR_APPLY_SATELLITE_YAW if apply_yaw is None else bool(apply_yaw)
    use_scale = ANCHOR_APPLY_SATELLITE_SCALE if apply_scale is None else bool(apply_scale)
    scale = float(raw_scale) if use_scale else 1.0
    yaw_deg = float(raw_yaw) if use_yaw else 0.0
    yaw_rad = math.radians(yaw_deg)
    raw = np.asarray(residual, dtype=np.float64).reshape(2, 3)
    pivot = np.asarray(pivot_utm, dtype=np.float64).reshape(2)
    result = np.zeros((2, 3), dtype=np.float64)
    result[:, :2] = scale * np.array([
        [math.cos(yaw_rad), -math.sin(yaw_rad)],
        [math.sin(yaw_rad), math.cos(yaw_rad)],
    ])
    # The raw translation is tied to raw scale/yaw and cannot be reused after
    # disabling either component. Keep the raw residual's mapping at pivot.
    target_at_pivot = raw[:, :2] @ pivot + raw[:, 2]
    result[:, 2] = target_at_pivot - result[:, :2] @ pivot
    return result


def accepted_anchor_component_history(
    states: list["WindowState"],
    key: str,
    limit: int,
) -> list[float]:
    """Return recent finite raw satellite components from accepted windows."""
    values: list[float] = []
    for state in reversed(states):
        if not state.anchor_accepted:
            continue
        value = state.anchor_stats.get(key)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        values.append(value)
        if len(values) >= limit:
            break
    return list(reversed(values))


def component_consensus(
    current: float,
    history: list[float],
    min_history: int,
    tolerance: float,
    circular: bool = False,
) -> tuple[bool, float | None, float | None]:
    """Check whether a component agrees with enough accepted anchor history."""
    if len(history) < int(min_history):
        return False, None, None
    reference = float(np.median(np.asarray(history, dtype=np.float64)))
    if circular:
        deviation = abs(wrap_deg(float(current) - reference))
    else:
        deviation = abs(float(current) / max(abs(reference), 1e-9) - 1.0)
    return bool(deviation <= float(tolerance)), reference, float(deviation)


def update_sequential_anchor_scale(
    current_multiplier: float,
    current_log_variance: float,
    raw_anchor_scale: float,
    inlier_ratio: float,
    inlier_count: float,
    rmse_m: float,
    enabled: bool | None = None,
) -> tuple[float, float, dict[str, Any]]:
    """One causal scale update; its result is used from the next window only."""
    current = float(current_multiplier)
    raw = float(raw_anchor_scale)
    filter_enabled = (
        bool(SEQUENTIAL_ANCHOR_SCALE_FILTER_ENABLED)
        if enabled is None else bool(enabled)
    )
    diagnostics: dict[str, Any] = {
        "enabled": filter_enabled,
        "multiplier_before": current,
        "log_variance_before": float(current_log_variance),
        "raw_anchor_scale": raw,
    }
    if not filter_enabled:
        diagnostics["reason"] = "disabled"
        return current, float(current_log_variance), diagnostics
    if not (np.isfinite(current) and current > 0.0 and np.isfinite(raw) and raw > 0.0):
        diagnostics["reason"] = "invalid_scale"
        return current, float(current_log_variance), diagnostics
    if not (SEQUENTIAL_ANCHOR_SCALE_MIN_RAW <= raw <= SEQUENTIAL_ANCHOR_SCALE_MAX_RAW):
        diagnostics["reason"] = "raw_scale_outside_causal_filter_range"
        return current, float(current_log_variance), diagnostics
    # Scalar Kalman update in log-scale.  The state begins uncertain, so an
    # early high-quality anchor can establish metric scale; later anchors only
    # refine it.  Low-support/high-RMSE matches receive larger R automatically.
    ratio_quality = float(np.clip((float(inlier_ratio) - MINIMA_MIN_INLIER_RATIO) / 0.45, 0.0, 1.0))
    count_quality = float(np.clip(float(inlier_count) / 1200.0, 0.0, 1.0))
    quality = float(math.sqrt(ratio_quality * count_quality))
    rmse_fraction = float(np.clip(float(rmse_m) / max(MINIMA_MAX_RMSE_M, 1e-6), 0.0, 1.0))
    measurement_std = float(np.clip(
        SEQUENTIAL_ANCHOR_SCALE_MIN_MEASUREMENT_STD_LOG
        + 0.14 * (1.0 - quality)
        + 0.035 * rmse_fraction,
        SEQUENTIAL_ANCHOR_SCALE_MIN_MEASUREMENT_STD_LOG,
        SEQUENTIAL_ANCHOR_SCALE_MAX_MEASUREMENT_STD_LOG,
    ))
    predicted_variance = max(float(current_log_variance), 1e-12) + (
        SEQUENTIAL_ANCHOR_SCALE_PROCESS_STD_LOG ** 2
    )
    measurement_variance = measurement_std ** 2
    gain = float(predicted_variance / (predicted_variance + measurement_variance))
    updated_log = math.log(current) + gain * math.log(raw)
    proposed = math.exp(updated_log)
    updated = float(np.clip(
        proposed,
        SEQUENTIAL_ANCHOR_SCALE_MIN_MULTIPLIER,
        SEQUENTIAL_ANCHOR_SCALE_MAX_MULTIPLIER,
    ))
    updated_variance = float((1.0 - gain) * predicted_variance)
    diagnostics.update({
        "accepted_for_scale_update": True,
        "kalman_gain": gain,
        "ratio_quality": ratio_quality,
        "count_quality": count_quality,
        "combined_quality": quality,
        "measurement_std_log": measurement_std,
        "log_variance_after": updated_variance,
        "multiplier_after": updated,
        "clamped": bool(updated != proposed),
    })
    return updated, updated_variance, diagnostics


def sequential_scale_warmup_observation(
    raw_anchor_scale: float,
    inlier_ratio: float,
    inlier_count: float,
    rmse_m: float,
) -> dict[str, float] | None:
    """Return a trustworthy pre-filter scale observation, else None."""
    raw = float(raw_anchor_scale)
    if not (
        np.isfinite(raw)
        and SEQUENTIAL_ANCHOR_SCALE_MIN_RAW <= raw <= SEQUENTIAL_ANCHOR_SCALE_MAX_RAW
        and float(inlier_ratio) >= SEQUENTIAL_ANCHOR_SCALE_WARMUP_MIN_RATIO
        and float(inlier_count) >= SEQUENTIAL_ANCHOR_SCALE_WARMUP_MIN_INLIERS
        and float(rmse_m) <= SEQUENTIAL_ANCHOR_SCALE_WARMUP_MAX_RMSE_M
        and abs(math.log(raw)) >= math.log(1.0 + SEQUENTIAL_ANCHOR_SCALE_WARMUP_MIN_DEVIATION)
    ):
        return None
    weight = float(max(inlier_ratio, 0.01) * max(inlier_count, 1.0) / max(rmse_m, 0.5))
    return {"raw_scale": raw, "log_scale": float(math.log(raw)), "weight": weight}


def bootstrap_sequential_anchor_scale(
    current_multiplier: float,
    observations: list[dict[str, float]],
) -> tuple[float, float, dict[str, Any]]:
    """Initialize the causal filter from consistent warm-up observations."""
    logs = np.asarray([item["log_scale"] for item in observations], dtype=np.float64)
    weights = np.asarray([item["weight"] for item in observations], dtype=np.float64)
    average_log = float(np.sum(logs * weights) / max(float(weights.sum()), 1e-12))
    multiplier = float(np.clip(
        float(current_multiplier) * math.exp(average_log),
        SEQUENTIAL_ANCHOR_SCALE_MIN_MULTIPLIER,
        SEQUENTIAL_ANCHOR_SCALE_MAX_MULTIPLIER,
    ))
    # Keep a finite posterior variance: later anchors may refine this initial
    # estimate but cannot immediately overwrite it.
    variance = max(float(np.var(logs)), SEQUENTIAL_ANCHOR_SCALE_MIN_MEASUREMENT_STD_LOG ** 2)
    return multiplier, variance, {
        "enabled": True,
        "activated": True,
        "reason": "consistent_high_quality_warmup",
        "warmup_count": int(len(observations)),
        "warmup_raw_scales": [float(item["raw_scale"]) for item in observations],
        "multiplier_after": multiplier,
        "log_variance_after": variance,
    }


def estimate_sequence_metric_scale(window_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Robustly summarize the *one* scale correction supported by a sequence.

    This is a calibration diagnostic, not another per-window correction.  The
    returned multiplier is meant for a second, normal inference pass with
    `ANCHOR_APPLY_SATELLITE_SCALE=False`, so the only metric-scale change is
    made at `sim2_from_first_gt`.
    """
    samples: list[dict[str, float]] = []
    for summary in window_summaries:
        if not bool(summary.get("anchor_accepted", False)):
            continue
        # Window summaries store this record under ``anchor``.  Keep the legacy
        # key as a fallback for older cached summaries.
        stats = summary.get("anchor", summary.get("anchor_stats", {})) or {}
        try:
            scale = float(stats.get("satellite_raw_scale"))
            inlier_ratio = float(stats.get("inlier_ratio"))
            inlier_count = float(stats.get("inlier_count"))
            rmse = float(stats.get("rmse"))
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(scale) and scale > 0.0 and np.isfinite(rmse) and rmse > 0.0):
            continue
        # Accepted anchors already pass the geometric gates.  Weight remaining
        # evidence by support and inverse reprojection/map residual.
        weight = max(inlier_ratio, 0.01) * max(inlier_count, 1.0) / max(rmse, 0.5)
        samples.append({
            "window_idx": float(summary.get("window_idx", -1)),
            "raw_scale": scale,
            "log_scale": float(math.log(scale)),
            "weight": float(weight),
        })
    if not samples:
        return {
            "available": False,
            "reason": "no_accepted_anchor_with_finite_scale",
            "recommended_multiplier": 1.0,
        }
    logs = np.asarray([s["log_scale"] for s in samples], dtype=np.float64)
    median_log = float(np.median(logs))
    mad = float(np.median(np.abs(logs - median_log)))
    # Never make the filter narrower than a 6% scale discrepancy.  This keeps
    # normal satellite-match noise from turning one sequence into one sample.
    keep_log_deviation = max(math.log(1.06), 2.5 * 1.4826 * mad)
    kept = [s for s in samples if abs(s["log_scale"] - median_log) <= keep_log_deviation]
    weights = np.asarray([s["weight"] for s in kept], dtype=np.float64)
    kept_logs = np.asarray([s["log_scale"] for s in kept], dtype=np.float64)
    robust_log = float(np.sum(weights * kept_logs) / max(float(weights.sum()), 1e-12))
    return {
        "available": True,
        "method": "accepted_anchor_weighted_log_mean_after_mad_trim",
        "sample_count": int(len(samples)),
        "kept_sample_count": int(len(kept)),
        "raw_scale_median": float(math.exp(median_log)),
        "raw_scale_mad_log": mad,
        "trim_log_deviation": keep_log_deviation,
        "recommended_multiplier": float(math.exp(robust_log)),
        "samples": samples,
        "kept_window_indices": [int(s["window_idx"]) for s in kept],
    }


def camera_target_from_ground_anchor_with_pitch(
    residual_for_pose: np.ndarray,
    ground_prior: dict[str, Any],
    pitch_deg: float,
    height_m: float | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Back-project a matched ground point using a replacement pitch value.

    The satellite residual constrains a ground point.  This helper is only for
    controlled ablations: it preserves the VGGT height and yaw, replacing the
    pitch used to recover the camera center from that matched ground point.
    """
    try:
        valid = bool(ground_prior.get("valid", False))
        ground_before = np.asarray(ground_prior["center_utm"], dtype=np.float64).reshape(2)
        vggt_height_m = float(ground_prior["height_m"])
        yaw_deg = float(ground_prior["pred_yaw_deg"])
        vggt_pitch_deg = float(ground_prior["pred_pitch_deg"])
        replacement_pitch_deg = float(pitch_deg)
    except (KeyError, TypeError, ValueError):
        return None, {"valid": False, "reason": "missing_ground_backprojection_inputs"}
    down_deg = abs(replacement_pitch_deg)
    if not (
        valid
        and np.isfinite(ground_before).all()
        and np.isfinite(vggt_height_m)
        and np.isfinite(yaw_deg)
        and np.isfinite(vggt_pitch_deg)
        and np.isfinite(replacement_pitch_deg)
        and vggt_height_m > 0.0
        and 0.1 < down_deg < 89.9
    ):
        return None, {"valid": False, "reason": "invalid_ground_backprojection_inputs"}
    replacement_height_m = float(height_m) if height_m is not None else vggt_height_m
    if not np.isfinite(replacement_height_m) or replacement_height_m <= 0.0:
        return None, {"valid": False, "reason": "invalid_replacement_height"}
    ground_after = apply_sim2(residual_for_pose, ground_before.reshape(1, 2))[0]
    ground_range_m = replacement_height_m / max(math.tan(math.radians(down_deg)), 1e-12)
    ground_range_m = min(float(ground_range_m), float(SATELLITE_MAX_PITCH_GROUND_RANGE_M))
    yaw_rad = math.radians(yaw_deg)
    offset = np.array([
        math.sin(yaw_rad) * ground_range_m,
        math.cos(yaw_rad) * ground_range_m,
    ], dtype=np.float64)
    target_camera = ground_after - offset
    vggt_down_deg = abs(vggt_pitch_deg)
    vggt_range_m = (
        vggt_height_m / max(math.tan(math.radians(vggt_down_deg)), 1e-12)
        if 0.1 < vggt_down_deg < 89.9
        else float("nan")
    )
    return target_camera, {
        "valid": True,
        "ground_prior_utm": ground_before.tolist(),
        "ground_matched_utm": ground_after.tolist(),
        "vggt_pitch_deg": vggt_pitch_deg,
        "replacement_pitch_deg": replacement_pitch_deg,
        "vggt_height_m": vggt_height_m,
        "replacement_height_m": replacement_height_m,
        "vggt_ground_range_m": float(vggt_range_m),
        "replacement_pitch_ground_range_m": float(ground_range_m),
        "ground_range_delta_m": float(ground_range_m - vggt_range_m),
    }


def fit_similarity(src_xy: np.ndarray, dst_xy: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray | None:
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if len(src) < 2 or src.shape != dst.shape:
        return None
    if weights is None:
        w = np.ones(len(src), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(src):
            w = np.ones(len(src), dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(w.sum()) <= 1e-12:
        w[:] = 1.0
    w /= float(w.sum())
    src_mean = np.sum(src * w[:, None], axis=0)
    dst_mean = np.sum(dst * w[:, None], axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    variance = float(np.sum(w * np.sum(src_c ** 2, axis=1)))
    if variance <= 1e-12:
        return None
    covariance = (dst_c * w[:, None]).T @ src_c
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    sign = np.eye(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        sign[-1, -1] = -1
    rotation = u @ sign @ vt
    scale = float(np.sum(singular * np.diag(sign)) / variance)
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


def similarity_ransac(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    confidences: np.ndarray | None,
    threshold: float,
    min_inliers: int,
    iterations: int = 1800,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    src, dst = src[finite], dst[finite]
    if confidences is None:
        conf = np.ones(len(src), dtype=np.float64)
    else:
        conf = np.asarray(confidences, dtype=np.float64).reshape(-1)[:len(finite)][finite]
        conf = np.where(np.isfinite(conf) & (conf > 0), conf, 1.0)
    if len(src) < max(3, int(min_inliers)):
        return None, np.zeros(len(src), dtype=bool), {
            "reason": "too_few_correspondences",
            "candidate_count": int(len(src)),
        }
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED + len(src))
    order = np.argsort(-conf)
    pool = order[:min(len(order), 1000)]
    best_mask = None
    best_matrix = None
    best_score = (-1, float("inf"))
    for _ in range(max(1, int(iterations))):
        ids = rng.choice(pool, 3, replace=False)
        matrix = fit_similarity(src[ids], dst[ids], conf[ids])
        if matrix is None:
            continue
        residual = np.linalg.norm(apply_sim2(matrix, src) - dst, axis=1)
        mask = residual <= float(threshold)
        count = int(mask.sum())
        if count < 2:
            continue
        rmse = float(np.sqrt(np.mean(residual[mask] ** 2)))
        if count > best_score[0] or (count == best_score[0] and rmse < best_score[1]):
            best_score = (count, rmse)
            best_mask = mask
            best_matrix = matrix
    if best_mask is None or int(best_mask.sum()) < int(min_inliers):
        return None, np.zeros(len(src), dtype=bool), {
            "reason": "ransac_failed_or_low_inliers",
            "candidate_count": int(len(src)),
            "inlier_count": int(0 if best_mask is None else best_mask.sum()),
        }
    refined = fit_similarity(src[best_mask], dst[best_mask], conf[best_mask])
    if refined is not None:
        best_matrix = refined
    residual = np.linalg.norm(apply_sim2(best_matrix, src) - dst, axis=1)
    best_mask = residual <= float(threshold)
    count = int(best_mask.sum())
    scale, yaw = sim2_scale_yaw(best_matrix)
    return best_matrix, best_mask, {
        "candidate_count": int(len(src)),
        "inlier_count": count,
        "inlier_ratio": float(count / max(len(src), 1)),
        "rmse": float(np.sqrt(np.mean(residual[best_mask] ** 2))) if count else float("nan"),
        "median_residual": float(np.median(residual[best_mask])) if count else float("nan"),
        "scale": scale,
        "yaw_deg": yaw,
        "threshold": float(threshold),
    }


def fit_similarity_anchored(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    anchor_src: np.ndarray,
    anchor_dst: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray | None:
    """LY constrained Sim(2): the propagated camera anchor is mapped exactly."""
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    anchor_src = np.asarray(anchor_src, dtype=np.float64).reshape(2)
    anchor_dst = np.asarray(anchor_dst, dtype=np.float64).reshape(2)
    if len(src) < 1 or src.shape != dst.shape:
        return None
    if weights is None:
        weight = np.ones(len(src), dtype=np.float64)
    else:
        weight = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weight) != len(src):
            weight = np.ones(len(src), dtype=np.float64)
    weight = np.where(np.isfinite(weight) & (weight > 0.0), weight, 0.0)
    if float(weight.sum()) <= 1e-12:
        weight[:] = 1.0
    weight /= float(weight.sum())
    src_centered = src - anchor_src
    dst_centered = dst - anchor_dst
    variance = float(np.sum(weight * np.sum(src_centered ** 2, axis=1)))
    if variance <= 1e-12 or not np.isfinite(variance):
        return None
    covariance = (dst_centered * weight[:, None]).T @ src_centered
    try:
        u, singular, vt = np.linalg.svd(covariance)
    except np.linalg.LinAlgError:
        return None
    sign = np.eye(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vt
    scale = float(np.sum(singular * np.diag(sign)) / variance)
    if not np.isfinite(scale) or scale <= 1e-12:
        return None
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = anchor_dst - matrix[:, :2] @ anchor_src
    return matrix


def _convex_hull_area(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return 0.0
    return float(abs(cv2.contourArea(cv2.convexHull(points.astype(np.float32)))))


def _bbox_area(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 2:
        return 0.0
    span = np.ptp(points, axis=0)
    return float(max(span[0] * span[1], 0.0))


def _spread_ratio(points_xy: np.ndarray) -> float:
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    if len(points) < 3:
        return 0.0
    singular = np.linalg.svd(points - np.mean(points, axis=0), compute_uv=False)
    if len(singular) < 2 or float(singular[0]) <= 1e-12:
        return 0.0
    return float(singular[-1] / singular[0])


def _lateral_span_ratio(
    sample_xy: np.ndarray,
    all_xy: np.ndarray,
    camera_center_xy: np.ndarray,
) -> float:
    sample = np.asarray(sample_xy, dtype=np.float64).reshape(-1, 2)
    points = np.asarray(all_xy, dtype=np.float64).reshape(-1, 2)
    center = np.asarray(camera_center_xy, dtype=np.float64).reshape(2)
    relative = sample - center
    direction = np.mean(relative, axis=0)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-9:
        return 0.0
    lateral = np.array([-direction[1], direction[0]], dtype=np.float64) / norm
    return float(np.ptp(sample @ lateral) / max(np.linalg.norm(np.ptp(points, axis=0)), 1e-9))


def similarity_ransac_anchored_like_ly(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    confidences: np.ndarray,
    uav_grid_ids: np.ndarray,
    sat_grid_ids: np.ndarray,
    camera_center_xy: np.ndarray,
    anchor_src_xy: np.ndarray,
    anchor_dst_xy: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    confidence = np.asarray(confidences, dtype=np.float64).reshape(-1)
    uav_grid = np.asarray(uav_grid_ids, dtype=np.int32).reshape(-1)
    sat_grid = np.asarray(sat_grid_ids, dtype=np.int32).reshape(-1)
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & np.isfinite(confidence)
    src, dst, confidence = src[finite], dst[finite], confidence[finite]
    uav_grid, sat_grid = uav_grid[finite], sat_grid[finite]
    count = len(src)
    failed_mask = np.zeros(count, dtype=bool)
    if count < int(MULTIFRAME_MIN_CORRESPONDENCES):
        return None, failed_mask, {"reason": "too_few_correspondences", "candidate_count": count}
    if (
        np.unique(uav_grid).size < int(MULTIFRAME_RANSAC_MIN_UAV_GRIDS)
        or np.unique(sat_grid).size < int(MULTIFRAME_RANSAC_MIN_SAT_GRIDS)
    ):
        return None, failed_mask, {"reason": "insufficient_grid_diversity", "candidate_count": count}
    order = np.argsort(-confidence)
    sample_pool = order[:min(count, 500)]
    score_ids = order[:min(count, 2000)]
    sample_size = min(
        max(
            3,
            int(MULTIFRAME_RANSAC_SAMPLE_SIZE),
            int(MULTIFRAME_RANSAC_MIN_UAV_GRIDS),
            int(MULTIFRAME_RANSAC_MIN_SAT_GRIDS),
        ),
        len(sample_pool),
    )
    rng = np.random.default_rng(GLOBAL_RANDOM_SEED + count)
    src_area = max(_bbox_area(src), 1e-12)
    dst_area = max(_bbox_area(dst), 1e-12)
    best_matrix: np.ndarray | None = None
    best_score = (-1, float("inf"), -1.0)
    valid_sets = 0
    for _ in range(int(MULTIFRAME_RANSAC_ITERATIONS)):
        chosen: np.ndarray | None = None
        for _retry in range(int(MULTIFRAME_RANSAC_MAX_SAMPLE_RETRIES)):
            candidate_ids = rng.choice(sample_pool, size=sample_size, replace=False)
            sample_src, sample_dst = src[candidate_ids], dst[candidate_ids]
            valid_sample = (
                np.unique(uav_grid[candidate_ids]).size >= int(MULTIFRAME_RANSAC_MIN_UAV_GRIDS)
                and np.unique(sat_grid[candidate_ids]).size >= int(MULTIFRAME_RANSAC_MIN_SAT_GRIDS)
                and _convex_hull_area(sample_src) / src_area >= float(MULTIFRAME_RANSAC_MIN_HULL_AREA_FRAC)
                and _convex_hull_area(sample_dst) / dst_area >= float(MULTIFRAME_RANSAC_MIN_HULL_AREA_FRAC)
                and _spread_ratio(sample_src) >= float(MULTIFRAME_RANSAC_COLLINEAR_EPS)
                and _spread_ratio(sample_dst) >= float(MULTIFRAME_RANSAC_COLLINEAR_EPS)
                and _lateral_span_ratio(sample_src, src, camera_center_xy)
                >= float(MULTIFRAME_RANSAC_MIN_LATERAL_SPAN_FRAC)
            )
            if valid_sample:
                chosen = candidate_ids
                valid_sets += 1
                break
        if chosen is None:
            continue
        matrix = fit_similarity_anchored(
            src[chosen], dst[chosen], anchor_src_xy, anchor_dst_xy, confidence[chosen]
        )
        if matrix is None:
            continue
        residual = np.linalg.norm(apply_sim2(matrix, src[score_ids]) - dst[score_ids], axis=1)
        inliers = residual <= float(MULTIFRAME_RANSAC_THRESHOLD_M)
        inlier_count = int(np.count_nonzero(inliers))
        if inlier_count <= 0:
            continue
        rmse = float(np.sqrt(np.mean(residual[inliers] ** 2)))
        conf_sum = float(np.sum(confidence[score_ids][inliers]))
        score = (inlier_count, -rmse, conf_sum)
        if score > (best_score[0], -best_score[1], best_score[2]):
            best_matrix = matrix
            best_score = (inlier_count, rmse, conf_sum)
    if best_matrix is None:
        return None, failed_mask, {
            "reason": "no_valid_minimal_sets" if valid_sets == 0 else "ransac_no_model",
            "candidate_count": count,
        }
    residual = np.linalg.norm(apply_sim2(best_matrix, src) - dst, axis=1)
    inliers = residual <= float(MULTIFRAME_RANSAC_THRESHOLD_M)
    if np.any(inliers):
        refined = fit_similarity_anchored(
            src[inliers], dst[inliers], anchor_src_xy, anchor_dst_xy, confidence[inliers]
        )
        if refined is not None:
            best_matrix = refined
    for _ in range(int(MULTIFRAME_ROBUST_ITERS)):
        residual = np.linalg.norm(apply_sim2(best_matrix, src) - dst, axis=1)
        support = residual <= 3.0 * float(MULTIFRAME_RANSAC_THRESHOLD_M)
        huber = np.ones_like(residual)
        large = residual > float(MULTIFRAME_HUBER_DELTA_M)
        huber[large] = float(MULTIFRAME_HUBER_DELTA_M) / np.maximum(residual[large], 1e-12)
        weights = np.maximum(confidence, 1e-6) * huber
        weights[~support] = 0.0
        refined = fit_similarity_anchored(src, dst, anchor_src_xy, anchor_dst_xy, weights)
        if refined is None:
            break
        best_matrix = refined
    residual = np.linalg.norm(apply_sim2(best_matrix, src) - dst, axis=1)
    inliers = residual <= float(MULTIFRAME_RANSAC_THRESHOLD_M)
    inlier_count = int(np.count_nonzero(inliers))
    scale, yaw = sim2_scale_yaw(best_matrix)
    return best_matrix, inliers, {
        "candidate_count": count,
        "inlier_count": inlier_count,
        "inlier_ratio": float(inlier_count / max(count, 1)),
        "rmse": float(np.sqrt(np.mean(residual[inliers] ** 2))) if inlier_count else float("nan"),
        "median_residual": float(np.median(residual[inliers])) if inlier_count else float("nan"),
        "scale": float(scale),
        "yaw_deg": float(yaw),
        "threshold": float(MULTIFRAME_RANSAC_THRESHOLD_M),
        "valid_minimal_sets": int(valid_sets),
        "uav_unique_grids": int(np.unique(uav_grid[inliers]).size) if inlier_count else 0,
        "sat_unique_grids": int(np.unique(sat_grid[inliers]).size) if inlier_count else 0,
        "anchored": True,
    }


@dataclass
class GroundGeometry:
    origin: np.ndarray
    basis_x: np.ndarray
    basis_y: np.ndarray
    up: np.ndarray
    camera_xy: np.ndarray
    camera_height: np.ndarray
    camera_forward_xy: np.ndarray
    point_xy: np.ndarray
    point_height: np.ndarray


@dataclass
class WindowState:
    window_idx: int
    frame_indices: list[int]
    frame_names: list[str]
    geometry: GroundGeometry
    local_to_utm: np.ndarray
    camera_utm: np.ndarray
    camera_yaw: np.ndarray
    height_scale_m_per_local: float
    anchor_accepted: bool
    anchor_stats: dict[str, Any] = field(default_factory=dict)


def predict_window_center_from_history(
    states: list[WindowState],
    target_frame_idx: float,
    history: int,
    min_history: int,
) -> dict[str, Any] | None:
    usable: list[tuple[float, np.ndarray, float, int]] = []
    for state in states[-max(int(history), 1):]:
        if not state.frame_indices or len(state.camera_utm) == 0:
            continue
        center_local_idx = len(state.frame_indices) // 2
        center_local_idx = min(center_local_idx, len(state.camera_utm) - 1)
        center_frame_idx = float(state.frame_indices[center_local_idx])
        center_xy = np.asarray(state.camera_utm[center_local_idx], dtype=np.float64)
        center_yaw = float(state.camera_yaw[center_local_idx])
        if np.isfinite(center_frame_idx) and np.isfinite(center_xy).all() and np.isfinite(center_yaw):
            usable.append((center_frame_idx, center_xy, center_yaw, int(state.window_idx)))
    if len(usable) < max(int(min_history), 1):
        return None
    t = np.asarray([item[0] for item in usable], dtype=np.float64)
    xy = np.asarray([item[1] for item in usable], dtype=np.float64)
    yaw_unwrapped = np.degrees(np.unwrap(np.radians([item[2] for item in usable])))
    if len(usable) >= 2 and float(np.ptp(t)) > 1e-6:
        pred_e = float(np.polyval(np.polyfit(t, xy[:, 0], 1), float(target_frame_idx)))
        pred_n = float(np.polyval(np.polyfit(t, xy[:, 1], 1), float(target_frame_idx)))
        pred_yaw = wrap_deg(float(np.polyval(np.polyfit(t, yaw_unwrapped, 1), float(target_frame_idx))))
        mode = "linear_fit"
    else:
        pred_e = float(xy[-1, 0])
        pred_n = float(xy[-1, 1])
        pred_yaw = wrap_deg(float(yaw_unwrapped[-1]))
        mode = "hold_last"
    return {
        "mode": mode,
        "history_count": int(len(usable)),
        "history_window_indices": [int(item[3]) for item in usable],
        "target_frame_idx": float(target_frame_idx),
        "predicted_center_utm": [pred_e, pred_n],
        "predicted_yaw_deg": float(pred_yaw),
    }


def camera_centers(extrinsics: np.ndarray) -> np.ndarray:
    r = np.asarray(extrinsics, dtype=np.float64)[:, :3, :3]
    t = np.asarray(extrinsics, dtype=np.float64)[:, :3, 3]
    return -np.einsum("nij,nj->ni", np.transpose(r, (0, 2, 1)), t)


def local_xy_from_xyz_canon(points_xyz: np.ndarray, pitch_deg: float) -> np.ndarray:
    """LY ground-plane canonicalization using allowed metadata pitch."""
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(pitch_deg):
        return np.column_stack([points[:, 0], points[:, 2]])
    angle = math.radians(abs(float(pitch_deg)))
    z_canon = -math.sin(angle) * points[:, 1] + math.cos(angle) * points[:, 2]
    return np.column_stack([points[:, 0], z_canon])


def estimate_ground_plane(points: np.ndarray, cameras: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 100:
        raise ValueError("Too few VGGT points for ground-plane estimation")
    rng = np.random.default_rng(seed)
    if len(pts) > 60_000:
        pts = pts[rng.choice(len(pts), 60_000, replace=False)]
    q05, q95 = np.percentile(pts, [5, 95], axis=0)
    scene_span = float(np.linalg.norm(q95 - q05))
    threshold = max(1e-4, 0.006 * scene_span)
    best_mask = None
    best_count = -1
    best_normal = None
    best_d = None
    for _ in range(420):
        ids = rng.choice(len(pts), 3, replace=False)
        p0, p1, p2 = pts[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = float(np.linalg.norm(normal))
        if norm < 1e-9:
            continue
        normal /= norm
        d = -float(normal @ p0)
        distance = np.abs(pts @ normal + d)
        mask = distance <= threshold
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask
            best_normal = normal
            best_d = d
    if best_mask is None or best_count < 100:
        raise ValueError("Ground-plane RANSAC failed")
    plane_pts = pts[best_mask]
    center = np.mean(plane_pts, axis=0)
    _, _, vt = np.linalg.svd(plane_pts - center, full_matrices=False)
    normal = vt[-1]
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    signed_camera = (np.asarray(cameras, dtype=np.float64) - center) @ normal
    if float(np.median(signed_camera)) < 0:
        normal = -normal
        signed_camera = -signed_camera
    return center, normal, {
        "sample_count": int(len(pts)),
        "inlier_count": int(best_count),
        "inlier_ratio": float(best_count / max(len(pts), 1)),
        "threshold_local": float(threshold),
        "scene_span_local": float(scene_span),
        "camera_height_local_median": float(np.median(signed_camera)),
    }


def build_ground_geometry(
    prediction: Any,
    seed: int,
    canonical_pitch_deg: float | None = None,
) -> tuple[GroundGeometry, dict[str, Any]]:
    points = np.asarray(prediction.points, dtype=np.float64)
    cameras = camera_centers(prediction.extrinsics)
    origin, up, stats = estimate_ground_plane(points, cameras, seed)
    trajectory = cameras[-1] - cameras[0] if len(cameras) > 1 else np.zeros(3)
    basis_x = trajectory - up * float(trajectory @ up)
    if float(np.linalg.norm(basis_x)) < 1e-7:
        first_c2w = np.asarray(prediction.extrinsics[0, :3, :3], dtype=np.float64).T
        basis_x = first_c2w[:, 0] - up * float(first_c2w[:, 0] @ up)
    basis_x /= max(float(np.linalg.norm(basis_x)), 1e-12)
    basis_y = np.cross(up, basis_x)
    basis_y /= max(float(np.linalg.norm(basis_y)), 1e-12)
    camera_delta = cameras - origin
    point_delta = points - origin
    camera_xy = np.column_stack([camera_delta @ basis_x, camera_delta @ basis_y])
    point_xy = np.column_stack([point_delta @ basis_x, point_delta @ basis_y])
    camera_h = camera_delta @ up
    point_h = point_delta @ up
    forward_xy = []
    for ext in np.asarray(prediction.extrinsics, dtype=np.float64):
        forward = ext[:3, :3].T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        forward_xy.append([float(forward @ basis_x), float(forward @ basis_y)])
    if canonical_pitch_deg is not None and np.isfinite(float(canonical_pitch_deg)):
        camera_xy = local_xy_from_xyz_canon(cameras, float(canonical_pitch_deg))
        point_xy = local_xy_from_xyz_canon(points, float(canonical_pitch_deg))
        forward_world = np.asarray([
            ext[:3, :3].T @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
            for ext in np.asarray(prediction.extrinsics, dtype=np.float64)
        ])
        forward_xy = local_xy_from_xyz_canon(forward_world, float(canonical_pitch_deg))
        stats["horizontal_coordinate_mode"] = "ly_pitch_canonical_xz"
        stats["canonical_pitch_deg"] = float(canonical_pitch_deg)
    else:
        stats["horizontal_coordinate_mode"] = "estimated_ground_plane"
    return GroundGeometry(
        origin=origin,
        basis_x=basis_x,
        basis_y=basis_y,
        up=up,
        camera_xy=camera_xy,
        camera_height=camera_h,
        camera_forward_xy=np.asarray(forward_xy, dtype=np.float64),
        point_xy=point_xy,
        point_height=point_h,
    ), stats


def yaw_from_forward(matrix: np.ndarray, forward_xy: np.ndarray) -> np.ndarray:
    linear = np.asarray(matrix, dtype=np.float64)[:, :2]
    out = []
    for vector in np.asarray(forward_xy, dtype=np.float64):
        global_vector = linear @ vector
        if float(np.linalg.norm(global_vector)) < 1e-10:
            out.append(float("nan"))
        else:
            out.append(wrap_deg(math.degrees(math.atan2(global_vector[0], global_vector[1]))))
    return np.asarray(out, dtype=np.float64)


def interpolate_sim2_per_frame(
    start_matrix: np.ndarray,
    end_matrix: np.ndarray,
    frame_count: int,
    hold_frame_count: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep overlap frames fixed, then ramp a valid Sim(2) correction to the window end."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be positive, got {frame_count}")
    hold = int(np.clip(int(hold_frame_count), 1, int(frame_count)))
    alpha = np.zeros(int(frame_count), dtype=np.float64)
    ramp_count = int(frame_count) - hold
    if ramp_count > 0:
        # Exactly ``hold`` frames stay fixed.  The first new frame must already
        # receive a non-zero correction; otherwise every window silently adds
        # one extra uncorrected frame at its boundary.
        alpha[hold:] = np.linspace(
            1.0 / float(ramp_count), 1.0, ramp_count, dtype=np.float64
        )
    matrices = (
        (1.0 - alpha)[:, None, None] * np.asarray(start_matrix, dtype=np.float64)[None, :, :]
        + alpha[:, None, None] * np.asarray(end_matrix, dtype=np.float64)[None, :, :]
    )
    return matrices, alpha


def apply_framewise_sim2(matrices: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    points = np.asarray(points_xy, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (2, 3):
        raise ValueError(f"Expected [N,2,3] Sim(2) matrices, got {matrices.shape}")
    if len(matrices) != len(points):
        raise ValueError("Framewise Sim(2) matrix and point counts must match")
    return np.einsum("nij,nj->ni", matrices[:, :, :2], points) + matrices[:, :, 2]


def yaw_from_forward_framewise(matrices: np.ndarray, forward_xy: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices, dtype=np.float64)
    forward = np.asarray(forward_xy, dtype=np.float64)
    if len(matrices) != len(forward):
        raise ValueError("Framewise Sim(2) matrix and forward-vector counts must match")
    global_forward = np.einsum("nij,nj->ni", matrices[:, :, :2], forward)
    norms = np.linalg.norm(global_forward, axis=1)
    yaw = np.full(len(global_forward), np.nan, dtype=np.float64)
    valid = norms >= 1e-10
    yaw[valid] = np.degrees(np.arctan2(global_forward[valid, 0], global_forward[valid, 1]))
    return np.asarray([wrap_deg(value) if np.isfinite(value) else value for value in yaw], dtype=np.float64)


def camera_to_body_rotation_opencv() -> np.ndarray:
    # Body axes: X=forward (image top), Y=right, Z=down.
    return np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def body_to_gt_ypr(R_body_to_enu: np.ndarray) -> dict[str, float]:
    enu_to_ned = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    R_body_to_ned = enu_to_ned @ np.asarray(R_body_to_enu, dtype=np.float64)
    yaw_deg, pitch_deg, roll_deg = SciRotation.from_matrix(R_body_to_ned).as_euler("ZYX", degrees=True)
    raw_pitch = wrap_deg(-float(pitch_deg))
    dji_like_pitch = wrap_deg(-(90.0 - abs(raw_pitch)))
    return {
        "yaw_deg": wrap_deg(float(yaw_deg)),
        "pitch_deg": float(dji_like_pitch),
        "roll_deg": wrap_deg(float(roll_deg)),
    }


def vggt_window_ypr(
    extrinsics: np.ndarray,
    geometry: GroundGeometry,
    local_to_utm: np.ndarray,
) -> list[dict[str, float]]:
    extrinsics_array = np.asarray(extrinsics, dtype=np.float64)
    matrices = np.asarray(local_to_utm, dtype=np.float64)
    if matrices.ndim == 2:
        matrices = np.repeat(matrices[None, :, :], len(extrinsics_array), axis=0)
    if matrices.shape != (len(extrinsics_array), 2, 3):
        raise ValueError(
            "Expected one [2,3] Sim(2) matrix or one per frame, "
            f"got {matrices.shape} for {len(extrinsics_array)} frames"
        )
    ground_basis_rows = np.vstack([geometry.basis_x, geometry.basis_y])
    camera_to_body = camera_to_body_rotation_opencv()
    output: list[dict[str, float]] = []
    for extrinsic, matrix in zip(extrinsics_array, matrices):
        scale, _ = sim2_scale_yaw(matrix)
        if not np.isfinite(scale) or scale <= 1e-12:
            raise ValueError(f"Invalid Sim(2) scale for attitude conversion: {scale}")
        rotation_2d = np.asarray(matrix[:, :2], dtype=np.float64) / float(scale)
        world_to_enu = np.vstack([
            rotation_2d @ ground_basis_rows,
            np.asarray(geometry.up, dtype=np.float64),
        ])
        camera_to_world = extrinsic[:3, :3].T
        camera_to_enu = world_to_enu @ camera_to_world
        body_to_enu = camera_to_enu @ camera_to_body
        output.append(body_to_gt_ypr(body_to_enu))
    return output


def circular_weighted_mean(values_deg: list[float], weights: list[float]) -> float:
    values = np.radians(np.asarray(values_deg, dtype=np.float64))
    w = np.asarray(weights, dtype=np.float64)
    s = float(np.sum(w * np.sin(values)))
    c = float(np.sum(w * np.cos(values)))
    return wrap_deg(math.degrees(math.atan2(s, c)))
