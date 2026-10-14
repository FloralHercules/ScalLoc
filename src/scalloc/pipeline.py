"""Pipeline from the validated fast ScalLoc pipeline."""
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
from .geometry import (GroundGeometry, WindowState, accepted_anchor_component_history, apply_framewise_sim2, apply_sim2, blend_sim2_at_local_pivot, bootstrap_sequential_anchor_scale, build_ground_geometry, camera_target_from_ground_anchor_with_pitch, circular_weighted_mean, component_consensus, compose_sim2, estimate_sequence_metric_scale, fit_similarity, interpolate_sim2_per_frame, predict_window_center_from_history, satellite_residual_for_pose, sequential_scale_warmup_observation, sim2_scale_yaw, update_sequential_anchor_scale, vggt_window_ypr, wrap_deg, yaw_from_forward, yaw_from_forward_framewise)
from .matching import (build_middle_frame_pitch_crop_prior, compare_bev_rescue_snapshots, estimate_multiframe_3d_utm_anchor, fuse_bev_and_multiframe_candidates, make_bev_rescue_snapshot, match_bev_anchor, match_bev_anchor_template, merge_utm_point_clouds_without_alignment, rasterize_prior_bev, save_rgb_point_cloud_ply)
from .reporting import (build_stage_timing_rows, collect_window_pitch_comparison, print_metrics_comparison, record_stage_timing, save_anchor_diagnostics_csv, save_batch_reports, save_comparison_csv_reports, save_window_attitude_csv, save_window_vggt_intrinsics_csv)
from .runtime_utils import (load_module_from_path, template_match_optimizer_path)
from .image_io import install_windows_unicode_io
install_windows_unicode_io()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    cv2.setRNGSeed(int(seed))


def metric_scale_multiplier_for_dataset(dataset_name: str) -> float:
    """Return the configured once-per-sequence metric scale correction."""
    return float(SEQUENCE_METRIC_SCALE_MAP.get(str(dataset_name), SEQUENCE_METRIC_SCALE_MULTIPLIER))


def sim2_from_first_gt(
    geometry: GroundGeometry,
    gt_e: float,
    gt_n: float,
    gt_yaw: float,
    gt_height: float,
    metric_scale_multiplier: float = SEQUENCE_METRIC_SCALE_MULTIPLIER,
) -> tuple[np.ndarray, dict[str, float]]:
    local_height = abs(float(geometry.camera_height[0]))
    if local_height < 1e-6:
        local_height = float(np.median(np.abs(geometry.camera_height)))
    if local_height < 1e-6:
        raise ValueError("Cannot recover VGGT scale from first-frame height")
    height_only_scale = float(gt_height) / local_height
    scale = height_only_scale * float(metric_scale_multiplier)
    forward = geometry.camera_forward_xy[0]
    if float(np.linalg.norm(forward)) < 1e-8 and len(geometry.camera_xy) > 1:
        forward = geometry.camera_xy[1] - geometry.camera_xy[0]
    local_angle = math.atan2(float(forward[1]), float(forward[0]))
    global_forward = np.array([math.sin(math.radians(gt_yaw)), math.cos(math.radians(gt_yaw))])
    global_angle = math.atan2(float(global_forward[1]), float(global_forward[0]))
    theta = global_angle - local_angle
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = np.array([gt_e, gt_n]) - matrix[:, :2] @ geometry.camera_xy[0]
    return matrix, {
        "scale_m_per_local": float(scale),
        "height_only_scale_m_per_local": float(height_only_scale),
        "sequence_metric_scale_multiplier": float(metric_scale_multiplier),
        "theta_deg": wrap_deg(math.degrees(theta)),
        "first_camera_height_local": float(local_height),
    }


def prior_from_shared_last_pose(
    geometry: GroundGeometry,
    frame_indices: list[int],
    previous: WindowState,
    scale_m_per_local: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not frame_indices or not previous.frame_indices:
        return None, {"reason": "empty_window_at_direct_stitch"}
    current_shared_idx = int(frame_indices[0])
    previous_shared_idx = int(previous.frame_indices[-1])
    if current_shared_idx != previous_shared_idx:
        return None, {
            "reason": "direct_stitch_boundary_frame_mismatch",
            "current_first_frame": current_shared_idx,
            "previous_last_frame": previous_shared_idx,
        }
    target_position = np.asarray(previous.camera_utm[-1], dtype=np.float64)
    target_yaw_deg = float(previous.camera_yaw[-1])
    local_forward = np.asarray(geometry.camera_forward_xy[0], dtype=np.float64)
    if float(np.linalg.norm(local_forward)) < 1e-8:
        return None, {
            "reason": "direct_stitch_first_frame_forward_is_degenerate",
            "shared_frame": current_shared_idx,
        }
    local_angle = math.atan2(float(local_forward[1]), float(local_forward[0]))
    target_forward = np.array([
        math.sin(math.radians(target_yaw_deg)),
        math.cos(math.radians(target_yaw_deg)),
    ], dtype=np.float64)
    target_angle = math.atan2(float(target_forward[1]), float(target_forward[0]))
    theta = target_angle - local_angle
    rotation = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)],
    ], dtype=np.float64)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = float(scale_m_per_local) * rotation
    matrix[:, 2] = target_position - matrix[:, :2] @ geometry.camera_xy[0]
    return matrix, {
        "source": "previous_last_pose_direct_stitch",
        "accepted": True,
        "shared_frame": current_shared_idx,
        "scale_m_per_local": float(scale_m_per_local),
        "theta_deg": wrap_deg(math.degrees(theta)),
        "target_position_utm": target_position.tolist(),
        "target_yaw_deg": target_yaw_deg,
        "position_residual_m": 0.0,
        "yaw_residual_deg": 0.0,
    }


def prior_from_overlap(
    geometry: GroundGeometry,
    frame_indices: list[int],
    previous: WindowState,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    previous_index = {int(idx): i for i, idx in enumerate(previous.frame_indices)}
    src, dst, shared = [], [], []
    for local_idx, global_idx in enumerate(frame_indices):
        if int(global_idx) not in previous_index:
            continue
        src.append(geometry.camera_xy[local_idx])
        dst.append(previous.camera_utm[previous_index[int(global_idx)]])
        shared.append(int(global_idx))
    if len(src) < 2:
        return None, {"reason": "too_few_overlap_frames", "overlap_frames": shared}
    matrix = fit_similarity(np.asarray(src), np.asarray(dst))
    if matrix is None:
        return None, {"reason": "overlap_similarity_failed", "overlap_frames": shared}
    residual = np.linalg.norm(apply_sim2(matrix, np.asarray(src)) - np.asarray(dst), axis=1)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    scale, yaw = sim2_scale_yaw(matrix)
    return matrix, {
        "overlap_frames": shared,
        "overlap_count": int(len(shared)),
        "rmse_m": rmse,
        "scale_m_per_local": scale,
        "theta_deg": yaw,
        "accepted": bool(np.isfinite(rmse) and rmse <= OVERLAP_MAX_RMSE_M),
    }


def prior_from_ly_accepted_anchor_map(
    geometry: GroundGeometry,
    frame_indices: list[int],
    accepted_anchor_by_frame: dict[int, np.ndarray],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """LY first-write-wins overlap fit using only accepted-window anchors."""
    src: list[np.ndarray] = []
    dst: list[np.ndarray] = []
    shared: list[int] = []
    for local_idx, global_idx in enumerate(frame_indices):
        anchor = accepted_anchor_by_frame.get(int(global_idx))
        if anchor is None:
            continue
        local_xy = np.asarray(geometry.camera_xy[local_idx], dtype=np.float64)
        anchor_xy = np.asarray(anchor, dtype=np.float64).reshape(2)
        if not (np.isfinite(local_xy).all() and np.isfinite(anchor_xy).all()):
            continue
        src.append(local_xy)
        dst.append(anchor_xy)
        shared.append(int(global_idx))
    if len(src) < 2:
        return None, {
            "reason": "too_few_accepted_overlap_anchors",
            "overlap_frames": shared,
            "overlap_count": int(len(shared)),
            "accepted": False,
        }
    matrix = fit_similarity(np.asarray(src), np.asarray(dst))
    if matrix is None:
        return None, {
            "reason": "accepted_overlap_similarity_failed",
            "overlap_frames": shared,
            "overlap_count": int(len(shared)),
            "accepted": False,
        }
    residual = np.linalg.norm(apply_sim2(matrix, np.asarray(src)) - np.asarray(dst), axis=1)
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    scale, yaw = sim2_scale_yaw(matrix)
    return matrix, {
        "source": "ly_first_write_accepted_anchor_map",
        "overlap_frames": shared,
        "overlap_count": int(len(shared)),
        "rmse_m": rmse,
        "scale_m_per_local": float(scale),
        "theta_deg": float(yaw),
        "accepted": bool(np.isfinite(rmse)),
    }


def optimize_global_positions(
    states: list[WindowState],
    frame_count: int,
    first_xy: np.ndarray,
    odometry_states: list[WindowState] | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if not USE_FINAL_GLOBAL_POSE_GRAPH or frame_count < 2:
        return None, {"enabled": False}
    position_variable_count = frame_count - 1
    optimize_scale = bool(POSE_GRAPH_OPTIMIZE_GLOBAL_SCALE)
    scale_variable_idx = 2 * position_variable_count if optimize_scale else -1
    variable_count = 2 * position_variable_count + (1 if optimize_scale else 0)
    rows: list[np.ndarray] = []
    rhs: list[float] = []

    def x_idx(frame_idx: int) -> int:
        return int(frame_idx) - 1

    def y_idx(frame_idx: int) -> int:
        return position_variable_count + int(frame_idx) - 1

    def add_equation(coeff: np.ndarray, target: float, weight: float) -> None:
        rows.append(coeff * float(weight))
        rhs.append(float(target) * float(weight))

    odom_weight = 1.0 / max(float(POSE_GRAPH_ODOM_SIGMA_M), 1e-6)
    anchor_count = 0
    odom_count = 0
    use_anchor_free_odometry = bool(
        POSE_GRAPH_USE_ANCHOR_FREE_ODOMETRY and odometry_states is not None
    )
    odometry_by_window = {
        int(state.window_idx): state
        for state in (odometry_states if use_anchor_free_odometry else states)
    }
    for state in states:
        odometry_state = odometry_by_window.get(int(state.window_idx), state)
        if odometry_state.frame_indices != state.frame_indices:
            raise RuntimeError(
                f"Window {state.window_idx} corrected/odometry frame indices do not match"
            )
        for local_idx in range(1, len(odometry_state.frame_indices)):
            i0 = int(odometry_state.frame_indices[local_idx - 1])
            i1 = int(odometry_state.frame_indices[local_idx])
            if i1 != i0 + 1:
                continue
            delta = (
                odometry_state.camera_utm[local_idx]
                - odometry_state.camera_utm[local_idx - 1]
            )
            delta = np.asarray(delta, dtype=np.float64)
            for axis, idx_fn in enumerate((x_idx, y_idx)):
                coeff = np.zeros(variable_count, dtype=np.float64)
                if i1 > 0:
                    coeff[idx_fn(i1)] += 1.0
                target = float(delta[axis])
                if i0 > 0:
                    coeff[idx_fn(i0)] -= 1.0
                else:
                    target = float(first_xy[axis]) + target
                if optimize_scale:
                    coeff[scale_variable_idx] -= float(delta[axis])
                add_equation(coeff, target, odom_weight)
            odom_count += 1
        if state.anchor_accepted:
            local_idx = len(state.frame_indices) // 2
            global_idx = int(state.frame_indices[local_idx])
            if global_idx <= 0:
                continue
            rmse = float(state.anchor_stats.get("rmse", POSE_GRAPH_ANCHOR_SIGMA_M))
            sigma = max(float(POSE_GRAPH_ANCHOR_SIGMA_M), rmse)
            anchor_xy = np.asarray(state.camera_utm[local_idx], dtype=np.float64)
            coeff_e = np.zeros(variable_count, dtype=np.float64)
            coeff_e[x_idx(global_idx)] = 1.0
            add_equation(coeff_e, float(anchor_xy[0]), 1.0 / sigma)
            coeff_n = np.zeros(variable_count, dtype=np.float64)
            coeff_n[y_idx(global_idx)] = 1.0
            add_equation(coeff_n, float(anchor_xy[1]), 1.0 / sigma)
            anchor_count += 1
    if not rows:
        return None, {"enabled": True, "optimized": False, "reason": "no_equations"}
    scale_prior_count = 0
    if optimize_scale:
        coeff = np.zeros(variable_count, dtype=np.float64)
        coeff[scale_variable_idx] = 1.0
        add_equation(
            coeff,
            0.0,
            1.0 / max(float(POSE_GRAPH_GLOBAL_SCALE_PRIOR_SIGMA), 1e-9),
        )
        scale_prior_count = 1
    matrix = np.vstack(rows)
    solution, *_ = np.linalg.lstsq(matrix, np.asarray(rhs), rcond=None)
    result = np.zeros((frame_count, 2), dtype=np.float64)
    result[0] = np.asarray(first_xy, dtype=np.float64)
    result[1:, 0] = solution[:position_variable_count]
    result[1:, 1] = solution[position_variable_count:2 * position_variable_count]
    scale_delta = float(solution[scale_variable_idx]) if optimize_scale else 0.0
    global_scale = 1.0 + scale_delta
    scale_warning = bool(
        optimize_scale
        and (
            global_scale < float(POSE_GRAPH_GLOBAL_SCALE_WARN_MIN)
            or global_scale > float(POSE_GRAPH_GLOBAL_SCALE_WARN_MAX)
        )
    )
    return result, {
        "enabled": True,
        "optimized": True,
        "variable_count": int(variable_count),
        "equation_count": int(len(rows)),
        "odometry_constraint_count": int(odom_count),
        "anchor_constraint_count": int(anchor_count),
        "scale_prior_constraint_count": int(scale_prior_count),
        "odometry_source": (
            "anchor_free_vggt" if use_anchor_free_odometry else "anchor_corrected_windows"
        ),
        "anchor_free_odometry_enabled": bool(use_anchor_free_odometry),
        "odom_sigma_m": float(POSE_GRAPH_ODOM_SIGMA_M),
        "anchor_sigma_m": float(POSE_GRAPH_ANCHOR_SIGMA_M),
        "global_scale_optimization_enabled": bool(optimize_scale),
        "global_scale_factor": float(global_scale),
        "global_scale_delta_from_height_prior": float(scale_delta),
        "global_scale_prior_sigma": float(POSE_GRAPH_GLOBAL_SCALE_PRIOR_SIGMA),
        "global_scale_warning": bool(scale_warning),
        "global_scale_warning_range": [
            float(POSE_GRAPH_GLOBAL_SCALE_WARN_MIN),
            float(POSE_GRAPH_GLOBAL_SCALE_WARN_MAX),
        ],
    }


def find_vggt_safetensors() -> Path | None:
    explicit = os.environ.get("VGGT_WEIGHTS_PATH", "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"VGGT_WEIGHTS_PATH does not exist: {path}")
        return path
    local = ROOT / "weights" / "VGGT" / "model.safetensors"
    if local.is_file():
        return local
    pattern = Path.home() / ".cache" / "huggingface" / "hub" / "models--facebook--VGGT-1B" / "snapshots"
    candidates = sorted(pattern.glob("*/model.safetensors"))
    return candidates[-1] if candidates else None


def load_vggt_once(device: torch.device, enable_track: bool = False):
    sys.path.insert(0, str(THIRD_PARTY_ROOT / "vggt"))
    from vggt.models.vggt import VGGT

    weight_path = find_vggt_safetensors()
    if weight_path is None:
        raise FileNotFoundError(
            "VGGT model.safetensors was not found. Put it in weights/VGGT/, "
            "set VGGT_WEIGHTS_PATH, or prepare the facebook/VGGT-1B HuggingFace cache."
        )
    from safetensors.torch import load_file

    if not BATCH_EXPERIMENT_COMPACT_MODE:
        print(f"[VGGT] loading model once: {weight_path}")
    model = VGGT(enable_track=bool(enable_track))
    state = load_file(str(weight_path), device="cpu")
    result = model.load_state_dict(state, strict=False)
    bad_missing = [
        k for k in result.missing_keys
        if bool(enable_track) or not k.startswith("track_head.")
    ]
    bad_unexpected = [
        k for k in result.unexpected_keys
        if bool(enable_track) or not k.startswith("track_head.")
    ]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"VGGT checkpoint mismatch: missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}")
    del state
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not BATCH_EXPERIMENT_COMPACT_MODE:
        print(f"[VGGT] model ready on {device}; track_head={bool(enable_track)}")
    return model


def safe_path_name(name: str) -> str:
    safe = "".join(ch if ch not in '<>:"/\\|?*' else "_" for ch in str(name))
    return safe.strip().strip(".") or "dataset"


def load_debug_failure_dataset_names() -> list[str]:
    """Load the curated debug subset generated from the 216-3 failure report."""
    path = Path(DEBUG_FAILURE_DATASETS_JSON)
    if not path.is_file():
        raise FileNotFoundError(
            "RUN_DEBUG_FAILURE_DATASETS_ONLY=True but the debug subset JSON "
            f"does not exist: {path}"
        )
    with open(path, "r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    names = payload.get("dataset_names") if isinstance(payload, dict) else None
    if not isinstance(names, list):
        raise ValueError(
            "Debug subset JSON must contain a list field named 'dataset_names': "
            f"{path}"
        )
    result: list[str] = []
    seen: set[str] = set()
    for value in names:
        name = str(value).strip()
        if name and name not in seen:
            result.append(name)
            seen.add(name)
    if not result:
        raise ValueError(f"Debug subset JSON has no usable dataset_names: {path}")
    return result


def discover_batch_datasets() -> list[str]:
    debug_names = load_debug_failure_dataset_names() if RUN_DEBUG_FAILURE_DATASETS_ONLY else None
    if BATCH_DATASET_NAMES:
        names = BATCH_DATASET_NAMES
        if debug_names is not None:
            debug_set = set(debug_names)
            names = [name for name in names if name in debug_set]
    elif debug_names is not None:
        names = debug_names
    else:
        names = [path.name for path in sorted(DATASET_ROOT.iterdir()) if path.is_dir()]
    valid: list[str] = []
    for name in names:
        dataset_dir = DATASET_ROOT / name
        image_dir = dataset_dir / "Thermal"
        csv_path = dataset_dir / "Thermal_info.csv"
        if not dataset_dir.is_dir() or not image_dir.is_dir() or not csv_path.exists():
            if not BATCH_EXPERIMENT_COMPACT_MODE:
                print(f"[BATCH][SKIP] {name}: missing Thermal/ or Thermal_info.csv")
            continue
        image_count = sum(1 for pattern in ("*.JPG", "*.jpg", "*.png") for _ in image_dir.glob(pattern))
        if image_count <= 0:
            if not BATCH_EXPERIMENT_COMPACT_MODE:
                print(f"[BATCH][SKIP] {name}: no thermal images")
            continue
        valid.append(name)
    if BATCH_MAX_DATASETS > 0:
        valid = valid[:BATCH_MAX_DATASETS]
    return valid


def create_runtime_context(batch_save_root: Path | None = None) -> dict[str, Any]:
    seed_everything(GLOBAL_RANDOM_SEED)
    legacy = load_module_from_path("legacy_minima_se2_runtime", PACKAGE_ROOT / "geometry_io.py")
    vggt_helpers = load_module_from_path("vggt_window_helpers", PACKAGE_ROOT / "reconstruction.py")

    class Opt:
        device = "cuda"
        pose_priori = "yp"
        Ref_type = REF_TYPE
        resize_ratio = 1.0
        crop_size_m = 1200
        crop_gain = 1

    opt = Opt()
    opt.save_dir = str(batch_save_root or (ROOT / "Result"))
    region_path = REGION_CONFIG_PATH
    with open(region_path, "r", encoding="utf-8") as file:
        region_config = yaml.safe_load(file)
    ref_map_full, _, _, _ = legacy.load_config_parameters_new(region_config, opt, REGION_NAME)
    if ref_map_full is None:
        raise RuntimeError("Cannot load satellite reference map")
    satellite_gsd = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_resolution"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_vggt_once(device, enable_track=VGGT_FIXED_K_TRACK_BA_ENABLED)
    anchor_match_method = str(ANCHOR_MATCH_METHOD or "minima").lower()
    template_module = None
    template_dinov2 = None
    if anchor_match_method in ("minima", "roma", "minima_roma"):
        from scalloc.matchers.minima import MINIMA_Init
        if BATCH_EXPERIMENT_COMPACT_MODE:
            with redirect_stdout(io.StringIO()):
                matcher = MINIMA_Init(MINIMA_METHOD)
        else:
            matcher = MINIMA_Init(MINIMA_METHOD)
        anchor_match_method = "minima"
        if ENABLE_BEV_GUIDED_LY_RETRY and BEV_GUIDED_LY_USE_TEMPLATE_COARSE:
            template_module = load_module_from_path(
                "template_match_optimizer_peakshape_runtime",
                template_match_optimizer_path(),
            )
            template_args_for_model = SimpleNamespace(
                dino_repo="DINOv2",
                dino_model="dinov2_vitb14",
            )
            template_dinov2 = template_module.load_dinov2_model(
                template_args_for_model, device
            )
    elif anchor_match_method == "template":
        matcher = None
        template_module = load_module_from_path(
            "template_match_optimizer_peakshape_runtime",
            template_match_optimizer_path(),
        )
        template_args_for_model = SimpleNamespace(
            dino_repo="DINOv2",
            dino_model="dinov2_vitb14",
        )
        template_dinov2 = template_module.load_dinov2_model(template_args_for_model, device)
    else:
        raise ValueError(f"Unsupported ANCHOR_MATCH_METHOD={ANCHOR_MATCH_METHOD!r}; use 'template' or 'minima'.")
    if not BATCH_EXPERIMENT_COMPACT_MODE:
        print(f"[ANCHOR] match method: {anchor_match_method}")
    return {
        "legacy": legacy,
        "vggt_helpers": vggt_helpers,
        "opt": opt,
        "region_config": region_config,
        "ref_map_full": ref_map_full,
        "satellite_gsd": satellite_gsd,
        "device": device,
        "model": model,
        "matcher": matcher,
        "anchor_match_method": anchor_match_method,
        "template_module": template_module,
        "template_dinov2": template_dinov2,
    }


def classify_first_window_orbit_motion(geometry: WindowGeometry) -> dict[str, Any]:
    """Classify centre-facing circular motion from VGGT-relative geometry only.

    Consecutive camera translations should be tangential to the horizontal
    viewing direction, while that direction rotates consistently in window 0.
    No GT or satellite result is used by this decision.
    """
    camera_xy = np.asarray(geometry.camera_xy, dtype=np.float64).reshape(-1, 2)
    forward_xy = np.asarray(geometry.camera_forward_xy, dtype=np.float64).reshape(-1, 2)
    pair_count = max(0, len(camera_xy) - 1)
    result: dict[str, Any] = {
        "enabled": bool(ENABLE_ORBIT_LEGACY_BEV_PIPELINE),
        "pair_count": int(pair_count),
        "valid_pair_count": 0,
        "support_ratio": 0.0,
        "perpendicular_angle_median_deg": float("nan"),
        "perpendicular_angle_p10_deg": float("nan"),
        "net_yaw_change_deg": float("nan"),
        "total_yaw_change_deg": float("nan"),
        "yaw_direction_consistency": float("nan"),
        "orbit_like": False,
        "reason": "too_few_frames",
    }
    if pair_count < 1:
        return result
    motion = np.diff(camera_xy, axis=0)
    motion_norm = np.linalg.norm(motion, axis=1)
    finite_motion = np.isfinite(motion).all(axis=1) & np.isfinite(motion_norm)
    positive_motion = motion_norm[finite_motion & (motion_norm > 1e-9)]
    if len(positive_motion) == 0:
        result["reason"] = "no_finite_relative_translation"
        return result
    # Relative to the window's own scale: this is available before anchoring.
    min_motion = max(1e-9, 0.05 * float(np.median(positive_motion)))
    view = forward_xy[:-1] + forward_xy[1:]
    view_norm = np.linalg.norm(view, axis=1)
    valid = (
        finite_motion
        & (motion_norm >= min_motion)
        & np.isfinite(view).all(axis=1)
        & np.isfinite(view_norm)
        & (view_norm > 1e-9)
    )
    valid_count = int(np.count_nonzero(valid))
    result["valid_pair_count"] = valid_count
    result["support_ratio"] = float(valid_count / max(pair_count, 1))
    result["minimum_motion_local"] = float(min_motion)
    if valid_count < int(ORBIT_LEGACY_BEV_MIN_VALID_PAIRS):
        result["reason"] = "too_few_valid_translation_view_pairs"
        return result
    motion_dir = motion[valid] / motion_norm[valid, None]
    view_dir = view[valid] / view_norm[valid, None]
    perpendicular_angle = np.degrees(np.arccos(np.clip(
        np.abs(np.sum(motion_dir * view_dir, axis=1)), 0.0, 1.0,
    )))
    headings = np.degrees(np.arctan2(forward_xy[:, 1], forward_xy[:, 0]))
    heading_delta = np.asarray([
        wrap_deg(float(current) - float(previous))
        for previous, current in zip(headings[:-1], headings[1:])
        if np.isfinite(previous) and np.isfinite(current)
    ], dtype=np.float64)
    total_yaw = float(np.sum(np.abs(heading_delta))) if len(heading_delta) else 0.0
    net_yaw = float(wrap_deg(float(headings[-1]) - float(headings[0])))
    direction_consistency = (
        float(abs(np.sum(heading_delta)) / max(total_yaw, 1e-9))
        if len(heading_delta) else 0.0
    )
    median_angle = float(np.median(perpendicular_angle))
    result.update({
        "perpendicular_angle_median_deg": median_angle,
        "perpendicular_angle_p10_deg": float(np.percentile(perpendicular_angle, 10.0)),
        "net_yaw_change_deg": net_yaw,
        "total_yaw_change_deg": total_yaw,
        "yaw_direction_consistency": direction_consistency,
        "per_pair_perpendicular_angle_deg": perpendicular_angle.tolist(),
    })
    result["orbit_like"] = bool(
        result["support_ratio"] >= float(ORBIT_LEGACY_BEV_MIN_SUPPORT_RATIO)
        and median_angle >= float(ORBIT_LEGACY_BEV_MIN_PERPENDICULAR_ANGLE_DEG)
        and abs(net_yaw) >= float(ORBIT_LEGACY_BEV_MIN_NET_YAW_CHANGE_DEG)
        and total_yaw >= float(ORBIT_LEGACY_BEV_MIN_TOTAL_YAW_CHANGE_DEG)
        and direction_consistency >= 0.55
    )
    result["reason"] = "orbit_like" if result["orbit_like"] else "motion_not_consistently_centre_facing_orbit"
    return result


def run_pipeline(
    dataset_name: str | None = None,
    runtime: dict[str, Any] | None = None,
    batch_save_root: Path | None = None,
) -> dict[str, Any]:
    run_start = time.perf_counter()
    stage_timings: dict[str, dict[str, float | int]] = {}
    window_timing_rows: list[dict[str, Any]] = []
    seed_everything(GLOBAL_RANDOM_SEED)
    compact_mode = bool(BATCH_EXPERIMENT_COMPACT_MODE)
    save_all_visualizations = bool(SAVE_ALL_VISUALIZATIONS)
    save_intermediate_outputs = not compact_mode
    save_window_point_clouds = save_all_visualizations
    save_window_vggt_intrinsics = save_all_visualizations
    dataset_name = dataset_name or DATASET_NAME
    if runtime is None:
        stage_started = time.perf_counter()
        runtime = create_runtime_context(batch_save_root)
        record_stage_timing(stage_timings, "runtime_initialization", stage_started)
    legacy = runtime["legacy"]
    vggt_helpers = runtime["vggt_helpers"]
    opt = runtime["opt"]
    region_config = runtime["region_config"]
    ref_map_full = runtime["ref_map_full"]
    satellite_gsd = float(runtime["satellite_gsd"])
    device = runtime["device"]
    model = runtime["model"]
    matcher = runtime["matcher"]
    anchor_match_method = runtime["anchor_match_method"]
    template_module = runtime["template_module"]
    template_dinov2 = runtime["template_dinov2"]

    dataset_dir = DATASET_ROOT / dataset_name
    image_dir = dataset_dir / "Thermal"
    csv_path = dataset_dir / "Thermal_info.csv"
    image_paths = sorted({
        Path(p).resolve()
        for pattern in ("*.JPG", "*.jpg", "*.png")
        for p in glob.glob(str(image_dir / pattern))
    })
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    gt_df = pd.read_csv(csv_path)
    gt_by_stem = {
        Path(str(row["filename"])).stem: row
        for _, row in gt_df.iterrows()
        if "filename" in row and pd.notna(row["filename"])
    }
    first = gt_df.iloc[0]
    first_gt = {
        "name": str(first["filename"]),
        "lat": float(first["lat"]),
        "lon": float(first["lon"]),
        "rel_height": float(first["rel_height"]),
        "abs_height": float(first["abs_height"]),
        "pitch": float(first["pitch"]),
        "roll": float(first["roll"]),
        "yaw": float(first["yaw"]),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if batch_save_root is None:
        save_dir = ROOT / "Result" / f"VGGT_Window_MINIMA_{timestamp}_{safe_path_name(dataset_name)}"
    else:
        save_dir = batch_save_root / safe_path_name(dataset_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    opt.save_dir = str(save_dir)
    first_e, first_n = legacy.deg2utm(REGION_NAME, region_config, first_gt["lon"], first_gt["lat"])
    if not compact_mode:
        print(f"\n[BATCH][DATASET] {dataset_name} -> {save_dir}")

    frame_names = [path.stem for path in image_paths]
    frame_observations: dict[int, list[dict[str, Any]]] = {}
    uncorrected_frame_observations: dict[int, list[dict[str, Any]]] = {}
    window_summaries: list[dict[str, Any]] = []
    visualization_anchors: list[dict[str, Any]] = []
    states: list[WindowState] = []
    uncorrected_states: list[WindowState] = []
    # LY keeps the first accepted absolute prediction for every global frame
    # and fits later overlapping windows against that stable anchor set.
    ly_accepted_anchor_by_frame: dict[int, np.ndarray] = {}
    consecutive_anchor_rejections = 0
    pending_bev_rescue: dict[str, Any] | None = None
    first_window_height_scale_m_per_local: float | None = None
    first_window_height_only_scale_m_per_local: float | None = None
    sequence_metric_scale_multiplier = metric_scale_multiplier_for_dataset(dataset_name)
    sequence_metric_scale_log_variance = float(SEQUENTIAL_ANCHOR_SCALE_INITIAL_STD_LOG ** 2)
    sequence_uses_legacy_bev_orbit = False
    orbit_motion_diagnostics: dict[str, Any] = {
        "enabled": bool(ENABLE_ORBIT_LEGACY_BEV_PIPELINE),
        "reason": "awaiting_first_window",
        "orbit_like": False,
    }
    sequence_sequential_scale_filter_enabled = bool(SEQUENTIAL_ANCHOR_SCALE_FILTER_ENABLED)
    sequence_metric_scale_filter_active = not bool(sequence_sequential_scale_filter_enabled)
    sequence_metric_scale_warmup: list[dict[str, float]] = []

    starts = list(range(0, len(image_paths), VGGT_WINDOW_STRIDE))
    starts = [s for s in starts if len(image_paths) - s >= 2]
    if VGGT_MAX_WINDOWS > 0:
        starts = starts[:VGGT_MAX_WINDOWS]
    window_progress = tqdm(
        starts,
        desc=f"{dataset_name} windows",
        unit="window",
        leave=False,
        disable=not compact_mode,
    )
    for window_idx, start in enumerate(window_progress):
        window_started = time.perf_counter()
        end = min(start + VGGT_WINDOW_SIZE, len(image_paths))
        indices = list(range(start, end))
        paths = [image_paths[i] for i in indices]
        names = [frame_names[i] for i in indices]
        window_timing_row: dict[str, Any] = {
            "window_idx": int(window_idx),
            "start_idx": int(start),
            "end_idx": int(end),
            "frame_count": int(len(indices)),
        }
        frame_metadata: list[dict[str, float]] = []
        for global_idx, frame_name in zip(indices, names):
            row = gt_by_stem.get(Path(frame_name).stem)
            if row is None and 0 <= int(global_idx) < len(gt_df):
                row = gt_df.iloc[int(global_idx)]
            pitch_value = float(row["pitch"]) if row is not None else float("nan")
            rel_height_value = (
                float(row["rel_height"]) if row is not None else float("nan")
            )
            abs_height_value = (
                float(row["abs_height"]) if row is not None else float("nan")
            )
            # Perturb every frame after the sequence's first frame; all later
            # calculations continue through the existing frame_metadata path.
            if (
                ENABLE_FRAME_PITCH_HEIGHT_PERTURBATION
                and int(global_idx) > 0
            ):
                rng = np.random.default_rng(
                    int(GLOBAL_RANDOM_SEED) + 7919 * int(global_idx)
                )
                if np.isfinite(pitch_value) and PITCH_NOISE_MAX_ABS_DEG > 0.0:
                    pitch_value += float(
                        rng.uniform(-PITCH_NOISE_MAX_ABS_DEG, PITCH_NOISE_MAX_ABS_DEG)
                    )
                if np.isfinite(rel_height_value) and HEIGHT_NOISE_MAX_ABS_FRACTION > 0.0:
                    rel_height_value *= 1.0 + float(
                        rng.uniform(
                            -HEIGHT_NOISE_MAX_ABS_FRACTION,
                            HEIGHT_NOISE_MAX_ABS_FRACTION,
                        )
                    )
            frame_metadata.append({
                "pitch_deg": pitch_value,
                "rel_height_m": rel_height_value,
                "abs_height_m": abs_height_value,
            })
        window_dir = save_dir / "windows" / f"window_{window_idx:04d}_{start:04d}_{end - 1:04d}"
        if save_intermediate_outputs or save_all_visualizations:
            window_dir.mkdir(parents=True, exist_ok=True)
        if not compact_mode:
            print(f"\n[WINDOW {window_idx:04d}] frames={start}:{end} VGGT local reconstruction")
        stage_started = time.perf_counter()
        prediction = vggt_helpers.run_vggt_reconstruction(
            model=model,
            image_paths=paths,
            device=device,
            preprocess="pad",
            target_size=VGGT_TARGET_SIZE,
            conf_percentile=VGGT_CONF_PERCENTILE,
            max_points=VGGT_MAX_POINTS,
            seed=GLOBAL_RANDOM_SEED + window_idx,
            use_depth_unproject=VGGT_USE_DEPTH_UNPROJECT,
            fixed_source_intrinsics=(
                VGGT_FIXED_SOURCE_INTRINSICS
                if (VGGT_FIXED_SOURCE_INTRINSICS_ENABLED or VGGT_FIXED_K_TRACK_BA_ENABLED)
                else None
            ),
            fixed_k_track_ba=VGGT_FIXED_K_TRACK_BA_ENABLED,
            track_query_columns=VGGT_TRACK_BA_QUERY_COLUMNS,
            track_query_rows=VGGT_TRACK_BA_QUERY_ROWS,
        )
        record_stage_timing(
            stage_timings, "vggt_reconstruction", stage_started, window_timing_row
        )
        canonical_pitch_deg = (
            -float(frame_metadata[0]["pitch_deg"])
            if (
                ENABLE_LY_DIRECT_ANCHOR_PIPELINE
                and not sequence_uses_legacy_bev_orbit
                and not USE_VGGT_PITCH_HEIGHT
                and frame_metadata
            )
            else None
        )
        stage_started = time.perf_counter()
        geometry, plane_stats = build_ground_geometry(
            prediction,
            GLOBAL_RANDOM_SEED + window_idx,
            canonical_pitch_deg=canonical_pitch_deg,
        )
        record_stage_timing(
            stage_timings, "ground_geometry", stage_started, window_timing_row
        )
        if window_idx == 0:
            orbit_motion_diagnostics = classify_first_window_orbit_motion(geometry)
            sequence_uses_legacy_bev_orbit = bool(
                ENABLE_ORBIT_LEGACY_BEV_PIPELINE
                and orbit_motion_diagnostics.get("orbit_like", False)
            )
            # The legacy BEV branch did not canonicalize the point surface with
            # the LY pitch prior.  Rebuild only the first orbit window so the
            # selected branch uses that same surface convention from its start.
            if sequence_uses_legacy_bev_orbit and canonical_pitch_deg is not None:
                stage_started = time.perf_counter()
                geometry, plane_stats = build_ground_geometry(
                    prediction,
                    GLOBAL_RANDOM_SEED + window_idx,
                    canonical_pitch_deg=None,
                )
                record_stage_timing(
                    stage_timings, "ground_geometry_legacy_bev_rebuild", stage_started,
                    window_timing_row,
                )
                orbit_motion_diagnostics["legacy_bev_surface_rebuilt"] = True
            sequence_sequential_scale_filter_enabled = bool(
                sequence_uses_legacy_bev_orbit
                or SEQUENTIAL_ANCHOR_SCALE_FILTER_ENABLED
            )
            sequence_metric_scale_filter_active = not bool(
                sequence_sequential_scale_filter_enabled
            )
        window_timing_row.update({
            "orbit_legacy_bev_active": bool(sequence_uses_legacy_bev_orbit),
            "orbit_detector_reason": str(orbit_motion_diagnostics.get("reason", "")),
            "orbit_detector_angle_median_deg": orbit_motion_diagnostics.get(
                "perpendicular_angle_median_deg"
            ),
            "orbit_detector_support_ratio": orbit_motion_diagnostics.get("support_ratio"),
        })

        stage_started = time.perf_counter()
        if window_idx == 0:
            prior, prior_stats = sim2_from_first_gt(
                geometry,
                first_e,
                first_n,
                first_gt["yaw"],
                first_gt["rel_height"],
                metric_scale_multiplier=sequence_metric_scale_multiplier,
            )
            window_height_scale_m_per_local = float(prior_stats["scale_m_per_local"])
            first_window_height_scale_m_per_local = window_height_scale_m_per_local
            uncorrected_window_height_scale_m_per_local = window_height_scale_m_per_local
            first_window_height_only_scale_m_per_local = float(
                prior_stats["height_only_scale_m_per_local"]
            )
            height_scale_stats = {
                "source": "first_frame_gt_height",
                "scale_m_per_local": window_height_scale_m_per_local,
            }
            prior_stats.update({"source": "first_frame_gt", "accepted": True})
            uncorrected_matrix = prior.copy()
            uncorrected_prior_stats = dict(prior_stats)
            uncorrected_prior_stats["source"] = "first_frame_gt_no_satellite_anchor"
        else:
            if (
                first_window_height_scale_m_per_local is None
                or first_window_height_only_scale_m_per_local is None
            ):
                raise RuntimeError("First-window height scale was not initialized")
            # Causal update: this window sees scale observations only from
            # earlier anchors.  Existing anchored windows are never rescaled.
            window_height_scale_m_per_local = float(
                first_window_height_only_scale_m_per_local * sequence_metric_scale_multiplier
            )
            uncorrected_window_height_scale_m_per_local = float(first_window_height_scale_m_per_local)
            height_scale_stats = {
                "source": "first_window_height_scale_with_causal_anchor_filter",
                "first_frame_gt_rel_height_m": float(first_gt["rel_height"]),
                "first_window_scale_m_per_local": float(first_window_height_scale_m_per_local),
                "first_window_height_only_scale_m_per_local": float(
                    first_window_height_only_scale_m_per_local
                ),
                "sequential_anchor_scale_multiplier": float(sequence_metric_scale_multiplier),
                "scale_m_per_local": float(window_height_scale_m_per_local),
                "assumption": "causal_anchor_scale_updates_only_affect_future_windows",
            }
            if VGGT_WINDOW_STITCH_MODE == 1:
                prior, prior_stats = prior_from_shared_last_pose(
                    geometry, indices, states[-1], window_height_scale_m_per_local
                )
                uncorrected_matrix, uncorrected_prior_stats = prior_from_shared_last_pose(
                    geometry,
                    indices,
                    uncorrected_states[-1],
                    uncorrected_window_height_scale_m_per_local,
                )
            elif VGGT_WINDOW_STITCH_MODE == 0:
                if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit:
                    prior, prior_stats = prior_from_ly_accepted_anchor_map(
                        geometry, indices, ly_accepted_anchor_by_frame
                    )
                    if prior is None or not bool(prior_stats.get("accepted", False)):
                        fallback_reason = dict(prior_stats)
                        prior, prior_stats = prior_from_overlap(geometry, indices, states[-1])
                        prior_stats["ly_anchor_map_fallback"] = fallback_reason
                else:
                    prior, prior_stats = prior_from_overlap(geometry, indices, states[-1])
                uncorrected_matrix, uncorrected_prior_stats = prior_from_overlap(
                    geometry, indices, uncorrected_states[-1]
                )
            else:
                raise ValueError(
                    f"Unsupported VGGT_WINDOW_STITCH_MODE={VGGT_WINDOW_STITCH_MODE}; use 0 or 1"
                )
            if prior is None or not bool(prior_stats.get("accepted", False)):
                raise RuntimeError(
                    f"Window {window_idx} cannot inherit its corrected-window prior: {prior_stats}"
                )
            prior_stats["stitch_mode"] = int(VGGT_WINDOW_STITCH_MODE)
            if uncorrected_matrix is None or not bool(uncorrected_prior_stats.get("accepted", False)):
                raise RuntimeError(
                    f"Window {window_idx} cannot inherit its no-anchor prior: "
                    f"{uncorrected_prior_stats}"
                )
            uncorrected_prior_stats["stitch_mode"] = int(VGGT_WINDOW_STITCH_MODE)

        if not np.isfinite(window_height_scale_m_per_local) or window_height_scale_m_per_local <= 0.0:
            raise RuntimeError(f"Window {window_idx} height scale is invalid")
        record_stage_timing(
            stage_timings, "window_prior_and_stitch", stage_started, window_timing_row
        )

        camera_utm_before_anchor = apply_sim2(prior, geometry.camera_xy)
        camera_yaw_before_anchor = yaw_from_forward(prior, geometry.camera_forward_xy)
        uncorrected_camera_utm = apply_sim2(uncorrected_matrix, geometry.camera_xy)
        uncorrected_camera_yaw = yaw_from_forward(uncorrected_matrix, geometry.camera_forward_xy)
        prior_window_ypr = vggt_window_ypr(
            prediction.extrinsics,
            geometry,
            prior,
        )
        satellite_crop_prior = build_middle_frame_pitch_crop_prior(
            geometry,
            window_height_scale_m_per_local,
            camera_utm_before_anchor,
            prior_window_ypr,
            names,
        )

        stage_started = time.perf_counter()
        if sequence_uses_legacy_bev_orbit:
            # This is intentionally the old BEV-only anchor route: no LY
            # candidate, no rescue status, and no dual-candidate arbitration.
            multiframe_candidate = None
            multiframe_anchor_stats = {
                "enabled": False,
                "accepted": False,
                "reason": "skipped_by_orbit_legacy_bev_state",
                "match_method": "ly_direct_skipped_for_orbit",
            }
        else:
            multiframe_candidate, multiframe_anchor_stats = estimate_multiframe_3d_utm_anchor(
                legacy=legacy,
                matcher=matcher,
                prediction=prediction,
                geometry=geometry,
                window_idx=window_idx,
                image_paths=paths,
                frame_names=names,
                prior=prior,
                camera_utm_prior=camera_utm_before_anchor,
                camera_yaw_prior=camera_yaw_before_anchor,
                predicted_ypr=prior_window_ypr,
                frame_metadata=frame_metadata,
                height_scale_m_per_local=window_height_scale_m_per_local,
                ref_map_full=ref_map_full,
                region_config=region_config,
                out_dir=window_dir,
                save_visualizations=save_all_visualizations,
            )
        record_stage_timing(
            stage_timings, "direct_multiframe_anchor_matching", stage_started,
            window_timing_row,
        )
        direct_detail = multiframe_anchor_stats.get("all_yaw_attempt_timing", {})
        window_timing_row.update({
            "direct_yaw_attempt_count": int(direct_detail.get("attempt_count", 0)),
            "direct_minima_call_count": int(direct_detail.get("minima_call_count", 0)),
            "direct_minima_matching_sec": float(
                direct_detail.get("minima_matching_sec", 0.0)
            ),
            "direct_raycast_surface_sec": float(
                direct_detail.get("raycast_surface_sec", 0.0)
            ),
            "direct_similarity_ransac_sec": float(
                direct_detail.get("similarity_ransac_sec", 0.0)
            ),
        })

        ly_gate_distance = multiframe_anchor_stats.get(
            "ly_overlap_position_gate_distance_m"
        )
        ly_large_innovation = bool(
            ly_gate_distance is not None
            and np.isfinite(float(ly_gate_distance))
            and float(ly_gate_distance) > float(MULTIFRAME_GATE_STRONG_DISTANCE_M)
        )
        bev_rescue_triggered = bool(
            not sequence_uses_legacy_bev_orbit
            and ENABLE_BEV_RESCUE_MODE
            and (
                multiframe_candidate is None
                or ly_large_innovation
                or consecutive_anchor_rejections > 0
            )
        )
        run_bev_branch = bool(
            sequence_uses_legacy_bev_orbit
            or ENABLE_BEV_ANCHOR
            or bev_rescue_triggered
        )
        rescue_level = int(
            max(
                consecutive_anchor_rejections,
                1 if bev_rescue_triggered and multiframe_candidate is None else 0,
            )
        )
        bev_extra_margin_m = float(min(
            float(BEV_RESCUE_MAX_EXTRA_MARGIN_M),
            float(BEV_RESCUE_EXTRA_MARGIN_STEP_M) * rescue_level,
        )) if bev_rescue_triggered else 0.0
        bev_guided_ly_retry_requested = bool(
            ENABLE_BEV_GUIDED_LY_RETRY
            and not sequence_uses_legacy_bev_orbit
            and ENABLE_LY_DIRECT_ANCHOR_PIPELINE
            and ENABLE_MULTIFRAME_3D_UTM_ANCHOR
            and ENABLE_BEV_RESCUE_MODE
            and (
                multiframe_candidate is None
                or ly_large_innovation
                or consecutive_anchor_rejections > 0
            )
        )
        if bev_guided_ly_retry_requested:
            bev_extra_margin_m = float(
                max(bev_extra_margin_m, float(BEV_GUIDED_LY_EXTRA_MARGIN_M))
            )

        bev_meta: dict[str, Any] = {
            "enabled": bool(run_bev_branch),
            "rescue_triggered": bool(bev_rescue_triggered),
            "rescue_level": int(rescue_level),
            "rescue_extra_margin_m": float(bev_extra_margin_m),
            "bev_guided_ly_retry_requested": bool(bev_guided_ly_retry_requested),
        }
        bev_residual: np.ndarray | None = None
        bev_anchor_stats: dict[str, Any] = {
            "enabled": bool(run_bev_branch),
            "accepted": False,
            "reason": "not_triggered",
            "match_method": f"bev_{anchor_match_method}",
            "scale_observation_only": False,
            "rescue_triggered": bool(bev_rescue_triggered),
            "consecutive_anchor_rejections_before": int(
                consecutive_anchor_rejections
            ),
            "rescue_extra_margin_m": float(bev_extra_margin_m),
            "bev_guided_ly_retry_requested": bool(bev_guided_ly_retry_requested),
        }
        stage_started = time.perf_counter()
        if run_bev_branch:
            bev, bev_mask, bev_meta = rasterize_prior_bev(
                geometry, prediction.colors_rgb, prior, satellite_gsd
            )
            bev_meta["middle_frame_pitch_crop_prior"] = satellite_crop_prior
            bev_meta.update({
                "enabled": True,
                "rescue_triggered": bool(bev_rescue_triggered),
                "rescue_level": int(rescue_level),
                "rescue_extra_margin_m": float(bev_extra_margin_m),
            })
        record_stage_timing(
            stage_timings, "bev_rasterization", stage_started, window_timing_row
        )
        stage_started = time.perf_counter()
        use_template_bev_coarse = bool(
            run_bev_branch
            and template_module is not None
            and template_dinov2 is not None
            and (
                anchor_match_method == "template"
                or (
                    bev_guided_ly_retry_requested
                    and BEV_GUIDED_LY_USE_TEMPLATE_COARSE
                )
            )
        )
        if use_template_bev_coarse:
            # The template optimizer needs temporary image files internally.
            # With the single visualization switch off, keep those files only
            # in an auto-cleaned temporary folder.
            if not save_all_visualizations:
                with tempfile.TemporaryDirectory(prefix="vggt_template_") as temp_dir:
                    bev_residual, bev_anchor_stats = match_bev_anchor_template(
                        legacy, bev, bev_mask, bev_meta, ref_map_full, region_config,
                        Path(temp_dir), satellite_crop_prior=satellite_crop_prior,
                        template_module=template_module, dinov2_model=template_dinov2, device=device,
                        save_visualizations=False,
                        extra_margin_m=bev_extra_margin_m,
                    )
            else:
                bev_residual, bev_anchor_stats = match_bev_anchor_template(
                    legacy,
                    bev,
                    bev_mask,
                    bev_meta,
                    ref_map_full,
                    region_config,
                    window_dir,
                    satellite_crop_prior=satellite_crop_prior,
                    template_module=template_module,
                    dinov2_model=template_dinov2,
                    device=device,
                    save_visualizations=True,
                    extra_margin_m=bev_extra_margin_m,
                )
        elif run_bev_branch:
            bev_residual, bev_anchor_stats = match_bev_anchor(
                legacy,
                matcher,
                bev,
                bev_mask,
                bev_meta,
                ref_map_full,
                region_config,
                window_dir,
                satellite_crop_prior=satellite_crop_prior,
                save_visualizations=save_all_visualizations,
                extra_margin_m=bev_extra_margin_m,
            )
        if run_bev_branch:
            bev_anchor_stats["bev_coarse_matcher"] = (
                "template" if use_template_bev_coarse else "minima"
            )
            bev_anchor_stats["bev_guided_ly_use_template_coarse"] = bool(
                BEV_GUIDED_LY_USE_TEMPLATE_COARSE
            )
        bev_anchor_stats.update({
            "rescue_triggered": bool(bev_rescue_triggered),
            "consecutive_anchor_rejections_before": int(
                consecutive_anchor_rejections
            ),
            "rescue_extra_margin_m": float(bev_extra_margin_m),
            "bev_guided_ly_retry_requested": bool(bev_guided_ly_retry_requested),
        })
        record_stage_timing(
            stage_timings, "bev_anchor_matching", stage_started, window_timing_row
        )
        bev_guided_ly_stats: dict[str, Any] = {
            "enabled": bool(ENABLE_BEV_GUIDED_LY_RETRY),
            "requested": bool(bev_guided_ly_retry_requested),
            "executed": False,
            "accepted": False,
            "reason": "not_requested" if not bev_guided_ly_retry_requested else "not_executed",
            "bev_extra_margin_m": float(bev_extra_margin_m),
        }
        if (
            bev_guided_ly_retry_requested
            and bev_residual is not None
            and not bool(bev_anchor_stats.get("scale_observation_only", False))
            and matcher is not None
        ):
            bev_quality_ok = bool(bev_anchor_stats.get("accepted", False))
            bev_inliers = int(bev_anchor_stats.get("inlier_count", 0))
            bev_ratio = float(bev_anchor_stats.get("inlier_ratio", 0.0))
            bev_rmse = float(bev_anchor_stats.get("rmse", float("inf")))
            bev_scale = float(bev_anchor_stats.get("scale", 1.0))
            bev_scale_log = abs(math.log(max(bev_scale, 1e-12)))
            bev_quality_ok = bool(
                bev_quality_ok
                and bev_inliers >= int(BEV_GUIDED_LY_MIN_INLIERS)
                and bev_ratio >= float(BEV_GUIDED_LY_MIN_INLIER_RATIO)
                and bev_rmse <= float(BEV_GUIDED_LY_MAX_RMSE_M)
                and bev_scale_log <= float(BEV_GUIDED_LY_MAX_SCALE_LOG)
            )
            bev_guided_ly_stats.update({
                "bev_quality_ok": bool(bev_quality_ok),
                "bev_inlier_count": int(bev_inliers),
                "bev_inlier_ratio": float(bev_ratio),
                "bev_rmse_m": float(bev_rmse) if np.isfinite(bev_rmse) else None,
                "bev_scale": float(bev_scale),
                "bev_scale_log_abs": float(bev_scale_log),
                "min_bev_inliers": int(BEV_GUIDED_LY_MIN_INLIERS),
                "min_bev_inlier_ratio": float(BEV_GUIDED_LY_MIN_INLIER_RATIO),
                "max_bev_rmse_m": float(BEV_GUIDED_LY_MAX_RMSE_M),
                "max_bev_scale_log": float(BEV_GUIDED_LY_MAX_SCALE_LOG),
            })
            if bev_quality_ok:
                stage_started = time.perf_counter()
                guided_prior = compose_sim2(bev_residual, prior)
                guided_camera_utm = apply_sim2(guided_prior, geometry.camera_xy)
                guided_camera_yaw = yaw_from_forward(
                    guided_prior, geometry.camera_forward_xy
                )
                guided_window_ypr = vggt_window_ypr(
                    prediction.extrinsics,
                    geometry,
                    guided_prior,
                )
                guided_satellite_crop_prior = build_middle_frame_pitch_crop_prior(
                    geometry,
                    window_height_scale_m_per_local,
                    guided_camera_utm,
                    guided_window_ypr,
                    names,
                )
                guided_candidate, guided_stats = estimate_multiframe_3d_utm_anchor(
                    legacy=legacy,
                    matcher=matcher,
                    prediction=prediction,
                    geometry=geometry,
                    window_idx=window_idx,
                    image_paths=paths,
                    frame_names=names,
                    prior=guided_prior,
                    camera_utm_prior=guided_camera_utm,
                    camera_yaw_prior=guided_camera_yaw,
                    predicted_ypr=guided_window_ypr,
                    frame_metadata=frame_metadata,
                    height_scale_m_per_local=window_height_scale_m_per_local,
                    ref_map_full=ref_map_full,
                    region_config=region_config,
                    out_dir=window_dir / "bev_guided_ly_retry",
                    save_visualizations=save_all_visualizations,
                )
                guided_detail = guided_stats.get("all_yaw_attempt_timing", {})
                record_stage_timing(
                    stage_timings,
                    "bev_guided_direct_multiframe_anchor_matching",
                    stage_started,
                    window_timing_row,
                )
                window_timing_row.update({
                    "bev_guided_direct_yaw_attempt_count": int(
                        guided_detail.get("attempt_count", 0)
                    ),
                    "bev_guided_direct_minima_call_count": int(
                        guided_detail.get("minima_call_count", 0)
                    ),
                    "bev_guided_direct_minima_matching_sec": float(
                        guided_detail.get("minima_matching_sec", 0.0)
                    ),
                    "bev_guided_direct_raycast_surface_sec": float(
                        guided_detail.get("raycast_surface_sec", 0.0)
                    ),
                    "bev_guided_direct_similarity_ransac_sec": float(
                        guided_detail.get("similarity_ransac_sec", 0.0)
                    ),
                })
                bev_guided_ly_stats.update({
                    "executed": True,
                    "accepted": bool(guided_candidate is not None),
                    "reason": str(guided_stats.get("reason", "")),
                    "inlier_count": int(guided_stats.get("inlier_count", 0)),
                    "inlier_ratio": float(guided_stats.get("inlier_ratio", 0.0)),
                    "rmse_m": guided_stats.get("rmse"),
                    "scale": guided_stats.get("scale"),
                    "guided_satellite_crop_prior": guided_satellite_crop_prior,
                })
                if guided_candidate is not None:
                    multiframe_candidate = guided_candidate
                    multiframe_anchor_stats = guided_stats
                    multiframe_anchor_stats["bev_guided_ly_retry_used"] = True
                    multiframe_anchor_stats["bev_guided_from_bev_anchor"] = {
                        "inlier_count": int(bev_inliers),
                        "inlier_ratio": float(bev_ratio),
                        "rmse_m": float(bev_rmse) if np.isfinite(bev_rmse) else None,
                        "scale": float(bev_scale),
                        "extra_margin_m": float(bev_extra_margin_m),
                    }
                    satellite_crop_prior = guided_satellite_crop_prior
                    bev_meta["middle_frame_pitch_crop_prior"] = satellite_crop_prior
                    bev_guided_ly_stats["reason"] = "accepted_replaced_direct_ly"
            else:
                bev_guided_ly_stats["reason"] = "bev_quality_gate_failed"
        elif bev_guided_ly_retry_requested:
            if bev_residual is None:
                bev_guided_ly_stats["reason"] = "no_accepted_bev_residual"
            elif bool(bev_anchor_stats.get("scale_observation_only", False)):
                bev_guided_ly_stats["reason"] = "bev_scale_observation_only"
            elif matcher is None:
                bev_guided_ly_stats["reason"] = "requires_minima_matcher"
        bev_anchor_stats["bev_guided_ly_retry"] = bev_guided_ly_stats
        multiframe_anchor_stats.setdefault(
            "bev_guided_ly_retry", bev_guided_ly_stats
        )
        if (
            bev_guided_ly_retry_requested
            and BEV_GUIDED_LY_USE_TEMPLATE_COARSE
            and not bool(bev_guided_ly_stats.get("accepted", False))
            and not bool(ENABLE_BEV_ANCHOR)
        ):
            bev_residual = None
            bev_anchor_stats["accepted"] = False
            bev_anchor_stats["reason"] = (
                "bev_guided_ly_retry_failed_suppress_bev_only"
            )
            bev_anchor_stats["bev_only_suppressed_without_guided_ly"] = True
        # Count the current missing LY candidate when deciding whether the
        # sequence is lost.  With a threshold of two, the second consecutive
        # LY failure can therefore confirm (but cannot originate) a BEV rescue.
        effective_rejection_count = int(consecutive_anchor_rejections)
        if multiframe_candidate is None:
            effective_rejection_count += 1
        bev_rescue_lost_mode = bool(
            not sequence_uses_legacy_bev_orbit
            and ENABLE_BEV_RESCUE_MODE
            and effective_rejection_count
            >= int(BEV_RESCUE_LOST_AFTER_REJECTIONS)
        )
        bev_anchor_stats["bev_rescue_lost_mode"] = bool(bev_rescue_lost_mode)
        multiframe_anchor_stats["bev_rescue_lost_mode"] = bool(
            bev_rescue_lost_mode
        )
        temporal_confirmation: dict[str, Any] = {
            "enabled": bool(ENABLE_BEV_RESCUE_TEMPORAL_CONFIRMATION),
            "status": "disabled",
            "confirmed": False,
            "pending_before": bool(pending_bev_rescue is not None),
            "pending_after": bool(pending_bev_rescue is not None),
        }
        if (
            not sequence_uses_legacy_bev_orbit
            and
            ENABLE_BEV_RESCUE_TEMPORAL_CONFIRMATION
            and ENABLE_BEV_RESCUE_MODE
            and not ENABLE_BEV_ANCHOR
        ):
            temporal_confirmation["status"] = "not_applicable"
            if multiframe_candidate is not None:
                # LY has recovered.  It is safer and cheaper to discard any
                # uncommitted BEV-only hypothesis than to carry stale evidence.
                temporal_confirmation["status"] = (
                    "ly_available_pending_cleared"
                    if pending_bev_rescue is not None
                    else "ly_available"
                )
                pending_bev_rescue = None
            elif (
                bev_residual is not None
                and not bool(bev_anchor_stats.get("scale_observation_only", False))
            ):
                current_bev_snapshot = make_bev_rescue_snapshot(
                    window_idx=window_idx,
                    frame_indices=indices,
                    geometry=geometry,
                    prior=prior,
                    bev_residual=bev_residual,
                )
                pending_age = (
                    int(window_idx) - int(pending_bev_rescue.get("window_idx", -1))
                    if pending_bev_rescue is not None
                    else None
                )
                if (
                    pending_bev_rescue is not None
                    and pending_age is not None
                    and 0 < pending_age <= int(BEV_RESCUE_PENDING_MAX_AGE_WINDOWS)
                ):
                    temporal_confirmation = compare_bev_rescue_snapshots(
                        pending_bev_rescue,
                        current_bev_snapshot,
                    )
                    temporal_confirmation["enabled"] = True
                    temporal_confirmation["pending_before"] = True
                    if (
                        bool(temporal_confirmation.get("confirmed", False))
                        and bev_rescue_lost_mode
                    ):
                        temporal_confirmation["status"] = "confirmed_and_lost"
                        bev_anchor_stats["bev_rescue_temporal_confirmed"] = True
                        pending_bev_rescue = None
                    else:
                        temporal_confirmation["status"] = (
                            "confirmed_waiting_for_lost"
                            if temporal_confirmation.get("confirmed", False)
                            else str(temporal_confirmation.get("reason", "disagreed"))
                        )
                        # A disagreement starts a fresh consecutive-evidence
                        # run from the current, still uncommitted hypothesis.
                        pending_bev_rescue = current_bev_snapshot
                else:
                    temporal_confirmation.update({
                        "status": (
                            "pending_expired_replaced"
                            if pending_bev_rescue is not None
                            else "pending_created"
                        ),
                        "confirmed": False,
                        "previous_window_idx": (
                            int(pending_bev_rescue.get("window_idx", -1))
                            if pending_bev_rescue is not None
                            else None
                        ),
                        "current_window_idx": int(window_idx),
                        "window_age": pending_age,
                    })
                    pending_bev_rescue = current_bev_snapshot
            elif pending_bev_rescue is not None:
                pending_age = int(window_idx) - int(
                    pending_bev_rescue.get("window_idx", -1)
                )
                if pending_age > int(BEV_RESCUE_PENDING_MAX_AGE_WINDOWS):
                    pending_bev_rescue = None
                    temporal_confirmation["status"] = "pending_expired_no_bev_candidate"
                else:
                    temporal_confirmation["status"] = "pending_retained_no_bev_candidate"
                    temporal_confirmation["window_age"] = int(pending_age)
            else:
                temporal_confirmation["status"] = "no_usable_bev_candidate"
            temporal_confirmation["pending_after"] = bool(
                pending_bev_rescue is not None
            )
        bev_anchor_stats["bev_rescue_temporal_confirmation"] = temporal_confirmation
        bev_anchor_stats.setdefault("bev_rescue_temporal_confirmed", False)
        stage_started = time.perf_counter()
        if sequence_uses_legacy_bev_orbit:
            # Equivalent to copy 34's direct BEV output: pass its residual to
            # the legacy pose/scale stage below without LY fusion or rescue.
            residual = bev_residual
            anchor_stats = dict(bev_anchor_stats)
            anchor_stats["dual_anchor_fusion"] = {
                "enabled": False,
                "decision": "orbit_legacy_bev_only",
                "bev": bev_anchor_stats,
                "multiframe": multiframe_anchor_stats,
            }
            anchor_stats["match_method"] = "orbit_legacy_bev_minima"
        else:
            # BEV is only a coarse prior for this retry path.  Once LY has
            # verified the recentered crop, its geometry is the sole pose
            # observation; do not blend the coarse BEV residual back in.
            bev_guided_ly_only_final = bool(
                bev_guided_ly_stats.get("accepted", False)
                and multiframe_anchor_stats.get("bev_guided_ly_retry_used", False)
                and not ENABLE_BEV_ANCHOR
            )
            residual, anchor_stats = fuse_bev_and_multiframe_candidates(
                prior=prior,
                pivot_local_xy=geometry.camera_xy[len(indices) // 2],
                bev_residual=None if bev_guided_ly_only_final else bev_residual,
                bev_stats=bev_anchor_stats,
                multiframe_candidate=multiframe_candidate,
                multiframe_stats=multiframe_anchor_stats,
            )
            if bev_guided_ly_only_final:
                fusion = anchor_stats.setdefault("dual_anchor_fusion", {})
                fusion.update({
                    "decision": "bev_guided_ly_only",
                    "bev_guided_ly_only_final": True,
                    "bev_excluded_from_final_fusion": True,
                    "bev": bev_anchor_stats,
                    "multiframe": multiframe_anchor_stats,
                })
                anchor_stats["dual_anchor_decision"] = "bev_guided_ly_only"
        scale_observation_only = bool(anchor_stats.get("scale_observation_only", False))
        anchor_accepted = bool(residual is not None and not scale_observation_only)
        anchor_stats["scale_observation_only"] = scale_observation_only
        rescue_fusion = anchor_stats.get("dual_anchor_fusion", {})
        rescue_consensus = rescue_fusion.get("bev_rescue_consensus", {})
        rescue_decision = str(rescue_fusion.get("decision", ""))
        bev_rescue_relocalized = bool(
            anchor_accepted
            and bev_rescue_triggered
            and (
                bool(multiframe_anchor_stats.get("bev_guided_ly_retry_used", False))
                or
                bool(rescue_consensus.get("activated", False))
                or rescue_decision in {
                    "bev_rescue_lost_relocalization",
                    "bev_rescue_temporally_confirmed_relocalization",
                    "bev_rescue_consensus_weighted_fusion",
                    "conflict_selected_bev",
                }
            )
        )
        if (
            anchor_accepted
            and bool(rescue_consensus.get("activated", False))
            and pending_bev_rescue is not None
        ):
            pending_bev_rescue = None
            temporal_confirmation["status"] = "same_window_consensus_pending_cleared"
            temporal_confirmation["pending_after"] = False
        anchor_stats["bev_rescue"] = {
            "enabled": bool(ENABLE_BEV_RESCUE_MODE),
            "triggered": bool(bev_rescue_triggered),
            "lost_mode": bool(bev_rescue_lost_mode),
            "consecutive_rejections_before": int(consecutive_anchor_rejections),
            "extra_margin_m": float(bev_extra_margin_m),
            "relocalized": bool(bev_rescue_relocalized),
            "decision": rescue_decision,
            "guided_ly_retry": bev_guided_ly_stats,
            "temporal_confirmation": temporal_confirmation,
        }
        window_timing_row.update({
            "bev_rescue_triggered": bool(bev_rescue_triggered),
            "bev_rescue_lost_mode": bool(bev_rescue_lost_mode),
            "bev_rescue_relocalized": bool(bev_rescue_relocalized),
            "bev_rescue_extra_margin_m": float(bev_extra_margin_m),
            "bev_guided_ly_retry_requested": bool(
                bev_guided_ly_stats.get("requested", False)
            ),
            "bev_guided_ly_retry_executed": bool(
                bev_guided_ly_stats.get("executed", False)
            ),
            "bev_guided_ly_retry_accepted": bool(
                bev_guided_ly_stats.get("accepted", False)
            ),
            "bev_rescue_temporal_confirmed": bool(
                temporal_confirmation.get("confirmed", False)
                and temporal_confirmation.get("status") == "confirmed_and_lost"
            ),
        })
        # The LY-style direct branch already performs its overlap-position gate
        # before returning a candidate.  Applying the legacy yaw/trajectory
        # continuity gates again would reject a transform LY itself accepts.
        method_gate_enabled = bool(
            ANCHOR_CONTINUITY_GATE_ENABLED
            and (not ENABLE_LY_DIRECT_ANCHOR_PIPELINE or sequence_uses_legacy_bev_orbit)
        )
        if anchor_match_method == "template":
            method_gate_enabled = method_gate_enabled and bool(TEMPLATE_ANCHOR_CONTINUITY_GATE_ENABLED)
        else:
            method_gate_enabled = method_gate_enabled and bool(MINIMA_ANCHOR_CONTINUITY_GATE_ENABLED)
        anchor_stats["continuity_gate_enabled"] = bool(method_gate_enabled)
        anchor_stats["continuity_gate_method"] = str(anchor_match_method)
        anchor_stats["position_innovation_warning_threshold_m"] = float(
            ANCHOR_LARGE_POSITION_INNOVATION_M
        )
        anchor_stats["large_position_innovation_blend"] = float(
            ANCHOR_LARGE_POSITION_INNOVATION_BLEND
        )
        anchor_stats["continuity_gate_yaw_threshold_deg"] = float(ANCHOR_MAX_YAW_INNOVATION_DEG)
        center_local_idx_for_gate = len(indices) // 2
        center_global_idx_for_gate = int(indices[center_local_idx_for_gate])
        center_frame_name_for_gate = str(names[center_local_idx_for_gate])
        pose_pivot_utm = np.asarray(
            camera_utm_before_anchor[center_local_idx_for_gate], dtype=np.float64
        )
        apply_satellite_yaw = bool(
            True if (ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit)
            else ANCHOR_APPLY_SATELLITE_YAW
        )
        apply_satellite_scale = bool(
            True if (ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit)
            else ANCHOR_APPLY_SATELLITE_SCALE
        )
        if sequence_sequential_scale_filter_enabled:
            # The raw scale remains an observation, but must not also alter
            # this window's Sim(2); that would double-apply the correction.
            apply_satellite_scale = False
        anchor_stats["sequential_anchor_scale_multiplier_before"] = float(
            sequence_metric_scale_multiplier
        )
        anchor_stats["sequential_anchor_scale_filter_enabled"] = bool(
            sequence_sequential_scale_filter_enabled
        )
        if residual is not None:
            raw_scale, raw_yaw_deg = sim2_scale_yaw(residual)
            scale_history = accepted_anchor_component_history(
                states,
                "satellite_raw_scale",
                int(ANCHOR_COMPONENT_CONSENSUS_MIN_HISTORY),
            )
            yaw_history = accepted_anchor_component_history(
                states,
                "satellite_raw_yaw_deg",
                int(ANCHOR_COMPONENT_CONSENSUS_MIN_HISTORY),
            )
            scale_consistent, scale_reference, scale_deviation = component_consensus(
                raw_scale,
                scale_history,
                int(ANCHOR_COMPONENT_CONSENSUS_MIN_HISTORY),
                float(ANCHOR_SCALE_CONSENSUS_REL_TOLERANCE),
            )
            yaw_consistent, yaw_reference, yaw_deviation = component_consensus(
                raw_yaw_deg,
                yaw_history,
                int(ANCHOR_COMPONENT_CONSENSUS_MIN_HISTORY),
                float(ANCHOR_YAW_CONSENSUS_DEG),
                circular=True,
            )
            if ANCHOR_ROBUST_COMPONENT_FUSION and (
                not ENABLE_LY_DIRECT_ANCHOR_PIPELINE or sequence_uses_legacy_bev_orbit
            ):
                apply_satellite_scale = apply_satellite_scale and scale_consistent
                apply_satellite_yaw = (
                    apply_satellite_yaw
                    and yaw_consistent
                    and abs(float(raw_yaw_deg)) <= float(ANCHOR_MAX_APPLIED_YAW_DEG)
                )
            if scale_observation_only:
                # Keep this match out of the current pose entirely.  Its raw
                # Sim(2) scale is evidence for future-window metric scale.
                apply_satellite_scale = False
                apply_satellite_yaw = False
                residual_for_pose = None
                applied_scale, applied_yaw_deg = 1.0, 0.0
            else:
                residual_for_pose = satellite_residual_for_pose(
                    residual,
                    pose_pivot_utm,
                    apply_yaw=apply_satellite_yaw,
                    apply_scale=apply_satellite_scale,
                )
                applied_scale, applied_yaw_deg = sim2_scale_yaw(residual_for_pose)
            anchor_stats.update({
                "satellite_raw_scale": float(raw_scale),
                "satellite_raw_yaw_deg": float(raw_yaw_deg),
                "satellite_applied_scale": float(applied_scale),
                "satellite_applied_yaw_deg": float(applied_yaw_deg),
                "satellite_apply_yaw": bool(apply_satellite_yaw),
                "satellite_apply_scale": bool(apply_satellite_scale),
                "robust_component_fusion_enabled": bool(ANCHOR_ROBUST_COMPONENT_FUSION),
                "scale_consensus_history": [float(value) for value in scale_history],
                "scale_consensus_reference": scale_reference,
                "scale_consensus_relative_deviation": scale_deviation,
                "scale_consensus_passed": bool(scale_consistent),
                "yaw_consensus_history_deg": [float(value) for value in yaw_history],
                "yaw_consensus_reference_deg": yaw_reference,
                "yaw_consensus_deviation_deg": yaw_deviation,
                "yaw_consensus_passed": bool(yaw_consistent),
            })
            suppressed_components = []
            if ANCHOR_APPLY_SATELLITE_SCALE and not apply_satellite_scale:
                suppressed_components.append("scale")
            if ANCHOR_APPLY_SATELLITE_YAW and not apply_satellite_yaw:
                suppressed_components.append("yaw")
            anchor_stats["satellite_suppressed_components"] = suppressed_components
        else:
            residual_for_pose = None
        candidate_matrix = compose_sim2(residual_for_pose, prior) if residual_for_pose is not None else None
        gt_center_row = gt_by_stem.get(Path(center_frame_name_for_gate).stem)
        gt_center_source = "filename"
        if gt_center_row is None and 0 <= center_global_idx_for_gate < len(gt_df):
            gt_center_row = gt_df.iloc[center_global_idx_for_gate]
            gt_center_source = "row_index_fallback"
        anchor_stats["gt_pitch_backprojection_enabled"] = bool(
            USE_GT_PITCH_FOR_ANCHOR_BACKPROJECTION
        )
        anchor_stats["gt_height_backprojection_enabled"] = bool(
            USE_GT_HEIGHT_FOR_ANCHOR_BACKPROJECTION
        )
        if (
            (USE_GT_PITCH_FOR_ANCHOR_BACKPROJECTION or USE_GT_HEIGHT_FOR_ANCHOR_BACKPROJECTION)
            and residual_for_pose is not None
            and candidate_matrix is not None
            and gt_center_row is not None
        ):
            target_camera, pitch_backprojection = camera_target_from_ground_anchor_with_pitch(
                residual_for_pose,
                satellite_crop_prior,
                (
                    float(gt_center_row["pitch"])
                    if USE_GT_PITCH_FOR_ANCHOR_BACKPROJECTION
                    else float(satellite_crop_prior["pred_pitch_deg"])
                ),
                (
                    float(gt_center_row["rel_height"])
                    if USE_GT_HEIGHT_FOR_ANCHOR_BACKPROJECTION
                    else None
                ),
            )
            anchor_stats["gt_pitch_backprojection"] = pitch_backprojection
            if target_camera is not None:
                candidate_matrix = candidate_matrix.copy()
                candidate_matrix[:, 2] = (
                    np.asarray(target_camera, dtype=np.float64)
                    - candidate_matrix[:, :2] @ geometry.camera_xy[center_local_idx_for_gate]
                )
                anchor_stats["gt_pitch_backprojection_target_camera_utm"] = (
                    np.asarray(target_camera, dtype=np.float64).tolist()
                )
            else:
                anchor_stats["gt_pitch_backprojection_skipped"] = True
        elif USE_GT_PITCH_FOR_ANCHOR_BACKPROJECTION or USE_GT_HEIGHT_FOR_ANCHOR_BACKPROJECTION:
            anchor_stats["gt_pitch_backprojection_skipped"] = True
        prior_center = apply_sim2(prior, geometry.camera_xy[[center_local_idx_for_gate]])[0]
        candidate_center = (
            apply_sim2(candidate_matrix, geometry.camera_xy[[center_local_idx_for_gate]])[0]
            if candidate_matrix is not None
            else None
        )
        anchor_stats["prior_center_utm"] = [float(prior_center[0]), float(prior_center[1])]
        anchor_stats["candidate_center_utm"] = (
            [float(candidate_center[0]), float(candidate_center[1])]
            if candidate_center is not None
            else None
        )
        anchor_stats["gt_center_frame"] = center_frame_name_for_gate
        try:
            if gt_center_row is None:
                raise ValueError("missing GT row")
            gt_e, gt_n = legacy.deg2utm(
                REGION_NAME,
                region_config,
                float(gt_center_row["lon"]),
                float(gt_center_row["lat"]),
            )
            gt_center = np.array([float(gt_e), float(gt_n)], dtype=np.float64)
            prior_gt_error = float(np.linalg.norm(prior_center - gt_center))
            candidate_gt_error = (
                float(np.linalg.norm(candidate_center - gt_center))
                if candidate_center is not None
                else None
            )
            anchor_stats.update({
                "gt_center_utm": [float(gt_center[0]), float(gt_center[1])],
                "gt_center_source": gt_center_source,
                "prior_center_to_gt_m": prior_gt_error,
                "candidate_center_to_gt_m": candidate_gt_error,
                "candidate_gt_diagnostic": (
                    "candidate_closer_to_gt"
                    if candidate_gt_error is not None and candidate_gt_error < prior_gt_error
                    else "prior_closer_to_gt"
                    if candidate_gt_error is not None
                    else "no_satellite_candidate"
                ),
            })
        except (KeyError, TypeError, ValueError) as exc:
            anchor_stats.update({
                "gt_center_utm": None,
                "gt_center_source": "unavailable",
                "gt_center_diagnostic_error": str(exc),
            })
        large_position_innovation = False
        large_yaw_innovation = False
        if residual_for_pose is not None and candidate_matrix is not None:
            innovation = float(np.linalg.norm(candidate_center - prior_center))
            _, residual_yaw = sim2_scale_yaw(residual)
            anchor_stats["position_innovation_m"] = innovation
            anchor_stats["yaw_innovation_deg"] = residual_yaw
            large_position_innovation = bool(
                innovation > float(ANCHOR_LARGE_POSITION_INNOVATION_M)
            )
            anchor_stats["large_position_innovation"] = large_position_innovation
            if large_position_innovation:
                anchor_stats["position_innovation_warning"] = "large_position_innovation"
            large_yaw_innovation = bool(
                apply_satellite_yaw
                and abs(residual_yaw) > float(ANCHOR_MAX_YAW_INNOVATION_DEG)
            )
            anchor_stats["large_yaw_innovation"] = large_yaw_innovation
            if (
                method_gate_enabled
                and ANCHOR_YAW_INNOVATION_HARD_REJECT_ENABLED
                and large_yaw_innovation
            ):
                if ENABLE_DUAL_ANCHOR_FUSION and not sequence_uses_legacy_bev_orbit:
                    # Innovation against a drifting prior is not evidence that a
                    # geometrically supported satellite observation is wrong.
                    # Keep it, but apply it conservatively below.
                    anchor_stats["yaw_innovation_warning"] = "large_yaw_innovation_softened"
                    anchor_stats["continuity_gate_softened_by_dual_anchor"] = True
                else:
                    anchor_accepted = False
                    anchor_stats["accepted"] = False
                    anchor_stats["reason"] = "yaw_innovation_too_large"
            if not method_gate_enabled or not apply_satellite_yaw:
                anchor_stats["continuity_gate_skipped"] = True
        elif scale_observation_only:
            anchor_stats["continuity_gate_skipped"] = True
            anchor_stats["continuity_gate_skip_reason"] = "scale_observation_only"

        extrapolation_prediction = predict_window_center_from_history(
            states,
            float(indices[center_local_idx_for_gate]),
            ANCHOR_EXTRAPOLATION_HISTORY,
            ANCHOR_EXTRAPOLATION_MIN_HISTORY,
        )
        anchor_stats["extrapolation_gate_enabled"] = bool(ANCHOR_EXTRAPOLATION_GATE_ENABLED)
        anchor_stats["extrapolation_gate_position_threshold_m"] = float(
            ANCHOR_EXTRAPOLATION_MAX_POSITION_ERROR_M
        )
        anchor_stats["extrapolation_gate_yaw_threshold_deg"] = float(
            ANCHOR_EXTRAPOLATION_MAX_YAW_ERROR_DEG
        )
        anchor_stats["extrapolation_history_required"] = int(ANCHOR_EXTRAPOLATION_MIN_HISTORY)
        if residual is not None and candidate_matrix is not None and extrapolation_prediction is not None:
            candidate_yaw = float(
                yaw_from_forward(
                    candidate_matrix,
                    geometry.camera_forward_xy[[center_local_idx_for_gate]],
                )[0]
            )
            predicted_center = np.asarray(
                extrapolation_prediction["predicted_center_utm"],
                dtype=np.float64,
            )
            predicted_yaw = float(extrapolation_prediction["predicted_yaw_deg"])
            extrap_pos_error = float(np.linalg.norm(candidate_center - predicted_center))
            extrap_yaw_error = abs(wrap_deg(candidate_yaw - predicted_yaw))
            anchor_stats.update({
                "extrapolation_gate_skipped": False,
                "extrapolation_mode": extrapolation_prediction["mode"],
                "extrapolation_history_count": int(extrapolation_prediction["history_count"]),
                "extrapolation_history_window_indices": extrapolation_prediction["history_window_indices"],
                "extrapolated_center_utm": extrapolation_prediction["predicted_center_utm"],
                "extrapolated_yaw_deg": float(predicted_yaw),
                "candidate_yaw_deg": float(candidate_yaw),
                "extrapolated_position_error_m": extrap_pos_error,
                "extrapolated_yaw_error_deg": extrap_yaw_error,
            })
            if (
                anchor_accepted
                and ANCHOR_EXTRAPOLATION_GATE_ENABLED
                and (not ENABLE_LY_DIRECT_ANCHOR_PIPELINE or sequence_uses_legacy_bev_orbit)
                and extrap_pos_error > ANCHOR_EXTRAPOLATION_MAX_POSITION_ERROR_M
            ):
                anchor_accepted = False
                anchor_stats["accepted"] = False
                anchor_stats["reason"] = "extrapolated_position_error_too_large"
            elif (
                anchor_accepted
                and ANCHOR_EXTRAPOLATION_GATE_ENABLED
                and (not ENABLE_LY_DIRECT_ANCHOR_PIPELINE or sequence_uses_legacy_bev_orbit)
                and extrap_yaw_error > ANCHOR_EXTRAPOLATION_MAX_YAW_ERROR_DEG
            ):
                anchor_accepted = False
                anchor_stats["accepted"] = False
                anchor_stats["reason"] = "extrapolated_yaw_error_too_large"
        elif residual is not None:
            anchor_stats["extrapolation_gate_skipped"] = True
            anchor_stats["extrapolation_skip_reason"] = (
                "not_enough_history" if extrapolation_prediction is None else "no_candidate_matrix"
            )

        scale_filter_has_observation = bool(
            sequence_sequential_scale_filter_enabled
            and residual is not None
            and (anchor_accepted or scale_observation_only)
        )
        if scale_filter_has_observation and not sequence_metric_scale_filter_active:
            warmup_observation = sequential_scale_warmup_observation(
                float(raw_scale),
                float(anchor_stats.get("inlier_ratio", 0.0)),
                float(anchor_stats.get("inlier_count", 0.0)),
                float(anchor_stats.get("rmse", float("nan"))),
            )
            if warmup_observation is None:
                sequence_metric_scale_warmup = []
                scale_filter_stats = {
                    "enabled": True,
                    "activated": False,
                    "reason": "warmup_anchor_not_high_quality_or_not_scale_informative",
                    "multiplier_before": float(sequence_metric_scale_multiplier),
                    "warmup_count": 0,
                }
            else:
                sequence_metric_scale_warmup.append(warmup_observation)
                sequence_metric_scale_warmup = sequence_metric_scale_warmup[
                    -int(SEQUENTIAL_ANCHOR_SCALE_WARMUP_REQUIRED):
                ]
                warmup_logs = np.asarray(
                    [item["log_scale"] for item in sequence_metric_scale_warmup],
                    dtype=np.float64,
                )
                same_direction = bool(np.all(warmup_logs > 0.0) or np.all(warmup_logs < 0.0))
                log_spread = float(np.ptp(warmup_logs)) if len(warmup_logs) > 1 else float("nan")
                ready = bool(
                    len(sequence_metric_scale_warmup) >= int(SEQUENTIAL_ANCHOR_SCALE_WARMUP_REQUIRED)
                    and same_direction
                    and log_spread <= float(SEQUENTIAL_ANCHOR_SCALE_WARMUP_MAX_LOG_SPREAD)
                )
                if ready:
                    (
                        sequence_metric_scale_multiplier,
                        sequence_metric_scale_log_variance,
                        scale_filter_stats,
                    ) = bootstrap_sequential_anchor_scale(
                        sequence_metric_scale_multiplier,
                        sequence_metric_scale_warmup,
                    )
                    sequence_metric_scale_filter_active = True
                else:
                    if len(sequence_metric_scale_warmup) >= int(SEQUENTIAL_ANCHOR_SCALE_WARMUP_REQUIRED):
                        # A contradictory third observation starts a fresh
                        # consecutive-evidence run rather than mixing regimes.
                        sequence_metric_scale_warmup = [warmup_observation]
                    scale_filter_stats = {
                        "enabled": True,
                        "activated": False,
                        "reason": "collecting_consistent_scale_warmup",
                        "multiplier_before": float(sequence_metric_scale_multiplier),
                        "warmup_count": int(len(sequence_metric_scale_warmup)),
                        "same_direction": same_direction,
                        "log_spread": log_spread,
                    }
        elif scale_filter_has_observation:
            (
                sequence_metric_scale_multiplier,
                sequence_metric_scale_log_variance,
                scale_filter_stats,
            ) = update_sequential_anchor_scale(
                sequence_metric_scale_multiplier,
                sequence_metric_scale_log_variance,
                float(raw_scale),
                float(anchor_stats.get("inlier_ratio", 0.0)),
                float(anchor_stats.get("inlier_count", 0.0)),
                float(anchor_stats.get("rmse", float("nan"))),
                enabled=sequence_sequential_scale_filter_enabled,
            )
        else:
            scale_filter_stats = {
                "enabled": bool(sequence_sequential_scale_filter_enabled),
                "reason": "no_usable_scale_observation",
                "multiplier_before": float(sequence_metric_scale_multiplier),
                "log_variance_before": float(sequence_metric_scale_log_variance),
            }
        anchor_stats["sequential_anchor_scale_update"] = scale_filter_stats
        anchor_stats["sequential_anchor_scale_filter_active"] = bool(
            sequence_metric_scale_filter_active
        )
        anchor_stats["sequential_anchor_scale_multiplier_after"] = float(
            sequence_metric_scale_multiplier
        )

        final_matrix = candidate_matrix.copy() if anchor_accepted else prior.copy()
        temporal_bev_relocalization = bool(
            anchor_accepted
            and candidate_matrix is not None
            and rescue_decision
            == "bev_rescue_temporally_confirmed_relocalization"
        )
        if temporal_bev_relocalization:
            blend = float(BEV_RESCUE_CONFIRMED_APPLY_BLEND)
            if not (0.0 < blend <= 1.0):
                raise ValueError(
                    "BEV_RESCUE_CONFIRMED_APPLY_BLEND must be in (0, 1], "
                    f"got {blend}"
                )
            final_matrix = blend_sim2_at_local_pivot(
                prior,
                candidate_matrix,
                geometry.camera_xy[center_local_idx_for_gate],
                blend,
            )
            anchor_stats["applied_anchor_mode"] = (
                "bev_temporally_confirmed_soft_blend"
            )
            anchor_stats["applied_anchor_blend"] = blend
        elif (
            anchor_accepted
            and window_idx > 0
            and (not ENABLE_LY_DIRECT_ANCHOR_PIPELINE or sequence_uses_legacy_bev_orbit)
            and (large_position_innovation or large_yaw_innovation)
        ):
            blend_candidates = [1.0]
            if large_position_innovation:
                blend_candidates.append(float(ANCHOR_LARGE_POSITION_INNOVATION_BLEND))
            if large_yaw_innovation:
                blend_candidates.append(float(ANCHOR_LARGE_YAW_INNOVATION_BLEND))
            blend = float(min(blend_candidates))
            if not (0.0 < blend <= 1.0):
                raise ValueError(
                    "Large-innovation blend must be in (0, 1], "
                    f"got {blend}"
                )
            final_matrix = blend_sim2_at_local_pivot(
                prior,
                candidate_matrix,
                geometry.camera_xy[center_local_idx_for_gate],
                blend,
            )
            anchor_stats["applied_anchor_mode"] = "large_innovation_soft_blended"
            anchor_stats["applied_anchor_blend"] = blend
        elif anchor_accepted:
            anchor_stats["applied_anchor_mode"] = "full"
            anchor_stats["applied_anchor_blend"] = 1.0
        if window_idx == 0:
            final_matrix[:, 2] = np.array([first_e, first_n]) - final_matrix[:, :2] @ geometry.camera_xy[0]
            anchor_stats["first_frame_gt_locked"] = True
        if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit:
            # LY assigns one accepted SWG transform to the complete window.  A
            # rejected window keeps the propagated transform unchanged.
            frame_matrices = np.repeat(final_matrix[None, :, :], len(indices), axis=0)
            frame_anchor_blend = np.full(
                len(indices), 1.0 if anchor_accepted else 0.0, dtype=np.float64
            )
            anchor_hold_frame_count = 0
            framewise_blend_mode = (
                "ly_full_window_swg" if anchor_accepted else "ly_relative_propagation"
            )
        else:
            anchor_hold_frame_count = 1 if window_idx == 0 else int(VGGT_WINDOW_OVERLAP)
            frame_matrices, frame_anchor_blend = interpolate_sim2_per_frame(
                prior,
                final_matrix,
                len(indices),
                anchor_hold_frame_count,
            )
            framewise_blend_mode = "linear_shared_frame_to_window_end"
        # LY uses a single SWG per window; the legacy branch keeps shared
        # overlap frames fixed and ramps the correction toward the window end.
        camera_utm = apply_framewise_sim2(frame_matrices, geometry.camera_xy)
        raw_camera_yaw = yaw_from_forward_framewise(frame_matrices, geometry.camera_forward_xy)
        if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit:
            yaw_delta = np.asarray(
                [
                    wrap_deg(float(raw) - float(base))
                    for raw, base in zip(raw_camera_yaw, uncorrected_camera_yaw)
                ],
                dtype=np.float64,
            )
            yaw_delta_applied = np.clip(
                yaw_delta,
                -float(LY_DIRECT_YAW_CORRECTION_LIMIT_DEG),
                float(LY_DIRECT_YAW_CORRECTION_LIMIT_DEG),
            )
            camera_yaw = np.asarray(
                [wrap_deg(float(base) + float(delta)) for base, delta in zip(
                    uncorrected_camera_yaw, yaw_delta_applied
                )],
                dtype=np.float64,
            )
            anchor_stats["ly_matcher_yaw_fusion"] = {
                "enabled": True,
                "baseline": "no_satellite_anchor_vggt_yaw",
                "limit_deg": float(LY_DIRECT_YAW_CORRECTION_LIMIT_DEG),
                "raw_delta_deg": yaw_delta.tolist(),
                "applied_delta_deg": yaw_delta_applied.tolist(),
                "clipped_frame_count": int(np.count_nonzero(
                    np.abs(yaw_delta - yaw_delta_applied) > 1e-9
                )),
            }
        else:
            camera_yaw = raw_camera_yaw
        local_anchor_delta = camera_utm - camera_utm_before_anchor
        anchor_stats["framewise_anchor_blend"] = {
            "mode": framewise_blend_mode,
            "start": float(frame_anchor_blend[0]),
            "end": float(frame_anchor_blend[-1]),
            "hold_frame_count": int(min(anchor_hold_frame_count, len(frame_anchor_blend))),
            "frame_count": int(len(frame_anchor_blend)),
        }
        weight = ANCHOR_ACCEPTED_WEIGHT if anchor_accepted else PROPAGATED_WINDOW_WEIGHT
        state = WindowState(
            window_idx=window_idx,
            frame_indices=indices,
            frame_names=names,
            geometry=geometry,
            local_to_utm=final_matrix,
            camera_utm=camera_utm,
            camera_yaw=camera_yaw,
            height_scale_m_per_local=window_height_scale_m_per_local,
            anchor_accepted=anchor_accepted,
            anchor_stats=anchor_stats,
        )
        states.append(state)
        if (
            ENABLE_LY_DIRECT_ANCHOR_PIPELINE
            and not sequence_uses_legacy_bev_orbit
            and anchor_accepted
        ):
            for local_idx, global_idx in enumerate(indices):
                if (
                    bev_rescue_relocalized
                    or int(global_idx) not in ly_accepted_anchor_by_frame
                ):
                    ly_accepted_anchor_by_frame[int(global_idx)] = np.asarray(
                        camera_utm[local_idx], dtype=np.float64
                    ).copy()
        if anchor_accepted:
            consecutive_anchor_rejections = 0
        else:
            consecutive_anchor_rejections += 1
        anchor_stats["bev_rescue"]["consecutive_rejections_after"] = int(
            consecutive_anchor_rejections
        )
        uncorrected_state = WindowState(
            window_idx=window_idx,
            frame_indices=indices,
            frame_names=names,
            geometry=geometry,
            local_to_utm=uncorrected_matrix,
            camera_utm=uncorrected_camera_utm,
            camera_yaw=uncorrected_camera_yaw,
            height_scale_m_per_local=uncorrected_window_height_scale_m_per_local,
            anchor_accepted=False,
            anchor_stats={"source": "no_satellite_anchor_baseline"},
        )
        uncorrected_states.append(uncorrected_state)
        record_stage_timing(
            stage_timings, "anchor_fusion_gating_and_pose", stage_started,
            window_timing_row,
        )
        stage_started = time.perf_counter()
        point_cloud_summary: dict[str, Any] = {"enabled": bool(save_window_point_clouds)}
        if save_window_point_clouds:
            local_path = window_dir / "pointcloud_local.ply"
            utm_path = window_dir / "pointcloud_utm.ply"
            local_count = save_rgb_point_cloud_ply(
                local_path,
                prediction.points,
                prediction.colors_rgb,
            )
            final_scale, _ = sim2_scale_yaw(final_matrix)
            point_utm_xy = apply_sim2(final_matrix, geometry.point_xy)
            point_utm_xyz = np.column_stack([
                point_utm_xy,
                np.asarray(geometry.point_height, dtype=np.float64)
                * float(window_height_scale_m_per_local),
            ])
            utm_count = save_rgb_point_cloud_ply(
                utm_path,
                point_utm_xyz,
                prediction.colors_rgb,
            )
            point_cloud_summary.update({
                "local_ply": str(local_path),
                "utm_ply": str(utm_path),
                "local_point_count": int(local_count),
                "utm_point_count": int(utm_count),
                "utm_axes": "x=easting_m, y=northing_m, z=height_above_fitted_ground_m",
                "utm_horizontal_sim2_scale_m_per_local": float(final_scale),
                "utm_height_scale_m_per_local": float(window_height_scale_m_per_local),
                "height_scale_source": str(height_scale_stats["source"]),
            })
            if not compact_mode:
                print(
                    f"[WINDOW {window_idx:04d}] point clouds saved: "
                    f"local={local_count}, utm={utm_count}"
                )
        record_stage_timing(
            stage_timings, "window_pointcloud_io", stage_started, window_timing_row
        )
        stage_started = time.perf_counter()
        predicted_window_ypr = vggt_window_ypr(
            prediction.extrinsics,
            geometry,
            frame_matrices,
        )
        predicted_window_height_m = (
            np.abs(np.asarray(geometry.camera_height, dtype=np.float64))
            * float(window_height_scale_m_per_local)
        )
        attitude_csv_path = window_dir / "window_gt_vggt_attitude.csv"
        attitude_row_count = 0
        if save_intermediate_outputs:
            attitude_row_count = save_window_attitude_csv(
                attitude_csv_path,
                gt_df,
                indices,
                names,
                predicted_window_ypr,
                predicted_window_height_m,
            )
            print(
                f"[WINDOW {window_idx:04d}] GT/VGGT attitude saved: "
                f"{attitude_csv_path} ({attitude_row_count} rows)"
            )
        intrinsics_csv_path = window_dir / "window_vggt_intrinsics.csv"
        intrinsics_row_count = 0
        if save_window_vggt_intrinsics:
            intrinsics_row_count = save_window_vggt_intrinsics_csv(
                intrinsics_csv_path,
                paths,
                names,
                prediction.intrinsics,
                prediction.images_hw,
                VGGT_TARGET_SIZE,
            )
            if not compact_mode:
                print(
                    f"[WINDOW {window_idx:04d}] VGGT intrinsics saved: "
                    f"{intrinsics_csv_path} ({intrinsics_row_count} rows)"
                )
        for local_idx, global_idx in enumerate(indices):
            frame_observations.setdefault(global_idx, []).append({
                "e": float(camera_utm[local_idx, 0]),
                "n": float(camera_utm[local_idx, 1]),
                "yaw": float(camera_yaw[local_idx]),
                "weight": float(weight),
                "window_idx": int(window_idx),
                "anchor_accepted": bool(anchor_accepted),
            })
            uncorrected_frame_observations.setdefault(global_idx, []).append({
                "e": float(uncorrected_camera_utm[local_idx, 0]),
                "n": float(uncorrected_camera_utm[local_idx, 1]),
                "yaw": float(uncorrected_camera_yaw[local_idx]),
                "weight": 1.0,
                "window_idx": int(window_idx),
                "anchor_accepted": False,
            })
        summary = {
            "window_idx": int(window_idx),
            "pipeline_branch": (
                "orbit_legacy_bev" if sequence_uses_legacy_bev_orbit else "ly_direct_with_bev_rescue"
            ),
            "orbit_motion_detector": orbit_motion_diagnostics,
            "anchor_match_method": anchor_match_method,
            "start_idx": int(start),
            "end_idx": int(end),
            "frame_names": names,
            "prior": prior_stats,
            "ground_plane": plane_stats,
            "bev": bev_meta,
            "anchor": anchor_stats,
            "anchor_accepted": bool(anchor_accepted),
            "height_scale": {
                "m_per_local": float(window_height_scale_m_per_local),
                "source": str(height_scale_stats["source"]),
                "propagation": height_scale_stats,
                "horizontal_sim2_scale_m_per_local": float(sim2_scale_yaw(final_matrix)[0]),
            },
            "point_clouds": point_cloud_summary,
            "attitude_comparison": {
                "csv": str(attitude_csv_path) if save_intermediate_outputs else None,
                "row_count": int(attitude_row_count),
                "columns": [
                    "frame",
                    "gt_yaw_deg", "gt_pitch_deg", "gt_roll_deg",
                    "gt_rel_height_m", "gt_abs_height_m",
                    "vggt_yaw_deg", "vggt_pitch_deg", "vggt_roll_deg",
                    "vggt_height_m",
                ],
                "vggt_orientation_frame": "body_to_ENU_then_NED_ZYX_in_GT_csv_convention",
            },
            "vggt_intrinsics": {
                "enabled": bool(save_window_vggt_intrinsics),
                "csv": str(intrinsics_csv_path) if save_window_vggt_intrinsics else None,
                "row_count": int(intrinsics_row_count),
                "model_input_hw": [int(prediction.images_hw[0]), int(prediction.images_hw[1])],
                "coordinate_note": (
                    "*_input_px is VGGT's padded model coordinate system; "
                    "*_source_px undoes the resize/pad transform and is still a VGGT estimate, not true K."
                ),
                "fixed_source_intrinsics_enabled": bool(VGGT_FIXED_SOURCE_INTRINSICS_ENABLED),
                "fixed_k_track_ba_enabled": bool(VGGT_FIXED_K_TRACK_BA_ENABLED),
                "fixed_source_intrinsics": (
                    VGGT_FIXED_SOURCE_INTRINSICS
                    if VGGT_FIXED_SOURCE_INTRINSICS_ENABLED else None
                ),
                "ray_correction": prediction.stats.get("fixed_intrinsics_ray_correction"),
                "track_ba": prediction.stats.get("fixed_k_track_ba"),
            },
            "local_to_utm_before_local_anchor": prior.tolist(),
            "local_to_utm_sim2": final_matrix.tolist(),
            "no_satellite_anchor_local_to_utm_sim2": uncorrected_matrix.tolist(),
            "camera_utm_before_local_anchor": camera_utm_before_anchor.tolist(),
            "camera_utm": camera_utm.tolist(),
            "camera_correction_delta_m": local_anchor_delta.tolist(),
            "camera_correction_norm_m": np.linalg.norm(local_anchor_delta, axis=1).tolist(),
            "camera_yaw_before_local_anchor_deg": camera_yaw_before_anchor.tolist(),
            "camera_yaw_deg": camera_yaw.tolist(),
            "baseline_no_satellite_anchor_prior": uncorrected_prior_stats,
            "baseline_no_satellite_anchor_camera_utm": uncorrected_camera_utm.tolist(),
            "baseline_no_satellite_anchor_camera_yaw_deg": uncorrected_camera_yaw.tolist(),
        }
        window_summaries.append(summary)
        center_local_idx = len(indices) // 2
        visualization_anchors.append({
            "ok": bool(anchor_accepted),
            "frame": names[center_local_idx],
            "pred_utm": camera_utm_before_anchor[center_local_idx].tolist(),
            "corrected_utm": camera_utm[center_local_idx].tolist(),
            "yaw_delta_deg": wrap_deg(
                float(camera_yaw[center_local_idx]) - float(camera_yaw_before_anchor[center_local_idx])
            ),
            "inlier_count": int(anchor_stats.get("inlier_count", 0)),
            "inlier_ratio": float(anchor_stats.get("inlier_ratio", 0.0)),
            "rmse_m": anchor_stats.get("rmse"),
            "window_idx": int(window_idx),
            "position_delta_m": local_anchor_delta[center_local_idx].tolist(),
        })
        if save_intermediate_outputs:
            with open(window_dir / "window_result.json", "w", encoding="utf-8") as file:
                json.dump(summary, file, indent=2, ensure_ascii=False)
        center_delta = local_anchor_delta[len(indices) // 2]
        center_yaw_delta = wrap_deg(
            float(camera_yaw[len(indices) // 2]) - float(camera_yaw_before_anchor[len(indices) // 2])
        )
        if not compact_mode:
            print(
                f"[WINDOW {window_idx:04d}] anchor={'accepted' if anchor_accepted else 'propagated'} "
                f"dE={center_delta[0]:.2f}m dN={center_delta[1]:.2f}m "
                f"dYaw={center_yaw_delta:.2f}deg "
                f"inliers={anchor_stats.get('inlier_count', 0)} rmse={anchor_stats.get('rmse')} scale={anchor_stats.get('scale')}"
            )
        record_stage_timing(
            stage_timings, "window_outputs_and_bookkeeping", stage_started,
            window_timing_row,
        )
        stage_started = time.perf_counter()
        del prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()
        record_stage_timing(
            stage_timings, "cuda_cache_cleanup", stage_started,
            window_timing_row,
        )
        window_timing_row["window_total_sec"] = float(
            time.perf_counter() - window_started
        )
        window_timing_rows.append(window_timing_row)

    stage_started = time.perf_counter()
    sequence_metric_scale_calibration = estimate_sequence_metric_scale(window_summaries)
    sequence_metric_scale_calibration.update({
        "current_multiplier": float(sequence_metric_scale_multiplier),
        "per_window_satellite_scale_enabled": bool(
            ANCHOR_APPLY_SATELLITE_SCALE and not sequence_uses_legacy_bev_orbit
        ),
        "second_pass_recipe": {
            "SEQUENCE_METRIC_SCALE_MULTIPLIER": float(
                sequence_metric_scale_calibration.get("recommended_multiplier", 1.0)
            ),
            "ANCHOR_APPLY_SATELLITE_SCALE": False,
        },
    })
    with open(save_dir / "sequence_metric_scale_calibration.json", "w", encoding="utf-8") as file:
        json.dump(sequence_metric_scale_calibration, file, indent=2, ensure_ascii=False)
    if not compact_mode:
        print(
            "[SCALE] sequence recommendation="
            f"{sequence_metric_scale_calibration.get('recommended_multiplier', 1.0):.5f} "
            f"from {sequence_metric_scale_calibration.get('kept_sample_count', 0)}/"
            f"{sequence_metric_scale_calibration.get('sample_count', 0)} accepted anchors"
        )
    record_stage_timing(stage_timings, "sequence_scale_summary", stage_started)

    stage_started = time.perf_counter()
    optimized_frame_count = max(frame_observations.keys()) + 1 if frame_observations else 0
    optimized_xy, pose_graph_summary = optimize_global_positions(
        states,
        optimized_frame_count,
        np.array([first_e, first_n], dtype=np.float64),
        odometry_states=uncorrected_states,
    )
    uncorrected_frame_count = max(uncorrected_frame_observations.keys()) + 1 if uncorrected_frame_observations else 0
    uncorrected_optimized_xy, uncorrected_pose_graph_summary = optimize_global_positions(
        uncorrected_states,
        uncorrected_frame_count,
        np.array([first_e, first_n], dtype=np.float64),
        odometry_states=uncorrected_states,
    )
    record_stage_timing(stage_timings, "global_position_optimization", stage_started)
    stage_started = time.perf_counter()
    trajectory: dict[str, Any] = {}
    uncorrected_trajectory: dict[str, Any] = {}
    for idx, name in enumerate(frame_names):
        observations = frame_observations.get(idx, [])
        if not observations:
            continue
        weights = np.asarray([o["weight"] for o in observations], dtype=np.float64)
        weights /= max(float(weights.sum()), 1e-12)
        if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit:
            # LY aggregates repeated overlap predictions with a coordinate-wise
            # median, which is robust to one disagreeing window.
            e = float(np.median(np.asarray([o["e"] for o in observations])))
            n = float(np.median(np.asarray([o["n"] for o in observations])))
        else:
            e = float(np.sum(weights * np.asarray([o["e"] for o in observations])))
            n = float(np.sum(weights * np.asarray([o["n"] for o in observations])))
        if optimized_xy is not None:
            e = float(optimized_xy[idx, 0])
            n = float(optimized_xy[idx, 1])
        yaw_values = [o["yaw"] for o in observations if np.isfinite(o["yaw"])]
        yaw_weights = [o["weight"] for o in observations if np.isfinite(o["yaw"])]
        yaw = circular_weighted_mean(yaw_values, yaw_weights) if yaw_values else 0.0
        if idx == 0:
            e, n, yaw = float(first_e), float(first_n), float(first_gt["yaw"])
        trajectory[name] = {
            "Translation_UTM": [e, n, float(first_gt["rel_height"])],
            "Yaw_deg": float(yaw),
            "Pitch_deg": float(first_gt["pitch"]) if idx == 0 else 0.0,
            "Roll_deg": float(first_gt["roll"]) if idx == 0 else 0.0,
            "Height": float(first_gt["rel_height"]),
            "VGGT_Window_Observations": observations,
        }
        uncorrected_observations = uncorrected_frame_observations.get(idx, [])
        if uncorrected_observations:
            un_weights = np.asarray([o["weight"] for o in uncorrected_observations], dtype=np.float64)
            un_weights /= max(float(un_weights.sum()), 1e-12)
            un_e = float(np.sum(un_weights * np.asarray([o["e"] for o in uncorrected_observations])))
            un_n = float(np.sum(un_weights * np.asarray([o["n"] for o in uncorrected_observations])))
            if uncorrected_optimized_xy is not None:
                un_e = float(uncorrected_optimized_xy[idx, 0])
                un_n = float(uncorrected_optimized_xy[idx, 1])
            un_yaw_values = [o["yaw"] for o in uncorrected_observations if np.isfinite(o["yaw"])]
            un_yaw_weights = [o["weight"] for o in uncorrected_observations if np.isfinite(o["yaw"])]
            un_yaw = circular_weighted_mean(un_yaw_values, un_yaw_weights) if un_yaw_values else 0.0
            if idx == 0:
                un_e, un_n, un_yaw = float(first_e), float(first_n), float(first_gt["yaw"])
            uncorrected_trajectory[name] = {
                "Translation_UTM": [un_e, un_n, float(first_gt["rel_height"])],
                "Yaw_deg": float(un_yaw),
                "Pitch_deg": float(first_gt["pitch"]) if idx == 0 else 0.0,
                "Roll_deg": float(first_gt["roll"]) if idx == 0 else 0.0,
                "Height": float(first_gt["rel_height"]),
                "VGGT_Window_Observations": uncorrected_observations,
            }
    trajectory["_vggt_window_minima"] = {
        "model_load_count": 1,
        "experiment_compact_mode": bool(compact_mode),
        "save_all_visualizations": bool(save_all_visualizations),
        "anchor_match_method": anchor_match_method,
        "window_size": int(VGGT_WINDOW_SIZE),
        "window_overlap": int(VGGT_WINDOW_OVERLAP),
        "window_stride": int(VGGT_WINDOW_STRIDE),
        "window_stitch_mode": int(VGGT_WINDOW_STITCH_MODE),
        "dense_window_overlap_enabled": bool(ENABLE_DENSE_WINDOW_OVERLAP),
        "window_count": int(len(window_summaries)),
        "accepted_anchor_count": int(sum(bool(w["anchor_accepted"]) for w in window_summaries)),
        "first_frame_gt_locked": True,
        "orbit_legacy_bev_state_machine": {
            "enabled": bool(ENABLE_ORBIT_LEGACY_BEV_PIPELINE),
            "active": bool(sequence_uses_legacy_bev_orbit),
            "decision_window": 0,
            "detector": orbit_motion_diagnostics,
            "active_policy": (
                "copy34_per_window_bev_causal_scale_and_framewise_ramp"
                if sequence_uses_legacy_bev_orbit
                else "ly_direct_with_conditional_bev_rescue"
            ),
        },
        "ly_direct_pipeline": {
            "enabled": bool(
                ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit
            ),
            "strict_vggt_surface_profile": bool(LY_STRICT_VGGT_SURFACE_PROFILE),
            "matcher": "minima",
            "full_window_swg": bool(
                ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not sequence_uses_legacy_bev_orbit
            ),
            "accepted_anchor_map_policy": "first_write_wins",
            "overlap_aggregation": "coordinate_median",
            "raycast_query": "zbuffer_exact_projected_pixel_distance",
            "yaw_fusion_limit_deg": float(LY_DIRECT_YAW_CORRECTION_LIMIT_DEG),
            "vggt_surface_compatibility_profile": {
                "confidence_percentile": float(VGGT_CONF_PERCENTILE),
                "use_depth_unproject": bool(VGGT_USE_DEPTH_UNPROJECT),
                "fixed_k_track_ba": bool(VGGT_FIXED_K_TRACK_BA_ENABLED),
                "reason": "validated_better_than_strict_ly_vggt_surface_on_current_thermal_dataset",
            },
        },
        "dual_anchor": {
            "bev_enabled": bool(ENABLE_BEV_ANCHOR),
            "bev_rescue_mode_enabled": bool(ENABLE_BEV_RESCUE_MODE),
            "bev_rescue_lost_after_rejections": int(
                BEV_RESCUE_LOST_AFTER_REJECTIONS
            ),
            "bev_rescue_extra_margin_step_m": float(
                BEV_RESCUE_EXTRA_MARGIN_STEP_M
            ),
            "bev_rescue_max_extra_margin_m": float(
                BEV_RESCUE_MAX_EXTRA_MARGIN_M
            ),
            "bev_rescue_agreement_position_m": float(
                BEV_RESCUE_AGREEMENT_POSITION_M
            ),
            "bev_rescue_agreement_yaw_deg": float(
                BEV_RESCUE_AGREEMENT_YAW_DEG
            ),
            "bev_rescue_agreement_scale_log": float(
                BEV_RESCUE_AGREEMENT_SCALE_LOG
            ),
            "bev_rescue_temporal_confirmation_enabled": bool(
                ENABLE_BEV_RESCUE_TEMPORAL_CONFIRMATION
            ),
            "bev_rescue_confirm_min_overlap_frames": int(
                BEV_RESCUE_CONFIRM_MIN_OVERLAP_FRAMES
            ),
            "bev_rescue_confirm_position_m": float(
                BEV_RESCUE_CONFIRM_POSITION_M
            ),
            "bev_rescue_confirm_yaw_deg": float(BEV_RESCUE_CONFIRM_YAW_DEG),
            "bev_rescue_confirm_scale_log": float(
                BEV_RESCUE_CONFIRM_SCALE_LOG
            ),
            "bev_rescue_pending_max_age_windows": int(
                BEV_RESCUE_PENDING_MAX_AGE_WINDOWS
            ),
            "bev_rescue_confirmed_apply_blend": float(
                BEV_RESCUE_CONFIRMED_APPLY_BLEND
            ),
            "bev_guided_ly_retry_enabled": bool(ENABLE_BEV_GUIDED_LY_RETRY),
            "bev_guided_ly_use_template_coarse": bool(
                BEV_GUIDED_LY_USE_TEMPLATE_COARSE
            ),
            "bev_guided_ly_extra_margin_m": float(BEV_GUIDED_LY_EXTRA_MARGIN_M),
            "bev_guided_ly_min_inliers": int(BEV_GUIDED_LY_MIN_INLIERS),
            "bev_guided_ly_min_inlier_ratio": float(BEV_GUIDED_LY_MIN_INLIER_RATIO),
            "bev_guided_ly_max_rmse_m": float(BEV_GUIDED_LY_MAX_RMSE_M),
            "bev_guided_ly_max_scale_log": float(BEV_GUIDED_LY_MAX_SCALE_LOG),
            "multiframe_3d_utm_enabled": bool(ENABLE_MULTIFRAME_3D_UTM_ANCHOR),
            "fusion_enabled": bool(ENABLE_DUAL_ANCHOR_FUSION),
            "multiframe_keyframe_count": int(MULTIFRAME_KEYFRAME_COUNT),
            "multiframe_satellite_crop_strategy": "ly_fixed_pixel_tile",
            "multiframe_satellite_tile_size_px": int(MULTIFRAME_SATELLITE_TILE_SIZE_PX),
            "multiframe_rotation_context_ratio": float(MULTIFRAME_ROTATION_CONTEXT_RATIO),
            "agreement_position_m": float(DUAL_ANCHOR_AGREEMENT_POSITION_M),
            "agreement_yaw_deg": float(DUAL_ANCHOR_AGREEMENT_YAW_DEG),
            "agreement_scale_log": float(DUAL_ANCHOR_AGREEMENT_SCALE_LOG),
            "application_policy": "independent_candidates_fused_then_applied_once",
        },
        "sequential_anchor_scale_filter": {
            "enabled": bool(sequence_sequential_scale_filter_enabled),
            "active": bool(sequence_metric_scale_filter_active),
            "final_multiplier": float(sequence_metric_scale_multiplier),
            "final_log_variance": float(sequence_metric_scale_log_variance),
            "initial_std_log": float(SEQUENTIAL_ANCHOR_SCALE_INITIAL_STD_LOG),
            "warmup_required": int(SEQUENTIAL_ANCHOR_SCALE_WARMUP_REQUIRED),
            "current_window_direct_satellite_scale_enabled": bool(
                ANCHOR_APPLY_SATELLITE_SCALE and not sequence_uses_legacy_bev_orbit
            ),
        },
        "sequence_metric_scale_calibration": sequence_metric_scale_calibration,
        "local_anchor_policy": "apply_each_accepted_anchor_to_the_current_window_before_next_window",
        "pose_graph": pose_graph_summary,
        "window_summaries": window_summaries,
    }
    trajectory["_segment_minima_se2"] = {
        "mode": "vggt_window_local_sim2_anchor_visualization",
        "anchor_match_method": anchor_match_method,
        "anchors": visualization_anchors,
        "anchor_count": int(len(visualization_anchors)),
        "applied_count": int(sum(bool(a["ok"]) for a in visualization_anchors)),
    }
    uncorrected_trajectory["_vggt_window_no_satellite_anchor"] = {
        "model_load_count": 1,
        "window_size": int(VGGT_WINDOW_SIZE),
        "window_overlap": int(VGGT_WINDOW_OVERLAP),
        "window_stride": int(VGGT_WINDOW_STRIDE),
        "window_stitch_mode": int(VGGT_WINDOW_STITCH_MODE),
        "window_count": int(len(window_summaries)),
        "satellite_anchor_count": 0,
        "first_frame_gt_locked": True,
        "policy": "first_frame_gt_then_overlap_sim2_only_without_any_satellite_anchor",
        "pose_graph": uncorrected_pose_graph_summary,
    }
    record_stage_timing(stage_timings, "trajectory_aggregation", stage_started)
    stage_started = time.perf_counter()
    trajectory_path = save_dir / "global_poses_utm.json"
    uncorrected_trajectory_path = save_dir / "global_poses_utm_uncorrected.json"
    anchor_diagnostics_path = save_dir / "anchor_diagnostics.csv"
    with open(trajectory_path, "w", encoding="utf-8") as file:
        json.dump(trajectory, file, indent=2, ensure_ascii=False)
    with open(save_dir / "window_anchor_summary.json", "w", encoding="utf-8") as file:
        json.dump(trajectory["_vggt_window_minima"], file, indent=2, ensure_ascii=False)
    anchor_diagnostics_rows = save_anchor_diagnostics_csv(window_summaries, anchor_diagnostics_path)
    with open(uncorrected_trajectory_path, "w", encoding="utf-8") as file:
        json.dump(uncorrected_trajectory, file, indent=2, ensure_ascii=False)
    record_stage_timing(stage_timings, "trajectory_and_anchor_serialization", stage_started)

    stage_started = time.perf_counter()
    merged_pointcloud_path = None
    merged_pointcloud_count = 0
    if save_window_point_clouds:
        utm_paths = [
            Path(summary["point_clouds"]["utm_ply"])
            for summary in window_summaries
            if summary.get("point_clouds", {}).get("utm_ply")
        ]
        if utm_paths:
            # Put the merged cloud at the sequence root, beside windows/.
            merged_pointcloud_path = save_dir / "pointcloud_utm_all_windows.ply"
            merged_pointcloud_count = merge_utm_point_clouds_without_alignment(
                utm_paths, merged_pointcloud_path
            )
            if not compact_mode:
                print(f"  merged UTM point cloud: {merged_pointcloud_path} ({merged_pointcloud_count} points)")
    record_stage_timing(stage_timings, "merged_pointcloud_io", stage_started)

    stage_started = time.perf_counter()
    if save_all_visualizations:
        legacy.visualize_comparison_on_map(
            global_trajectory=trajectory,
            csv_path=str(csv_path),
            region=REGION_NAME,
            opt=opt,
            region_config=region_config,
            ref_map_full=ref_map_full,
            out_dir=str(save_dir),
            crop_size_m=2500,
            stream_window=VGGT_WINDOW_SIZE,
            stream_stride=VGGT_WINDOW_STRIDE,
            color_pred_by_segment=False,
            draw_trajectory_lines=True,
            draw_trajectory_arrows=True,
            uncorrected_trajectory=uncorrected_trajectory,
            draw_uncorrected_trajectory_points=True,
        )
    record_stage_timing(stage_timings, "final_trajectory_visualization", stage_started)
    stage_started = time.perf_counter()
    uncorrected_metrics = legacy.compute_and_save_trajectory_metrics(
        global_trajectory=uncorrected_trajectory,
        csv_path=str(csv_path),
        region=REGION_NAME,
        region_config=region_config,
        out_dir=str(save_dir),
        runtime_sec=time.perf_counter() - run_start,
        output_name="trajectory_metrics_uncorrected.pkl",
        label="vggt_window_overlap_only_no_satellite_anchor",
    )
    corrected_metrics = legacy.compute_and_save_trajectory_metrics(
        global_trajectory=trajectory,
        csv_path=str(csv_path),
        region=REGION_NAME,
        region_config=region_config,
        out_dir=str(save_dir),
        runtime_sec=time.perf_counter() - run_start,
        output_name="trajectory_metrics.pkl",
        label="vggt_window_minima_local_anchor",
    )
    pitch_by_frame = collect_window_pitch_comparison(window_summaries)
    summary_csv_path, per_frame_csv_path = save_comparison_csv_reports(
        uncorrected_metrics,
        corrected_metrics,
        save_dir,
        pitch_by_frame=pitch_by_frame,
    )
    record_stage_timing(stage_timings, "metrics_and_comparison_reports", stage_started)
    sequence_total_sec = float(time.perf_counter() - run_start)
    timing_rows = build_stage_timing_rows(
        stage_timings,
        sequence_total_sec,
        len(window_summaries),
        len(frame_names),
    )
    timing_summary_path = save_dir / "runtime_stage_breakdown.csv"
    timing_window_path = save_dir / "runtime_window_breakdown.csv"
    pd.DataFrame(timing_rows).to_csv(
        timing_summary_path, index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(window_timing_rows).to_csv(
        timing_window_path, index=False, encoding="utf-8-sig"
    )
    if not compact_mode:
        print_metrics_comparison(uncorrected_metrics, corrected_metrics)
        print(f"\n[Done] VGGT window MINIMA pipeline: {save_dir}")
        print(f"  trajectory: {trajectory_path}")
        print(f"  no anchor:  {uncorrected_trajectory_path}")
        print(f"  anchors:    {save_dir / 'window_anchor_summary.json'}")
        print(f"  anchor csv: {anchor_diagnostics_path} ({anchor_diagnostics_rows} rows)")
        print(f"  compare:    {save_dir / 'trajectory_compare_utm.jpg'}")
        print(f"  summary CSV:{summary_csv_path}")
        print(f"  frame CSV:  {per_frame_csv_path}")
        print(f"  timing:     {timing_summary_path}")
    return {
        "dataset_name": dataset_name,
        "save_dir": str(save_dir),
        "uncorrected_metrics": uncorrected_metrics,
        "corrected_metrics": corrected_metrics,
        "summary_csv": str(summary_csv_path),
        "per_frame_csv": str(per_frame_csv_path),
        "anchor_diagnostics_csv": str(anchor_diagnostics_path),
        "merged_utm_pointcloud": str(merged_pointcloud_path) if merged_pointcloud_path else None,
        "merged_utm_point_count": int(merged_pointcloud_count),
        "window_count": int(len(window_summaries)),
        "accepted_anchor_count": int(sum(bool(w["anchor_accepted"]) for w in window_summaries)),
        "frame_count": int(len(frame_names)),
        "sequence_metric_scale_calibration": sequence_metric_scale_calibration,
        "runtime_sec": sequence_total_sec,
        "timing_rows": timing_rows,
        "window_timing_rows": window_timing_rows,
        "timing_summary_csv": str(timing_summary_path),
        "timing_window_csv": str(timing_window_path),
    }


def run_batch() -> Path:
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = ROOT / "Result" / f"VGGT_Window_MINIMA_BATCH_{batch_timestamp}"
    batch_root.mkdir(parents=True, exist_ok=True)
    dataset_names = discover_batch_datasets()
    with open(batch_root / "batch_datasets.json", "w", encoding="utf-8") as file:
        json.dump(dataset_names, file, indent=2, ensure_ascii=False)
    if not BATCH_EXPERIMENT_COMPACT_MODE:
        print(f"[BATCH] output root: {batch_root}")
        print(f"[BATCH] datasets: {len(dataset_names)}")

    runtime = create_runtime_context(batch_root)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    batch_progress = tqdm(
        dataset_names,
        desc="VGGT batch",
        unit="sequence",
        disable=not BATCH_EXPERIMENT_COMPACT_MODE,
    )
    for dataset_name in batch_progress:
        try:
            result = run_pipeline(dataset_name, runtime, batch_root)
            results.append(result)
            mean_error = float(
                result["corrected_metrics"].get("position", {}).get("error_mean_m", float("nan"))
            )
            if BATCH_EXPERIMENT_COMPACT_MODE:
                batch_progress.set_postfix_str(f"mean={mean_error:.3f} m")
                tqdm.write(f"[{dataset_name}] corrected mean error: {mean_error:.3f} m")
        except Exception as exc:
            if BATCH_EXPERIMENT_COMPACT_MODE:
                tqdm.write(f"[BATCH][FAIL] {dataset_name}: {exc}")
            else:
                print(f"[BATCH][FAIL] {dataset_name}: {exc}")
            failures.append({"dataset_name": dataset_name, "error": repr(exc)})
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_batch_reports(batch_root, results, failures)
    return batch_root


pipeline_start = time.perf_counter()
