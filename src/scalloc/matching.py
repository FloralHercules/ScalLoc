"""Matching from the validated fast ScalLoc pipeline."""
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
from .geometry import (GroundGeometry, apply_sim2, circular_weighted_mean, compose_sim2, invert_sim2, sim2_from_pivot, sim2_scale_yaw, similarity_ransac, similarity_ransac_anchored_like_ly, wrap_deg, yaw_from_forward)
from .reporting import (_vggt_pad_preprocess_affine)
from .runtime_utils import (load_module_from_path, template_match_optimizer_path)


def _multiframe_keyframe_indices(frame_count: int) -> list[int]:
    if int(MULTIFRAME_KEYFRAME_COUNT) <= 0:
        return list(range(max(0, int(frame_count))))
    count = min(max(1, int(MULTIFRAME_KEYFRAME_COUNT)), max(1, int(frame_count)))
    return sorted({int(round(value)) for value in np.linspace(0, frame_count - 1, count)})


def _source_pixels_to_vggt(
    points_px: np.ndarray,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    sx, sy, pad_left, pad_top, _rw, _rh = _vggt_pad_preprocess_affine(
        int(source_width), int(source_height), int(VGGT_TARGET_SIZE)
    )
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([
        points[:, 0] * float(sx) + float(pad_left),
        points[:, 1] * float(sy) + float(pad_top),
    ])


def _dual_grid_balanced_indices(
    uav_px: np.ndarray,
    sat_px: np.ndarray,
    confidence: np.ndarray,
    uav_shape: tuple[int, int],
    sat_shape: tuple[int, int],
) -> np.ndarray:
    """Keep confident matches while preventing one texture patch dominating a frame."""
    uav = np.asarray(uav_px, dtype=np.float64).reshape(-1, 2)
    sat = np.asarray(sat_px, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if len(uav) == 0:
        return np.zeros((0,), dtype=np.int64)
    grid = max(1, int(MULTIFRAME_GRID_SIZE))
    uh, uw = max(1, int(uav_shape[0])), max(1, int(uav_shape[1]))
    sh, sw = max(1, int(sat_shape[0])), max(1, int(sat_shape[1]))
    ugx = np.clip((uav[:, 0] / uw * grid).astype(np.int32), 0, grid - 1)
    ugy = np.clip((uav[:, 1] / uh * grid).astype(np.int32), 0, grid - 1)
    sgx = np.clip((sat[:, 0] / sw * grid).astype(np.int32), 0, grid - 1)
    sgy = np.clip((sat[:, 1] / sh * grid).astype(np.int32), 0, grid - 1)
    uav_grid = ugy * grid + ugx
    sat_grid = sgy * grid + sgx
    order = np.argsort(-np.where(np.isfinite(conf), conf, -np.inf))
    uav_counts = np.zeros(grid * grid, dtype=np.int32)
    sat_counts = np.zeros(grid * grid, dtype=np.int32)
    selected: list[int] = []
    for idx in order:
        uav_id = int(uav_grid[idx])
        sat_id = int(sat_grid[idx])
        if (
            uav_counts[uav_id] >= int(MULTIFRAME_MAX_MATCHES_PER_GRID_PAIR)
            or sat_counts[sat_id] >= int(MULTIFRAME_MAX_MATCHES_PER_GRID_PAIR)
        ):
            continue
        uav_counts[uav_id] += 1
        sat_counts[sat_id] += 1
        selected.append(int(idx))
        if len(selected) >= int(MULTIFRAME_MAX_CORRESPONDENCES_PER_FRAME):
            break
    return np.asarray(selected, dtype=np.int64)


def _grid_ids_for_points(
    points_px: np.ndarray,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    height, width = max(1, int(shape_hw[0])), max(1, int(shape_hw[1]))
    grid = max(1, int(MULTIFRAME_GRID_SIZE))
    gx = np.floor(np.clip(points[:, 0], 0.0, width - 1.0) / width * grid).astype(np.int32)
    gy = np.floor(np.clip(points[:, 1], 0.0, height - 1.0) / height * grid).astype(np.int32)
    return np.clip(gy, 0, grid - 1) * grid + np.clip(gx, 0, grid - 1)


def _candidate_quality_score(stats: dict[str, Any], multiframe: bool = False) -> float:
    if not bool(stats.get("accepted", False)):
        return 0.0
    count = max(0.0, float(stats.get("inlier_count", 0.0)))
    ratio = max(0.0, float(stats.get("inlier_ratio", 0.0)))
    rmse = float(stats.get("rmse", stats.get("rmse_m", float("inf"))))
    if not np.isfinite(rmse):
        return 0.0
    count_score = min(1.0, math.log1p(count) / math.log1p(500.0))
    ratio_score = min(1.0, ratio / 0.20)
    rmse_score = math.exp(-rmse / 8.0)
    score = 0.35 * count_score + 0.35 * ratio_score + 0.30 * rmse_score
    if multiframe:
        required_frame_count = max(
            len(stats.get("selected_local_indices", [])), 1
        )
        frame_score = min(
            1.0,
            float(stats.get("inlier_frame_count", 0.0))
            / float(required_frame_count),
        )
        score *= 0.65 + 0.35 * frame_score
    return float(np.clip(score, 1e-6, 1.0))


def crop_satellite_tile_at_utm_like_ly(
    ref_map_full: np.ndarray,
    region_config: dict[str, Any],
    center_e: float,
    center_n: float,
    tile_size_px: int,
) -> tuple[np.ndarray, float, float, float, bool, dict[str, Any]]:
    """Fixed-pixel satellite crop used by ly/ObliqueLoc's anchor branch."""
    resolution = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_resolution"])
    init_e = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_initialX"])
    init_n = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_initialY"])
    map_h, map_w = ref_map_full.shape[:2]
    size = int(tile_size_px)
    if size <= 0:
        raise ValueError(f"tile_size_px must be positive, got {size}")
    if size > map_w or size > map_h:
        raise ValueError(
            f"tile_size_px={size} exceeds satellite map size {map_w}x{map_h}"
        )

    center_x = (float(center_e) - init_e) / resolution
    center_y = (init_n - float(center_n)) / resolution
    half = size // 2
    x0_raw = int(round(center_x - half))
    y0_raw = int(round(center_y - half))
    x0 = int(np.clip(x0_raw, 0, map_w - size))
    y0 = int(np.clip(y0_raw, 0, map_h - size))
    x1, y1 = x0 + size, y0 + size
    crop = ref_map_full[y0:y1, x0:x1].copy()
    if crop.shape[:2] != (size, size):
        raise ValueError(f"fixed satellite crop has shape {crop.shape[:2]}, expected {(size, size)}")

    origin_e = init_e + x0 * resolution
    origin_n = init_n - y0 * resolution
    actual_center_e = origin_e + 0.5 * size * resolution
    actual_center_n = origin_n - 0.5 * size * resolution
    fully_inside = bool(
        x0_raw >= 0 and y0_raw >= 0
        and x0_raw + size <= map_w and y0_raw + size <= map_h
    )
    meta: dict[str, Any] = {
        "crop_strategy": "ly_fixed_pixel_tile",
        "tile_size_px": int(size),
        "source_x0_px": int(x0),
        "source_y0_px": int(y0),
        "source_x1_px": int(x1),
        "source_y1_px": int(y1),
        "requested_center_e": float(center_e),
        "requested_center_n": float(center_n),
        "actual_center_e": float(actual_center_e),
        "actual_center_n": float(actual_center_n),
        "center_shift_m": float(math.hypot(actual_center_e - center_e, actual_center_n - center_n)),
        "min_e": float(origin_e),
        "max_e": float(origin_e + size * resolution),
        "min_n": float(origin_n - size * resolution),
        "max_n": float(origin_n),
        "width_m": float(size * resolution),
        "height_m": float(size * resolution),
        "gsd_m_per_px": float(resolution),
    }
    return crop, float(origin_e), float(origin_n), float(resolution), fully_inside, meta


def rotate_satellite_tile_with_ly_context(
    legacy: Any,
    ref_map_full: np.ndarray,
    north_up_tile: np.ndarray,
    crop_meta: dict[str, Any],
    rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Rotate a padded real-map patch, then center-crop to the fixed tile size."""
    height, width = north_up_tile.shape[:2]
    center = (width * 0.5, height * 0.5)
    forward = cv2.getRotationMatrix2D(center, float(rotation_deg), 1.0).astype(np.float64)
    inverse = cv2.invertAffineTransform(forward).astype(np.float64)
    if abs(float(rotation_deg) % 360.0) < 1e-9:
        return legacy.make_satellite_thermal_gray(north_up_tile), forward, inverse, 0

    pad_each = int(math.ceil(max(width, height) * float(MULTIFRAME_ROTATION_CONTEXT_RATIO)))
    x0 = int(crop_meta["source_x0_px"])
    y0 = int(crop_meta["source_y0_px"])
    desired_x0, desired_y0 = x0 - pad_each, y0 - pad_each
    desired_x1, desired_y1 = x0 + width + pad_each, y0 + height + pad_each
    padded_w, padded_h = desired_x1 - desired_x0, desired_y1 - desired_y0
    padded_shape = (padded_h, padded_w) + tuple(ref_map_full.shape[2:])
    padded = np.zeros(padded_shape, dtype=ref_map_full.dtype)
    map_h, map_w = ref_map_full.shape[:2]
    source_x0, source_y0 = max(0, desired_x0), max(0, desired_y0)
    source_x1, source_y1 = min(map_w, desired_x1), min(map_h, desired_y1)
    if source_x1 > source_x0 and source_y1 > source_y0:
        padded[
            source_y0 - desired_y0:source_y1 - desired_y0,
            source_x0 - desired_x0:source_x1 - desired_x0,
        ] = ref_map_full[source_y0:source_y1, source_x0:source_x1]

    padded_input = legacy.make_satellite_thermal_gray(padded)
    padded_forward = cv2.getRotationMatrix2D(
        (padded_w * 0.5, padded_h * 0.5), float(rotation_deg), 1.0
    )
    rotated_padded = cv2.warpAffine(
        padded_input,
        padded_forward,
        (padded_w, padded_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    start_x = (padded_w - width) // 2
    start_y = (padded_h - height) // 2
    rotated = rotated_padded[start_y:start_y + height, start_x:start_x + width].copy()
    return rotated, forward, inverse, int(pad_each)


def build_multiframe_raycast_surface(
    prediction: Any,
    selected_local_indices: list[int],
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list[dict[str, Any]],
]:
    """Project the shared VGGT cloud only into selected keyframes."""
    points = np.asarray(prediction.points, dtype=np.float64).reshape(-1, 3)
    confidence = np.asarray(prediction.confidences, dtype=np.float64).reshape(-1)
    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    intrinsics = np.asarray(prediction.intrinsics, dtype=np.float64)
    image_h, image_w = [int(value) for value in prediction.images_hw]
    point_frame_ids = np.asarray(prediction.point_frame_ids, dtype=np.int32).reshape(-1)
    finite_surface = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    index_maps: dict[int, np.ndarray] = {}
    depth_maps: dict[int, np.ndarray] = {}
    projected_xy_by_frame: dict[int, np.ndarray] = {}
    projected_depth_by_frame: dict[int, np.ndarray] = {}
    stats: list[dict[str, Any]] = []
    frame_count = int(len(extrinsics))
    selected = sorted({int(value) for value in selected_local_indices})
    invalid = [value for value in selected if value < 0 or value >= frame_count]
    if invalid:
        raise IndexError(
            f"Raycast keyframe indices out of range for {frame_count} frames: {invalid}"
        )
    for local_idx in selected:
        index_map = np.full((image_h, image_w), -1, dtype=np.int32)
        depth_map = np.full((image_h, image_w), np.inf, dtype=np.float32)
        ext = extrinsics[local_idx]
        intr = intrinsics[local_idx]
        camera = points @ ext[:3, :3].T + ext[:3, 3]
        depth = camera[:, 2]
        valid = finite_surface & np.isfinite(camera).all(axis=1) & (depth > 1e-8)
        projected_xy = np.full((len(points), 2), np.nan, dtype=np.float64)
        projected_xy[valid, 0] = (
            intr[0, 0] * camera[valid, 0] / depth[valid] + intr[0, 2]
        )
        projected_xy[valid, 1] = (
            intr[1, 1] * camera[valid, 1] / depth[valid] + intr[1, 2]
        )
        if MULTIFRAME_RAYCAST_EXCLUDE_QUERY_FRAME_POINTS and len(point_frame_ids) == len(valid):
            valid &= point_frame_ids != int(local_idx)
        ids = np.flatnonzero(valid)
        if len(ids):
            x = intr[0, 0] * camera[ids, 0] / depth[ids] + intr[0, 2]
            y = intr[1, 1] * camera[ids, 1] / depth[ids] + intr[1, 2]
            xi = np.rint(x).astype(np.int32)
            yi = np.rint(y).astype(np.int32)
            inside = (xi >= 0) & (xi < image_w) & (yi >= 0) & (yi < image_h)
            ids, xi, yi = ids[inside], xi[inside], yi[inside]
            z = depth[ids]
            if len(ids):
                flat = yi.astype(np.int64) * image_w + xi.astype(np.int64)
                order = np.lexsort((z, flat))
                flat_sorted = flat[order]
                first = np.ones(len(order), dtype=bool)
                first[1:] = flat_sorted[1:] != flat_sorted[:-1]
                chosen = order[first]
                chosen_flat = flat[chosen]
                index_map.reshape(-1)[chosen_flat] = ids[chosen].astype(np.int32)
                depth_map.reshape(-1)[chosen_flat] = z[chosen].astype(np.float32)
        index_maps[local_idx] = index_map
        depth_maps[local_idx] = depth_map
        projected_xy_by_frame[local_idx] = projected_xy.astype(np.float32)
        projected_depth_by_frame[local_idx] = depth.astype(np.float32)
        stats.append({
            "frame_local_idx": int(local_idx),
            "surface_points": int(np.count_nonzero(finite_surface)),
            "occupied_pixels": int(np.count_nonzero(index_map >= 0)),
        })
    return (
        index_maps,
        depth_maps,
        projected_xy_by_frame,
        projected_depth_by_frame,
        stats,
    )


def query_multiframe_raycast(
    index_map: np.ndarray,
    depth_map: np.ndarray,
    projected_xy: np.ndarray,
    projected_depth: np.ndarray,
    mapped_px: np.ndarray,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return z-buffer point ids and pixel distances near each query pixel."""
    points = np.asarray(mapped_px, dtype=np.float64).reshape(-1, 2)
    output_ids = np.full(len(points), -1, dtype=np.int64)
    output_dist = np.full(len(points), np.inf, dtype=np.float64)
    height, width = index_map.shape[:2]
    radius = max(0, int(math.ceil(float(radius_px))))
    for query_idx, (x, y) in enumerate(points):
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xi, yi = int(round(float(x))), int(round(float(y)))
        if xi < 0 or xi >= width or yi < 0 or yi >= height:
            continue
        x0, x1 = max(0, xi - radius), min(width, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(height, yi + radius + 1)
        local_ids = index_map[y0:y1, x0:x1]
        valid = local_ids >= 0
        if not np.any(valid):
            continue
        ids = local_ids[valid].astype(np.int64)
        ids = np.unique(ids)
        depths = np.asarray(projected_depth[ids], dtype=np.float64)
        px = np.asarray(projected_xy[ids], dtype=np.float64)
        finite = np.isfinite(px).all(axis=1) & np.isfinite(depths) & (depths > 1e-8)
        ids, depths, px = ids[finite], depths[finite], px[finite]
        if len(ids) == 0:
            continue
        distance_sq = np.sum((px - np.array([x, y])) ** 2, axis=1)
        inside = distance_sq <= float(radius_px) ** 2 + 1e-9
        if np.any(inside):
            ids, depths, distance_sq = ids[inside], depths[inside], distance_sq[inside]
        if float(MULTIFRAME_RAYCAST_DEPTH_WEIGHT) > 0.0:
            normalized_depth = (depths - np.min(depths)) / max(float(np.ptp(depths)), 1e-9)
            best = int(np.argmin(distance_sq + float(MULTIFRAME_RAYCAST_DEPTH_WEIGHT) * normalized_depth))
        else:
            min_depth = float(np.min(depths))
            near = depths <= min_depth + max(1e-6, float(MULTIFRAME_RAYCAST_DEPTH_EPSILON))
            near_ids = np.flatnonzero(near)
            best = int(near_ids[int(np.argmin(distance_sq[near]))])
        output_ids[query_idx] = int(ids[best])
        output_dist[query_idx] = float(math.sqrt(float(distance_sq[best])))
    return output_ids, output_dist


def _estimate_multiframe_3d_utm_anchor_attempt(
    legacy: Any,
    matcher: Any,
    prediction: Any,
    geometry: GroundGeometry,
    window_idx: int,
    image_paths: list[Path],
    frame_names: list[str],
    prior: np.ndarray,
    camera_utm_prior: np.ndarray,
    camera_yaw_prior: np.ndarray,
    predicted_ypr: list[dict[str, float]],
    frame_metadata: list[dict[str, float]],
    height_scale_m_per_local: float,
    ref_map_full: np.ndarray,
    region_config: dict[str, Any],
    out_dir: Path,
    save_visualizations: bool,
    yaw_offset_deg: float,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Estimate one absolute local->UTM Sim(2) from several original UAV frames."""
    summary: dict[str, Any] = {
        "enabled": bool(ENABLE_MULTIFRAME_3D_UTM_ANCHOR),
        "accepted": False,
        "match_method": "multiframe_original_image_3d_to_utm",
        "selected_local_indices": [],
        "selected_frames": [],
        "per_frame": [],
        "yaw_offset_deg": float(yaw_offset_deg),
        "timing": {
            "raycast_surface_sec": 0.0,
            "minima_matching_sec": 0.0,
            "similarity_ransac_sec": 0.0,
            "minima_call_count": 0,
        },
    }
    if not ENABLE_MULTIFRAME_3D_UTM_ANCHOR:
        summary["reason"] = "disabled"
        return None, summary
    if matcher is None:
        summary["reason"] = "requires_minima_matcher"
        return None, summary

    point_frame_ids = getattr(prediction, "point_frame_ids", None)
    point_xs = getattr(prediction, "point_xs", None)
    point_ys = getattr(prediction, "point_ys", None)
    if point_frame_ids is None or point_xs is None or point_ys is None:
        summary["reason"] = "vggt_point_pixel_provenance_unavailable"
        return None, summary
    point_frame_ids = np.asarray(point_frame_ids, dtype=np.int32).reshape(-1)
    point_xs = np.asarray(point_xs, dtype=np.int32).reshape(-1)
    point_ys = np.asarray(point_ys, dtype=np.int32).reshape(-1)
    point_confidence = np.asarray(prediction.confidences, dtype=np.float64).reshape(-1)
    if not (
        len(point_frame_ids) == len(point_xs) == len(point_ys)
        == len(geometry.point_xy) == len(point_confidence)
    ):
        summary["reason"] = "vggt_point_provenance_shape_mismatch"
        return None, summary

    from scalloc.matchers.minima import MINIMA_match

    src_parts: list[np.ndarray] = []
    dst_parts: list[np.ndarray] = []
    conf_parts: list[np.ndarray] = []
    frame_parts: list[np.ndarray] = []
    uav_grid_parts: list[np.ndarray] = []
    sat_grid_parts: list[np.ndarray] = []
    visualization_parts: list[dict[str, Any]] = []
    selected_local = _multiframe_keyframe_indices(len(image_paths))
    summary["selected_local_indices"] = [int(value) for value in selected_local]
    summary["selected_frames"] = [str(frame_names[value]) for value in selected_local]
    summary["required_inlier_frame_count"] = (
        int(len(selected_local))
        if MULTIFRAME_REQUIRE_ALL_SELECTED_INLIER_FRAMES
        else 0
    )
    summary["require_all_selected_inlier_frames"] = bool(
        MULTIFRAME_REQUIRE_ALL_SELECTED_INLIER_FRAMES
    )
    prior_scale, _prior_yaw = sim2_scale_yaw(prior)
    raycast_index_maps: dict[int, np.ndarray] = {}
    raycast_depth_maps: dict[int, np.ndarray] = {}
    raycast_projected_xy: dict[int, np.ndarray] = {}
    raycast_projected_depth: dict[int, np.ndarray] = {}
    if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and MULTIFRAME_USE_RAYCAST:
        raycast_started = time.perf_counter()
        (
            raycast_index_maps,
            raycast_depth_maps,
            raycast_projected_xy,
            raycast_projected_depth,
            raycast_stats,
        ) = (
            build_multiframe_raycast_surface(prediction, selected_local)
        )
        summary["timing"]["raycast_surface_sec"] = float(
            time.perf_counter() - raycast_started
        )
        summary["vggt_query_mode"] = "raycast"
        summary["raycast_stats"] = raycast_stats
        summary["raycast_selected_local_indices"] = [
            int(value) for value in selected_local
        ]
        summary["raycast_frame_count"] = int(len(raycast_stats))
    else:
        summary["vggt_query_mode"] = "pointmap_nearest"

    for local_idx in selected_local:
        frame_record: dict[str, Any] = {
            "local_idx": int(local_idx),
            "frame": str(frame_names[local_idx]),
            "status": "failed",
        }
        image = cv2.imread(str(image_paths[local_idx]), cv2.IMREAD_UNCHANGED)
        if image is None:
            frame_record["reason"] = "image_read_failed"
            summary["per_frame"].append(frame_record)
            continue
        source_h, source_w = image.shape[:2]
        ypr = predicted_ypr[local_idx]
        if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and not USE_VGGT_PITCH_HEIGHT:
            metadata = frame_metadata[local_idx] if local_idx < len(frame_metadata) else {}
            yaw_deg = float(camera_yaw_prior[local_idx])
            pitch_deg = float(metadata.get("pitch_deg", float("nan")))
            height_m = float(metadata.get("rel_height_m", float("nan")))
            if not np.isfinite(height_m) or height_m <= 0.0:
                height_m = float(metadata.get("abs_height_m", float("nan")))
            crop_center_source = "ly_propagated_camera_metadata_pitch_rel_height"
        else:
            yaw_deg = float(ypr.get("yaw_deg", float("nan")))
            pitch_deg = float(ypr.get("pitch_deg", float("nan")))
            height_m = abs(float(geometry.camera_height[local_idx])) * float(height_scale_m_per_local)
            crop_center_source = "prior_camera_vggt_pitch_ground_intersection"
        camera_center = np.asarray(camera_utm_prior[local_idx], dtype=np.float64)
        down_deg = abs(pitch_deg)
        if (
            np.isfinite(yaw_deg) and np.isfinite(down_deg) and np.isfinite(height_m)
            and 0.1 < down_deg < 89.9 and height_m > 0.0
        ):
            ground_range = min(
                height_m / max(math.tan(math.radians(down_deg)), 1e-12),
                float(SATELLITE_MAX_PITCH_GROUND_RANGE_M),
            )
            yaw_rad = math.radians(yaw_deg)
            crop_center = camera_center + np.array([
                math.sin(yaw_rad) * ground_range,
                math.cos(yaw_rad) * ground_range,
            ], dtype=np.float64)
        else:
            ground_range = float("nan")
            crop_center = camera_center.copy()
            crop_center_source = "prior_camera_fallback"
        try:
            sat, sat_origin_e, sat_origin_n, sat_gsd, inside, crop_bounds = (
                crop_satellite_tile_at_utm_like_ly(
                    ref_map_full,
                    region_config,
                    float(crop_center[0]),
                    float(crop_center[1]),
                    int(MULTIFRAME_SATELLITE_TILE_SIZE_PX),
                )
            )
        except ValueError as exc:
            frame_record["reason"] = f"satellite_crop_failed:{exc}"
            summary["per_frame"].append(frame_record)
            continue
        uav_input = legacy.make_raw_thermal_bgr(
            image, np.full((source_h, source_w), 255, dtype=np.uint8)
        )
        rotation_deg = (
            float(yaw_deg) + float(yaw_offset_deg)
            if np.isfinite(yaw_deg) else float(yaw_offset_deg)
        )
        sat_north_input = legacy.make_satellite_thermal_gray(sat)
        sat_input, rotation_forward, rotation_inverse, rotation_context_px = (
            rotate_satellite_tile_with_ly_context(
                legacy,
                ref_map_full,
                sat,
                crop_bounds,
                rotation_deg,
            )
        )
        # Keep the on-disk path short.  With the full DJI stem this directory
        # pushed the satellite/match filenames beyond Win32's common 260-char
        # limit; cv2.imwrite then silently saved only the shortest UAV filename.
        # The full frame name remains available in summary["per_frame"].
        frame_out_dir = out_dir / "multiframe_3d_utm" / f"frame_{local_idx:02d}"
        if save_visualizations:
            frame_out_dir.mkdir(parents=True, exist_ok=True)
        match_started = time.perf_counter()
        raw0, raw1, raw_conf = MINIMA_match(
            uav_input,
            sat_input,
            matcher,
            str(frame_out_dir),
            "raw_matches.jpg",
            need_ransac=False,
            show_matches=False,
        )
        summary["timing"]["minima_matching_sec"] = float(
            summary["timing"]["minima_matching_sec"]
        ) + float(time.perf_counter() - match_started)
        summary["timing"]["minima_call_count"] = int(
            summary["timing"]["minima_call_count"]
        ) + 1
        p0 = np.asarray(raw0, dtype=np.float64).reshape(-1, 2)
        p1_rotated = np.asarray(raw1, dtype=np.float64).reshape(-1, 2)
        p1 = (
            p1_rotated @ rotation_inverse[:, :2].T + rotation_inverse[:, 2]
            if len(p1_rotated) else p1_rotated.copy()
        )
        confidence = np.asarray(raw_conf, dtype=np.float64).reshape(-1)
        finite = (
            np.isfinite(p0).all(axis=1) & np.isfinite(p1).all(axis=1)
            & np.isfinite(p1_rotated).all(axis=1)
            & np.isfinite(confidence) & (confidence >= float(MINIMA_MIN_CONF))
        )
        finite &= (
            (p0[:, 0] >= 0.0) & (p0[:, 0] < source_w)
            & (p0[:, 1] >= 0.0) & (p0[:, 1] < source_h)
            & (p1_rotated[:, 0] >= 0.0) & (p1_rotated[:, 0] < sat_input.shape[1])
            & (p1_rotated[:, 1] >= 0.0) & (p1_rotated[:, 1] < sat_input.shape[0])
            & (p1[:, 0] >= 0.0) & (p1[:, 0] < sat_north_input.shape[1])
            & (p1[:, 1] >= 0.0) & (p1[:, 1] < sat_north_input.shape[0])
        )
        p0, p1, p1_rotated, confidence = (
            p0[finite], p1[finite], p1_rotated[finite], confidence[finite]
        )
        balanced = _dual_grid_balanced_indices(
            p0, p1, confidence, (source_h, source_w), sat_north_input.shape[:2]
        )
        p0, p1, p1_rotated, confidence = (
            p0[balanced], p1[balanced], p1_rotated[balanced], confidence[balanced]
        )
        if save_visualizations:
            cv2.imwrite(str(frame_out_dir / "uav_match_input.jpg"), uav_input)
            cv2.imwrite(str(frame_out_dir / "satellite_match_input.jpg"), sat_input)
            save_matches(
                uav_input,
                sat_input,
                p0,
                p1_rotated,
                frame_out_dir / "direct_balanced_matches.jpg",
            )

        if len(p0) == 0:
            frame_record.update({
                "reason": "no_balanced_matches_or_pointmap_points",
                "raw_matches": int(len(raw0)),
                "balanced_matches": int(len(p0)),
            })
            summary["per_frame"].append(frame_record)
            continue
        mapped_px = _source_pixels_to_vggt(p0, source_w, source_h)
        if local_idx in raycast_index_maps:
            mapped_point_ids, lookup_distance = query_multiframe_raycast(
                raycast_index_maps[local_idx],
                raycast_depth_maps[local_idx],
                raycast_projected_xy[local_idx],
                raycast_projected_depth[local_idx],
                mapped_px,
                float(MULTIFRAME_POINT_LOOKUP_RADIUS_PX),
            )
        else:
            frame_point_ids = np.flatnonzero(
                (point_frame_ids == int(local_idx)) & (point_xs >= 0) & (point_ys >= 0)
            )
            if len(frame_point_ids) == 0:
                frame_record.update({
                    "reason": "no_pointmap_points_for_frame",
                    "raw_matches": int(len(raw0)),
                    "balanced_matches": int(len(p0)),
                })
                summary["per_frame"].append(frame_record)
                continue
            lookup_px = np.column_stack([point_xs[frame_point_ids], point_ys[frame_point_ids]])
            tree = cKDTree(lookup_px.astype(np.float64))
            lookup_distance, lookup_index = tree.query(mapped_px, k=1)
            mapped_point_ids = frame_point_ids[np.asarray(lookup_index, dtype=np.int64)]
        valid_lookup = (
            np.isfinite(lookup_distance)
            & (lookup_distance <= float(MULTIFRAME_POINT_LOOKUP_RADIUS_PX))
            & (mapped_point_ids >= 0)
        )
        valid_ids = np.flatnonzero(valid_lookup)
        if len(valid_ids):
            valid_lookup[valid_ids] &= np.isfinite(
                geometry.point_xy[mapped_point_ids[valid_ids]]
            ).all(axis=1)
        if not np.any(valid_lookup):
            frame_record.update({
                "reason": "no_vggt_points_within_lookup_radius",
                "raw_matches": int(len(raw0)),
                "balanced_matches": int(len(p0)),
            })
            summary["per_frame"].append(frame_record)
            continue
        mapped_point_ids = mapped_point_ids[valid_lookup]
        p0_valid = p0[valid_lookup]
        p1_valid = p1[valid_lookup]
        p1_rotated_valid = p1_rotated[valid_lookup]
        match_conf = confidence[valid_lookup]
        if ENABLE_LY_DIRECT_ANCHOR_PIPELINE:
            # LY ranks and refines SWG using matcher confidence only.  VGGT
            # confidence controls the reconstructed surface, not match weight.
            combined_conf = match_conf
        else:
            point_conf = np.maximum(point_confidence[mapped_point_ids], 1e-6)
            point_conf /= max(float(np.median(point_conf)), 1e-6)
            combined_conf = match_conf * np.clip(point_conf, 0.25, 4.0)
        src_parts.append(np.asarray(geometry.point_xy[mapped_point_ids], dtype=np.float64))
        dst_parts.append(pixels_to_utm(p1_valid, sat_origin_e, sat_origin_n, sat_gsd))
        conf_parts.append(np.asarray(combined_conf, dtype=np.float64))
        frame_parts.append(np.full(len(mapped_point_ids), int(local_idx), dtype=np.int32))
        uav_grid_parts.append(_grid_ids_for_points(p0_valid, (source_h, source_w)))
        sat_grid_parts.append(_grid_ids_for_points(p1_valid, sat_north_input.shape[:2]))
        if save_visualizations:
            visualization_parts.append({
                "out_dir": frame_out_dir,
                "uav_input": uav_input,
                "satellite_input": sat_input,
                "uav_points": p0_valid,
                "satellite_points": p1_rotated_valid,
                "count": int(len(mapped_point_ids)),
            })
        frame_record.update({
            "status": "ok",
            "reason": "",
            "raw_matches": int(len(raw0)),
            "balanced_matches": int(len(p0)),
            "mapped_correspondences": int(len(mapped_point_ids)),
            "lookup_distance_median_px": float(np.median(lookup_distance[valid_lookup])),
            "crop_center_utm": crop_center.tolist(),
            "crop_center_source": crop_center_source,
            "ground_range_m": float(ground_range) if np.isfinite(ground_range) else None,
            "satellite_rotation_deg": float(rotation_deg),
            "satellite_crop_strategy": str(crop_bounds.get("crop_strategy", "")),
            "satellite_tile_size_px": int(MULTIFRAME_SATELLITE_TILE_SIZE_PX),
            "satellite_rotation_context_px": int(rotation_context_px),
            "satellite_rotation_inverse": rotation_inverse.tolist(),
            "satellite_crop_inside_map": bool(inside),
            "satellite_crop_bounds_utm": crop_bounds,
        })
        if save_visualizations:
            save_matches(
                uav_input,
                sat_input,
                p0_valid,
                p1_rotated_valid,
                frame_out_dir / "pointmap_mapped_matches.jpg",
            )
        summary["per_frame"].append(frame_record)

    summary["raw_matches"] = int(sum(
        int(record.get("raw_matches", 0)) for record in summary["per_frame"]
    ))
    summary["valid_matches"] = int(sum(
        int(record.get("mapped_correspondences", 0)) for record in summary["per_frame"]
    ))
    if not src_parts:
        summary["reason"] = "no_mapped_correspondences"
        return None, summary
    src = np.concatenate(src_parts, axis=0)
    dst = np.concatenate(dst_parts, axis=0)
    confidence = np.concatenate(conf_parts, axis=0)
    frame_ids = np.concatenate(frame_parts, axis=0)
    uav_grid_ids = np.concatenate(uav_grid_parts, axis=0)
    sat_grid_ids = np.concatenate(sat_grid_parts, axis=0)
    summary["candidate_count"] = int(len(src))
    summary["mapped_frame_count"] = int(np.unique(frame_ids).size)
    summary["candidate_uav_unique_grids"] = int(np.unique(uav_grid_ids).size)
    summary["candidate_sat_unique_grids"] = int(np.unique(sat_grid_ids).size)
    if len(src) < int(MULTIFRAME_MIN_CORRESPONDENCES):
        summary["reason"] = "too_few_correspondences"
        return None, summary

    ransac_started = time.perf_counter()
    if ENABLE_LY_DIRECT_ANCHOR_PIPELINE:
        candidate, inliers, ransac_stats = similarity_ransac_anchored_like_ly(
            src,
            dst,
            confidence,
            uav_grid_ids,
            sat_grid_ids,
            geometry.camera_xy[min(len(geometry.camera_xy) - 1, len(image_paths) // 2)],
            geometry.camera_xy[0],
            camera_utm_prior[0],
        )
    else:
        candidate, inliers, ransac_stats = similarity_ransac(
            src,
            dst,
            confidence,
            float(MULTIFRAME_RANSAC_THRESHOLD_M),
            int(MULTIFRAME_MIN_INLIERS),
        )
    summary["timing"]["similarity_ransac_sec"] = float(
        time.perf_counter() - ransac_started
    )
    summary.update(ransac_stats)
    if save_visualizations and len(inliers) == len(src):
        offset = 0
        inliers = np.asarray(inliers, dtype=bool)
        for visual in visualization_parts:
            count = int(visual["count"])
            frame_inliers = inliers[offset:offset + count]
            offset += count
            if np.any(frame_inliers):
                save_matches(
                    visual["uav_input"],
                    visual["satellite_input"],
                    np.asarray(visual["uav_points"])[frame_inliers],
                    np.asarray(visual["satellite_points"])[frame_inliers],
                    Path(visual["out_dir"]) / "ransac_inlier_matches.jpg",
                )
    if candidate is None:
        summary["reason"] = str(ransac_stats.get("reason", "similarity_failed"))
        return None, summary
    inlier_frames = np.unique(frame_ids[inliers])
    summary["inlier_frame_count"] = int(len(inlier_frames))
    summary["inlier_local_indices"] = [int(value) for value in inlier_frames]
    reasons: list[str] = []
    if float(summary.get("inlier_ratio", 0.0)) < float(MULTIFRAME_MIN_INLIER_RATIO):
        reasons.append("low_inlier_ratio")
    if int(summary.get("inlier_count", 0)) < int(MULTIFRAME_MIN_INLIERS):
        reasons.append("too_few_inliers")
    if (
        MULTIFRAME_REQUIRE_ALL_SELECTED_INLIER_FRAMES
        and len(inlier_frames) < len(selected_local)
    ):
        reasons.append("too_few_inlier_frames")
    if float(summary.get("rmse", float("inf"))) > float(MULTIFRAME_MAX_RMSE_M):
        reasons.append("high_rmse")
    visual_gate_pass = bool(not reasons)
    position_gate_distance_m = float("nan")
    position_gate_decision = "disabled"
    if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and int(window_idx) > 0:
        gate_local_idx = min(
            len(geometry.camera_xy) - 1,
            max(0, int(VGGT_WINDOW_OVERLAP) - 1),
        )
        visual_overlap_position = apply_sim2(
            candidate, geometry.camera_xy[gate_local_idx:gate_local_idx + 1]
        )[0]
        position_gate_distance_m = float(
            np.linalg.norm(visual_overlap_position - camera_utm_prior[gate_local_idx])
        )
        sat_unique_grids = int(summary.get("candidate_sat_unique_grids", 0))
        if position_gate_distance_m <= float(MULTIFRAME_GATE_STRONG_DISTANCE_M):
            position_gate_decision = "strong_accept"
        elif (
            position_gate_distance_m <= float(MULTIFRAME_GATE_WEAK_DISTANCE_M)
            and int(summary.get("inlier_count", 0)) >= int(MULTIFRAME_GATE_WEAK_MIN_INLIERS)
            and float(summary.get("rmse", float("inf"))) <= float(MULTIFRAME_GATE_WEAK_MAX_RMSE_M)
            and sat_unique_grids >= int(MULTIFRAME_GATE_WEAK_MIN_SAT_GRIDS)
        ):
            position_gate_decision = "weak_accept"
        elif (
            visual_gate_pass
            and position_gate_distance_m <= float(MULTIFRAME_GATE_OVERRIDE_MAX_DISTANCE_M)
            and int(summary.get("inlier_count", 0)) >= 80
            and float(summary.get("rmse", float("inf"))) <= 10.0
            and sat_unique_grids >= int(MULTIFRAME_GATE_OVERRIDE_MIN_SAT_GRIDS)
            and abs(wrap_deg(float(yaw_offset_deg))) <= 15.0
        ):
            position_gate_decision = "override_visual_strong"
        else:
            position_gate_decision = "reject"
            reasons.append("ly_overlap_position_gate")
    elif ENABLE_LY_DIRECT_ANCHOR_PIPELINE:
        position_gate_decision = "window0_skip"
    residual = compose_sim2(candidate, invert_sim2(prior))
    residual_scale, residual_yaw = sim2_scale_yaw(residual)
    absolute_scale, absolute_yaw = sim2_scale_yaw(candidate)
    summary.update({
        "accepted": bool(not reasons),
        "reason": ";".join(reasons),
        "absolute_scale_m_per_local": float(absolute_scale),
        "absolute_yaw_deg": float(absolute_yaw),
        "residual_scale": float(residual_scale),
        "residual_yaw_deg": float(residual_yaw),
        "prior_scale_m_per_local": float(prior_scale),
        "quality_score": _candidate_quality_score({**summary, "accepted": not reasons}, multiframe=True),
        "ly_overlap_position_gate_distance_m": (
            float(position_gate_distance_m) if np.isfinite(position_gate_distance_m) else None
        ),
        "ly_overlap_position_gate_decision": position_gate_decision,
        "visual_gate_pass": bool(visual_gate_pass),
        # Keep the valid geometric hypothesis even when a branch-local gate
        # rejects it, so an independent BEV candidate can verify it later.
        "raw_candidate_matrix": candidate.tolist(),
    })
    return (candidate if not reasons else None), summary


def estimate_multiframe_3d_utm_anchor(
    legacy: Any,
    matcher: Any,
    prediction: Any,
    geometry: GroundGeometry,
    window_idx: int,
    image_paths: list[Path],
    frame_names: list[str],
    prior: np.ndarray,
    camera_utm_prior: np.ndarray,
    camera_yaw_prior: np.ndarray,
    predicted_ypr: list[dict[str, float]],
    frame_metadata: list[dict[str, float]],
    height_scale_m_per_local: float,
    ref_map_full: np.ndarray,
    region_config: dict[str, Any],
    out_dir: Path,
    save_visualizations: bool,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """LY ordered yaw attempts around the propagated compass heading."""
    offsets = (
        MULTIFRAME_YAW_OFFSETS_DEG
        if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else (0.0,)
    )
    attempt_log: list[dict[str, Any]] = []
    selected_result: np.ndarray | None = None
    selected_stats: dict[str, Any] | None = None
    for attempt_idx, yaw_offset in enumerate(offsets):
        attempt_started = time.perf_counter()
        attempt_dir = (
            out_dir / f"ly_yaw_attempt_{attempt_idx:02d}_{float(yaw_offset):+06.1f}deg"
            if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else out_dir
        )
        result, stats = _estimate_multiframe_3d_utm_anchor_attempt(
            legacy=legacy,
            matcher=matcher,
            prediction=prediction,
            geometry=geometry,
            window_idx=window_idx,
            image_paths=image_paths,
            frame_names=frame_names,
            prior=prior,
            camera_utm_prior=camera_utm_prior,
            camera_yaw_prior=camera_yaw_prior,
            predicted_ypr=predicted_ypr,
            frame_metadata=frame_metadata,
            height_scale_m_per_local=height_scale_m_per_local,
            ref_map_full=ref_map_full,
            region_config=region_config,
            out_dir=attempt_dir,
            save_visualizations=save_visualizations,
            yaw_offset_deg=float(yaw_offset),
        )
        attempt_runtime_sec = float(time.perf_counter() - attempt_started)
        attempt_timing = stats.get("timing", {})
        attempt_log.append({
            "attempt_idx": int(attempt_idx),
            "yaw_offset_deg": float(yaw_offset),
            "visual_gate_pass": bool(stats.get("visual_gate_pass", False)),
            "accepted": bool(stats.get("accepted", False)),
            "reason": str(stats.get("reason", "")),
            "inlier_count": int(stats.get("inlier_count", 0)),
            "inlier_ratio": float(stats.get("inlier_ratio", 0.0)),
            "rmse_m": stats.get("rmse"),
            "runtime_sec": attempt_runtime_sec,
            "raycast_surface_sec": float(attempt_timing.get("raycast_surface_sec", 0.0)),
            "minima_matching_sec": float(attempt_timing.get("minima_matching_sec", 0.0)),
            "similarity_ransac_sec": float(attempt_timing.get("similarity_ransac_sec", 0.0)),
            "minima_call_count": int(attempt_timing.get("minima_call_count", 0)),
        })
        selected_result, selected_stats = result, stats
        if bool(stats.get("visual_gate_pass", False)):
            break
    if selected_stats is None:
        return None, {"accepted": False, "reason": "no_yaw_attempt_executed"}
    chosen_offset = float(selected_stats.get("yaw_offset_deg", 0.0))
    if (
        ENABLE_LY_DIRECT_ANCHOR_PIPELINE
        and abs(wrap_deg(chosen_offset)) > float(MULTIFRAME_MAX_ACCEPTED_YAW_OFFSET_DEG)
    ):
        selected_result = None
        selected_stats["accepted"] = False
        selected_stats["reason"] = "large_yaw_offset_fallback"
    selected_stats["yaw_attempts"] = attempt_log
    selected_stats["chosen_yaw_offset_deg"] = float(chosen_offset)
    selected_stats["all_yaw_attempt_timing"] = {
        "attempt_count": int(len(attempt_log)),
        "runtime_sec": float(sum(item["runtime_sec"] for item in attempt_log)),
        "raycast_surface_sec": float(sum(item["raycast_surface_sec"] for item in attempt_log)),
        "minima_matching_sec": float(sum(item["minima_matching_sec"] for item in attempt_log)),
        "similarity_ransac_sec": float(sum(item["similarity_ransac_sec"] for item in attempt_log)),
        "minima_call_count": int(sum(item["minima_call_count"] for item in attempt_log)),
    }
    return selected_result, selected_stats


def fuse_bev_and_multiframe_candidates(
    prior: np.ndarray,
    pivot_local_xy: np.ndarray,
    bev_residual: np.ndarray | None,
    bev_stats: dict[str, Any],
    multiframe_candidate: np.ndarray | None,
    multiframe_stats: dict[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fuse independent candidates, returning one residual to apply exactly once."""
    bev_scale_only = bool(bev_stats.get("scale_observation_only", False))
    bev_candidate = (
        compose_sim2(bev_residual, prior)
        if bev_residual is not None and not bev_scale_only else None
    )
    raw_bev_candidate: np.ndarray | None = None
    raw_bev_residual = bev_stats.get("raw_residual_sim2")
    if raw_bev_residual is not None:
        try:
            raw_bev_candidate = compose_sim2(
                np.asarray(raw_bev_residual, dtype=np.float64).reshape(2, 3),
                prior,
            )
        except (TypeError, ValueError):
            raw_bev_candidate = None
    raw_multiframe_candidate: np.ndarray | None = None
    raw_multiframe = multiframe_stats.get("raw_candidate_matrix")
    if raw_multiframe is not None:
        try:
            raw_multiframe_candidate = np.asarray(
                raw_multiframe, dtype=np.float64
            ).reshape(2, 3)
        except (TypeError, ValueError):
            raw_multiframe_candidate = None

    rescue_consensus: dict[str, Any] = {
        "enabled": bool(ENABLE_BEV_RESCUE_MODE),
        "activated": False,
        "raw_bev_candidate_available": bool(raw_bev_candidate is not None),
        "raw_multiframe_candidate_available": bool(raw_multiframe_candidate is not None),
    }
    if (
        ENABLE_BEV_RESCUE_MODE
        and raw_bev_candidate is not None
        and raw_multiframe_candidate is not None
        and (bev_candidate is None or multiframe_candidate is None)
    ):
        pivot = np.asarray(pivot_local_xy, dtype=np.float64).reshape(1, 2)
        raw_bev_position = apply_sim2(raw_bev_candidate, pivot)[0]
        raw_multi_position = apply_sim2(raw_multiframe_candidate, pivot)[0]
        raw_bev_scale, raw_bev_yaw = sim2_scale_yaw(raw_bev_candidate)
        raw_multi_scale, raw_multi_yaw = sim2_scale_yaw(raw_multiframe_candidate)
        raw_position_difference = float(
            np.linalg.norm(raw_bev_position - raw_multi_position)
        )
        raw_yaw_difference = abs(wrap_deg(raw_bev_yaw - raw_multi_yaw))
        raw_scale_log_difference = abs(math.log(
            max(raw_bev_scale, 1e-12) / max(raw_multi_scale, 1e-12)
        ))
        raw_agreement = bool(
            raw_position_difference <= float(BEV_RESCUE_AGREEMENT_POSITION_M)
            and raw_yaw_difference <= float(BEV_RESCUE_AGREEMENT_YAW_DEG)
            and raw_scale_log_difference <= float(BEV_RESCUE_AGREEMENT_SCALE_LOG)
        )
        rescue_consensus.update({
            "agreement": raw_agreement,
            "position_difference_m": raw_position_difference,
            "yaw_difference_deg": raw_yaw_difference,
            "scale_log_difference": raw_scale_log_difference,
        })
        if raw_agreement:
            bev_candidate = raw_bev_candidate
            multiframe_candidate = raw_multiframe_candidate
            rescue_consensus["activated"] = True

    bev_score = _candidate_quality_score(bev_stats, multiframe=False) if bev_candidate is not None else 0.0
    multi_score = (
        _candidate_quality_score(multiframe_stats, multiframe=True)
        if multiframe_candidate is not None else 0.0
    )
    fusion: dict[str, Any] = {
        "enabled": bool(ENABLE_DUAL_ANCHOR_FUSION),
        "bev_enabled": bool(ENABLE_BEV_ANCHOR),
        "multiframe_enabled": bool(ENABLE_MULTIFRAME_3D_UTM_ANCHOR),
        "bev_candidate_available": bool(bev_candidate is not None),
        "multiframe_candidate_available": bool(multiframe_candidate is not None),
        "bev_quality_score": float(bev_score),
        "multiframe_quality_score": float(multi_score),
        "bev": bev_stats,
        "multiframe": multiframe_stats,
        "bev_rescue_consensus": rescue_consensus,
    }

    representative: dict[str, Any]
    candidate: np.ndarray | None
    if bev_candidate is not None and multiframe_candidate is not None:
        pivot = np.asarray(pivot_local_xy, dtype=np.float64).reshape(1, 2)
        bev_position = apply_sim2(bev_candidate, pivot)[0]
        multi_position = apply_sim2(multiframe_candidate, pivot)[0]
        bev_scale, bev_yaw = sim2_scale_yaw(bev_candidate)
        multi_scale, multi_yaw = sim2_scale_yaw(multiframe_candidate)
        position_difference = float(np.linalg.norm(bev_position - multi_position))
        yaw_difference = abs(wrap_deg(bev_yaw - multi_yaw))
        scale_log_difference = abs(math.log(max(bev_scale, 1e-12) / max(multi_scale, 1e-12)))
        agreement = bool(
            position_difference <= float(DUAL_ANCHOR_AGREEMENT_POSITION_M)
            and yaw_difference <= float(DUAL_ANCHOR_AGREEMENT_YAW_DEG)
            and scale_log_difference <= float(DUAL_ANCHOR_AGREEMENT_SCALE_LOG)
        )
        fusion.update({
            "position_difference_m": position_difference,
            "yaw_difference_deg": yaw_difference,
            "scale_log_difference": scale_log_difference,
            "agreement": agreement,
        })
        if ENABLE_DUAL_ANCHOR_FUSION and agreement:
            weights = np.asarray([max(bev_score, 1e-6), max(multi_score, 1e-6)], dtype=np.float64)
            weights /= float(weights.sum())
            pivot_position = weights[0] * bev_position + weights[1] * multi_position
            fused_log_scale = weights[0] * math.log(bev_scale) + weights[1] * math.log(multi_scale)
            fused_yaw = circular_weighted_mean([bev_yaw, multi_yaw], weights.tolist())
            candidate = sim2_from_pivot(
                pivot_local_xy,
                pivot_position,
                math.exp(fused_log_scale),
                fused_yaw,
            )
            representative = multiframe_stats if multi_score >= bev_score else bev_stats
            fusion.update({
                "decision": (
                    "bev_rescue_consensus_weighted_fusion"
                    if rescue_consensus["activated"]
                    else "consensus_weighted_fusion"
                ),
                "bev_weight": float(weights[0]),
                "multiframe_weight": float(weights[1]),
            })
        else:
            rescue_conflict_prefers_multiframe = bool(
                ENABLE_BEV_RESCUE_MODE
                and not ENABLE_BEV_ANCHOR
                and bev_stats.get("rescue_triggered", False)
            )
            choose_multiframe = bool(
                rescue_conflict_prefers_multiframe
                or multi_score
                >= float(DUAL_ANCHOR_CONFLICT_MULTIFRAME_SCORE_RATIO)
                * max(bev_score, 1e-6)
            )
            candidate = multiframe_candidate if choose_multiframe else bev_candidate
            representative = multiframe_stats if choose_multiframe else bev_stats
            if rescue_conflict_prefers_multiframe:
                fusion["decision"] = "conflict_prefer_multiframe_during_bev_rescue"
            else:
                fusion["decision"] = (
                    "conflict_selected_multiframe"
                    if choose_multiframe
                    else "conflict_selected_bev"
                )
    elif multiframe_candidate is not None:
        candidate = multiframe_candidate
        representative = multiframe_stats
        fusion.update({"decision": "multiframe_only", "agreement": None})
    elif bev_candidate is not None:
        temporal_confirmation_required = bool(
            ENABLE_BEV_RESCUE_TEMPORAL_CONFIRMATION
            and ENABLE_BEV_RESCUE_MODE
            and not ENABLE_BEV_ANCHOR
            and bev_stats.get("rescue_triggered", False)
        )
        temporal_confirmation_passed = bool(
            bev_stats.get("bev_rescue_temporal_confirmed", False)
        )
        bev_rescue_waiting = bool(
            ENABLE_BEV_RESCUE_MODE
            and not ENABLE_BEV_ANCHOR
            and bev_stats.get("rescue_triggered", False)
            and (
                (
                    temporal_confirmation_required
                    and not temporal_confirmation_passed
                )
                or (
                    not temporal_confirmation_required
                    and not bev_stats.get("bev_rescue_lost_mode", False)
                )
            )
        )
        if bev_rescue_waiting:
            failed = dict(bev_stats)
            fusion["decision"] = (
                "bev_rescue_waiting_for_temporal_confirmation"
                if temporal_confirmation_required
                else "bev_rescue_waiting_for_lost_or_consensus"
            )
            failed.update({
                "accepted": False,
                "reason": fusion["decision"],
                "match_method": "dual_bev_multiframe_pending",
                "scale_observation_only": False,
                "dual_anchor_fusion": fusion,
            })
            return None, failed
        candidate = bev_candidate
        representative = bev_stats
        fusion.update({
            "decision": (
                "bev_rescue_temporally_confirmed_relocalization"
                if temporal_confirmation_passed
                else "bev_rescue_lost_relocalization"
                if bev_stats.get("bev_rescue_lost_mode", False)
                else "bev_only"
            ),
            "agreement": None,
        })
    elif bev_residual is not None and bev_scale_only:
        # Preserve the existing scale-filter-only observation when neither branch
        # has a full pose candidate.
        preserved = dict(bev_stats)
        preserved["dual_anchor_fusion"] = fusion
        preserved["match_method"] = "bev_scale_observation_only"
        fusion["decision"] = "bev_scale_observation_only"
        return bev_residual, preserved
    else:
        failed = dict(bev_stats)
        failed.update({
            "accepted": False,
            "reason": "bev_and_multiframe_failed",
            "match_method": "dual_bev_multiframe_failed",
            "scale_observation_only": False,
            "dual_anchor_fusion": fusion,
        })
        fusion["decision"] = "no_candidate"
        return None, failed

    residual = compose_sim2(candidate, invert_sim2(prior))
    residual_scale, residual_yaw = sim2_scale_yaw(residual)
    output = dict(representative)
    output.update({
        "accepted": True,
        "reason": "",
        "match_method": "dual_bev_multiframe",
        "scale_observation_only": False,
        "scale": float(residual_scale),
        "theta_deg": float(residual_yaw),
        "dual_anchor_fusion": fusion,
        "dual_anchor_decision": fusion.get("decision"),
    })
    return residual, output


def interpolate_orthophoto_for_matching(
    img_bgr: np.ndarray,
    mask_img: np.ndarray,
    inpaint_radius: int = 3,
    blur_sigma: float = 0.75,
    blend_alpha: float = 0.55,
) -> np.ndarray:
    """Densify the orthophoto inside its valid footprint without using its border."""
    if img_bgr is None or mask_img is None:
        return img_bgr
    mask_bool = np.asarray(mask_img) > 127
    if not np.any(mask_bool):
        return img_bgr
    out = np.asarray(img_bgr).copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out.copy()
    valid_gray = gray[mask_bool]
    fill_value = int(np.median(valid_gray)) if valid_gray.size > 0 else 0
    work = out.copy()
    work[~mask_bool] = fill_value
    dark_holes = (gray <= 2) & mask_bool
    if np.any(dark_holes):
        work = cv2.inpaint(
            work,
            dark_holes.astype(np.uint8) * 255,
            int(inpaint_radius),
            cv2.INPAINT_TELEA,
        )
    height, width = work.shape[:2]
    upsampled = cv2.resize(
        work, (width * 2, height * 2), interpolation=cv2.INTER_LANCZOS4
    )
    interpolated = cv2.resize(
        upsampled, (width, height), interpolation=cv2.INTER_AREA
    )
    if blur_sigma is not None and blur_sigma > 0:
        smooth = cv2.GaussianBlur(
            interpolated, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma
        )
    else:
        smooth = interpolated
    blended = cv2.addWeighted(work, 1.0 - blend_alpha, smooth, blend_alpha, 0)
    out[mask_bool] = blended[mask_bool]
    out[~mask_bool] = 0
    return out


def postprocess_orthophoto_for_matching(
    img_bgr: np.ndarray,
    mask_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Port of the morphology, inpaint and interpolation path in generate_orthophoto."""
    mask_raw = (np.asarray(mask_img) > 0).astype(np.uint8) * 255
    if img_bgr is None or not np.any(mask_raw > 0):
        return img_bgr, mask_raw, {"applied": False, "reason": "empty_input_or_mask"}
    image = np.asarray(img_bgr).copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    valid = mask_raw > 0
    valid_gray = gray[valid]
    if valid_gray.size > 0:
        p_min, p_max = np.percentile(valid_gray, [1, 99])
        if p_max > p_min:
            gray_norm = np.clip(
                (gray.astype(np.float32) - p_min) / (p_max - p_min) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            image = cv2.merge([gray_norm, gray_norm, gray_norm])
            image[~valid] = 0
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_dilated = cv2.dilate(mask_raw, kernel_dilate, iterations=1)
    mask_final = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)
    hole_mask = cv2.bitwise_and(mask_final, cv2.bitwise_not(mask_raw))
    image_closed = cv2.bitwise_and(image, image, mask=mask_raw)
    if np.any(hole_mask > 0):
        image_closed = cv2.inpaint(
            image_closed, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA
        )
    image_closed[mask_final == 0] = 0
    image_interpolated = interpolate_orthophoto_for_matching(
        image_closed,
        mask_final,
        inpaint_radius=3,
        blur_sigma=0.75,
        blend_alpha=0.55,
    )
    return image_interpolated, mask_final, {
        "applied": True,
        "method": "mask_dilate_close_telea_lanczos_gaussian_blend",
        "dilate_kernel": [9, 9],
        "close_kernel": [15, 15],
        "telea_inpaint_radius": 5,
        "interpolation_inpaint_radius": 3,
        "blur_sigma": 0.75,
        "blend_alpha": 0.55,
        "raw_valid_pixels": int(np.count_nonzero(mask_raw)),
        "processed_valid_pixels": int(np.count_nonzero(mask_final)),
        "filled_hole_pixels": int(np.count_nonzero(hole_mask)),
    }


def rasterize_prior_bev(
    geometry: GroundGeometry,
    colors_rgb: np.ndarray,
    local_to_utm: np.ndarray,
    satellite_gsd: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point_utm = apply_sim2(local_to_utm, geometry.point_xy)
    camera_utm = apply_sim2(local_to_utm, geometry.camera_xy)
    camera_center = np.median(camera_utm, axis=0)
    distance = np.linalg.norm(point_utm - camera_center, axis=1)
    finite = np.isfinite(point_utm).all(axis=1) & np.isfinite(geometry.point_height)
    if not BEV_USE_FULL_POINT_CLOUD:
        finite &= distance <= float(BEV_MAX_RADIUS_M)
        if int(finite.sum()) < 1000:
            finite = np.isfinite(point_utm).all(axis=1) & np.isfinite(geometry.point_height)
    pts = point_utm[finite]
    height = geometry.point_height[finite]
    colors = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)[finite]
    if len(pts) == 0:
        raise ValueError("No finite VGGT points are available for BEV rasterization")
    if BEV_USE_FULL_POINT_CLOUD:
        low = np.min(pts, axis=0)
        high = np.max(pts, axis=0)
    else:
        low = np.percentile(pts, 0.5, axis=0)
        high = np.percentile(pts, 99.5, axis=0)
    margin = max(8.0, 8.0 * float(satellite_gsd))
    min_e, min_n = low - margin
    max_e, max_n = high + margin
    footprint_center = np.array([(min_e + max_e) * 0.5, (min_n + max_n) * 0.5], dtype=np.float64)
    width = max(32, int(math.ceil((max_e - min_e) / satellite_gsd)) + 1)
    height_px = max(32, int(math.ceil((max_n - min_n) / satellite_gsd)) + 1)
    spatial_cap_applied = False
    if not BEV_USE_FULL_POINT_CLOUD and max(width, height_px) > BEV_MAX_SIDE_PX:
        # For oblique imagery the ground footprint can be far ahead of the camera.
        # Keep the capped BEV around that footprint, not around the camera trajectory.
        center_e, center_n = footprint_center
        half = 0.5 * BEV_MAX_SIDE_PX * satellite_gsd
        min_e, max_e = center_e - half, center_e + half
        min_n, max_n = center_n - half, center_n + half
        width = height_px = BEV_MAX_SIDE_PX
        inside = (
            (pts[:, 0] >= min_e) & (pts[:, 0] <= max_e)
            & (pts[:, 1] >= min_n) & (pts[:, 1] <= max_n)
        )
        pts, height, colors = pts[inside], height[inside], colors[inside]
        footprint_center = np.array([center_e, center_n], dtype=np.float64)
        spatial_cap_applied = True
    col = np.floor((pts[:, 0] - min_e) / satellite_gsd).astype(np.int64)
    row = np.floor((max_n - pts[:, 1]) / satellite_gsd).astype(np.int64)
    inside = (col >= 0) & (col < width) & (row >= 0) & (row < height_px)
    col, row, height, colors = col[inside], row[inside], height[inside], colors[inside]
    flat = row * width + col
    order = np.lexsort((height, flat))
    flat_sorted = flat[order]
    keep = np.r_[flat_sorted[1:] != flat_sorted[:-1], True]
    selected = order[keep]
    image = np.zeros((height_px, width, 3), dtype=np.uint8)
    mask = np.zeros((height_px, width), dtype=np.uint8)
    image[row[selected], col[selected]] = colors[selected][:, ::-1]
    mask[row[selected], col[selected]] = 255
    image, mask, postprocess_stats = postprocess_orthophoto_for_matching(image, mask)
    prior_scale, prior_theta_deg = sim2_scale_yaw(local_to_utm)
    center_offset = footprint_center - camera_center
    return image, mask, {
        "origin_e": float(min_e),
        "origin_n": float(max_n),
        "gsd": float(satellite_gsd),
        "width": int(width),
        "height": int(height_px),
        # Kept as the crop-center compatibility key. It is intentionally the
        # visible ground footprint center, not the oblique camera position.
        "center_utm": [float(footprint_center[0]), float(footprint_center[1])],
        "footprint_center_utm": [float(footprint_center[0]), float(footprint_center[1])],
        "camera_center_utm": [float(camera_center[0]), float(camera_center[1])],
        "camera_to_footprint_center_offset_m": [float(center_offset[0]), float(center_offset[1])],
        "camera_to_footprint_center_distance_m": float(np.linalg.norm(center_offset)),
        "bounds_utm": {
            "min_e": float(min_e),
            "max_e": float(max_e),
            "min_n": float(min_n),
            "max_n": float(max_n),
        },
        "orientation": "north_up_utm",
        "rotation_source": "propagated_local_to_utm_sim2",
        "prior_sim2_scale": float(prior_scale),
        "prior_sim2_theta_deg": float(prior_theta_deg),
        "extent_m": [float(max_e - min_e), float(max_n - min_n)],
        "point_count": int(len(selected)),
        "source_point_count": int(len(point_utm)),
        "finite_point_count": int(np.count_nonzero(finite)),
        "use_full_point_cloud": bool(BEV_USE_FULL_POINT_CLOUD),
        "radius_filter_applied": bool(not BEV_USE_FULL_POINT_CLOUD),
        "percentile_extent_filter_applied": bool(not BEV_USE_FULL_POINT_CLOUD),
        "spatial_cap_applied": bool(spatial_cap_applied),
        "orthophoto_postprocess": postprocess_stats,
    }


def pixels_to_utm(points_px: np.ndarray, origin_e: float, origin_n: float, gsd: float) -> np.ndarray:
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    return np.column_stack([origin_e + pts[:, 0] * gsd, origin_n - pts[:, 1] * gsd])


def mask_bbox_center_xy(mask: np.ndarray | None, fallback_shape: tuple[int, ...]) -> tuple[float, float]:
    if mask is not None and np.any(mask > 127):
        x, y, w, h = cv2.boundingRect((mask > 127).astype(np.uint8))
        return float(x + w * 0.5), float(y + h * 0.5)
    h, w = fallback_shape[:2]
    return float(w * 0.5), float(h * 0.5)


def save_rgb_point_cloud_ply(path: Path, points_xyz: np.ndarray, colors_rgb: np.ndarray) -> int:
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    colors = np.asarray(colors_rgb, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError(f"Point/color count mismatch: {len(points)} != {len(colors)}")
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    colors = colors[finite]
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        file.write(header)
        vertices.tofile(file)
    return int(len(vertices))


def merge_utm_point_clouds_without_alignment(
    input_paths: list[Path],
    output_path: Path,
) -> int:
    """Concatenate same-layout binary PLYs; coordinates are left untouched."""
    payloads: list[bytes] = []
    total_vertices = 0
    for path in input_paths:
        if not path.exists():
            continue
        raw = path.read_bytes()
        header_end = raw.find(b"end_header\n")
        if header_end < 0:
            raise ValueError(f"Invalid PLY header: {path}")
        header_end += len(b"end_header\n")
        header = raw[:header_end].decode("ascii")
        count_line = next((line for line in header.splitlines() if line.startswith("element vertex ")), None)
        if count_line is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        total_vertices += int(count_line.split()[-1])
        payloads.append(raw[header_end:])
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {total_vertices}\n"
        "property double x\nproperty double y\nproperty double z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    with open(output_path, "wb") as file:
        file.write(header)
        for payload in payloads:
            file.write(payload)
    return int(total_vertices)


def save_matches(image0: np.ndarray, image1: np.ndarray, p0: np.ndarray, p1: np.ndarray, path: Path) -> None:
    if len(p0) == 0:
        return
    count = min(len(p0), 400)
    k0 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in p0[:count]]
    k1 = [cv2.KeyPoint(float(x), float(y), 1) for x, y in p1[:count]]
    matches = [cv2.DMatch(i, i, 0) for i in range(count)]
    vis = cv2.drawMatches(image0, k0, image1, k1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(str(path), vis)


def save_crop_alignment_debug(
    satellite: np.ndarray,
    sat_origin_e: float,
    sat_origin_n: float,
    sat_gsd: float,
    bev_meta: dict[str, Any],
    path: Path,
    bev_bgr: np.ndarray | None = None,
    bev_mask: np.ndarray | None = None,
    residual_sim2: np.ndarray | None = None,
    residual_accepted: bool | None = None,
    overlay_alpha: float = SATELLITE_BEV_OVERLAY_ALPHA,
) -> dict[str, Any]:
    debug = satellite.copy()
    if debug.ndim == 2:
        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)

    h, w = debug.shape[:2]

    def to_px(e: float, n: float) -> tuple[int, int]:
        return int(round((e - sat_origin_e) / sat_gsd)), int(round((sat_origin_n - n) / sat_gsd))

    bounds = bev_meta["bounds_utm"]
    overlay_applied = False
    overlay_affine = None
    corrected_footprint_px = None
    corrected_footprint_center_px = None
    corrected_camera_px = None
    if residual_sim2 is not None and bev_bgr is not None and bev_mask is not None:
        residual_matrix = np.asarray(residual_sim2, dtype=np.float64).reshape(2, 3)
        bev_h, bev_w = bev_bgr.shape[:2]
        source_px = np.asarray(
            [[0.0, 0.0], [float(bev_w - 1), 0.0], [0.0, float(bev_h - 1)]],
            dtype=np.float32,
        )
        source_utm = pixels_to_utm(
            source_px,
            float(bev_meta["origin_e"]),
            float(bev_meta["origin_n"]),
            float(bev_meta["gsd"]),
        )
        corrected_utm = apply_sim2(residual_matrix, source_utm)
        destination_px = np.column_stack([
            (corrected_utm[:, 0] - float(sat_origin_e)) / float(sat_gsd),
            (float(sat_origin_n) - corrected_utm[:, 1]) / float(sat_gsd),
        ]).astype(np.float32)
        overlay_affine = cv2.getAffineTransform(source_px, destination_px)
        warped_bev = cv2.warpAffine(
            bev_bgr,
            overlay_affine,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            (np.asarray(bev_mask) > 127).astype(np.uint8) * 255,
            overlay_affine,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_overlay = warped_mask > 127
        if np.any(valid_overlay):
            alpha = float(np.clip(overlay_alpha, 0.0, 1.0))
            blended = cv2.addWeighted(debug, 1.0 - alpha, warped_bev, alpha, 0.0)
            debug[valid_overlay] = blended[valid_overlay]
            overlay_applied = True
            cv2.imwrite(str(path.with_name("bev_sim2_warped_to_satellite.png")), warped_bev)
            cv2.imwrite(str(path.with_name("bev_sim2_warped_mask.png")), warped_mask)

        footprint_corners_utm = np.asarray([
            [bounds["min_e"], bounds["max_n"]],
            [bounds["max_e"], bounds["max_n"]],
            [bounds["max_e"], bounds["min_n"]],
            [bounds["min_e"], bounds["min_n"]],
        ], dtype=np.float64)
        corrected_corners_utm = apply_sim2(residual_matrix, footprint_corners_utm)
        corrected_footprint_px = np.asarray([
            to_px(float(e), float(n)) for e, n in corrected_corners_utm
        ], dtype=np.int32)
        corrected_footprint_center = apply_sim2(
            residual_matrix,
            np.asarray(bev_meta["footprint_center_utm"], dtype=np.float64).reshape(1, 2),
        )[0]
        corrected_camera = apply_sim2(
            residual_matrix,
            np.asarray(bev_meta["camera_center_utm"], dtype=np.float64).reshape(1, 2),
        )[0]
        corrected_footprint_center_px = to_px(*corrected_footprint_center)
        corrected_camera_px = to_px(*corrected_camera)

    top_left = to_px(bounds["min_e"], bounds["max_n"])
    bottom_right = to_px(bounds["max_e"], bounds["min_n"])
    cv2.rectangle(debug, top_left, bottom_right, (0, 255, 255), 3)
    if corrected_footprint_px is not None:
        corrected_color = (0, 255, 0) if residual_accepted else (0, 165, 255)
        cv2.polylines(debug, [corrected_footprint_px], True, corrected_color, 4, cv2.LINE_AA)
        cv2.drawMarker(
            debug, corrected_footprint_center_px, corrected_color,
            cv2.MARKER_CROSS, 30, 3,
        )
        cv2.drawMarker(
            debug, corrected_camera_px, corrected_color,
            cv2.MARKER_TILTED_CROSS, 30, 3,
        )
        status = "accepted" if residual_accepted else "candidate/rejected"
        cv2.putText(
            debug,
            f"matched Sim(2) BEV overlay ({status}, alpha={float(overlay_alpha):.2f})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            corrected_color,
            2,
            cv2.LINE_AA,
        )

    footprint_px = to_px(*bev_meta["footprint_center_utm"])
    camera_px = to_px(*bev_meta["camera_center_utm"])
    pitch_prior = bev_meta.get("middle_frame_pitch_crop_prior") or {}
    pitch_center = pitch_prior.get("center_utm")
    pitch_center_px = to_px(*pitch_center) if pitch_center is not None else None
    cv2.drawMarker(debug, footprint_px, (0, 255, 255), cv2.MARKER_CROSS, 28, 3)
    cv2.drawMarker(debug, camera_px, (255, 255, 0), cv2.MARKER_TILTED_CROSS, 28, 3)
    cv2.putText(debug, "BEV footprint center", (footprint_px[0] + 10, footprint_px[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(debug, "camera center", (camera_px[0] + 10, camera_px[1] + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    if pitch_center_px is not None:
        cv2.drawMarker(debug, pitch_center_px, (0, 255, 0), cv2.MARKER_STAR, 32, 3)
        cv2.putText(debug, "middle-frame pitch crop center", (pitch_center_px[0] + 10, pitch_center_px[1] + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    arrow_x = max(35, w - 55)
    cv2.arrowedLine(debug, (arrow_x, 75), (arrow_x, 25), (0, 0, 255), 3, tipLength=0.25)
    cv2.putText(debug, "N", (arrow_x - 10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(debug, "north-up; prior Sim(2) already applied", (20, max(30, h - 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), debug)

    crop_min_e = float(sat_origin_e)
    crop_max_e = float(sat_origin_e + w * sat_gsd)
    crop_max_n = float(sat_origin_n)
    crop_min_n = float(sat_origin_n - h * sat_gsd)
    footprint_inside = (
        bounds["min_e"] >= crop_min_e and bounds["max_e"] <= crop_max_e
        and bounds["min_n"] >= crop_min_n and bounds["max_n"] <= crop_max_n
    )
    return {
        "satellite_crop_bounds_utm": {
            "min_e": crop_min_e,
            "max_e": crop_max_e,
            "min_n": crop_min_n,
            "max_n": crop_max_n,
        },
        "bev_footprint_inside_satellite_crop": bool(footprint_inside),
        "bev_footprint_center_px": [int(footprint_px[0]), int(footprint_px[1])],
        "camera_center_px": [int(camera_px[0]), int(camera_px[1])],
        "middle_frame_pitch_crop_center_px": (
            [int(pitch_center_px[0]), int(pitch_center_px[1])]
            if pitch_center_px is not None else None
        ),
        "matched_sim2_bev_overlay_applied": bool(overlay_applied),
        "matched_sim2_bev_overlay_alpha": float(overlay_alpha),
        "matched_sim2_candidate_accepted": (
            None if residual_accepted is None else bool(residual_accepted)
        ),
        "matched_sim2_bev_to_satellite_affine_px": (
            overlay_affine.tolist() if overlay_affine is not None else None
        ),
        "corrected_bev_footprint_px": (
            corrected_footprint_px.tolist() if corrected_footprint_px is not None else None
        ),
    }


def crop_satellite_to_bev_footprint(
    ref_map_full: np.ndarray,
    region_config: dict[str, Any],
    bounds_utm: dict[str, float],
    extra_margin_m: float = 0.0,
) -> tuple[np.ndarray, float, float, float, tuple[int, int], bool, dict[str, float]]:
    resolution = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_resolution"])
    init_e = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_initialX"])
    init_n = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_initialY"])
    if not np.isfinite(SATELLITE_FOOTPRINT_EXPAND_M) or SATELLITE_FOOTPRINT_EXPAND_M < 0.0:
        raise ValueError(
            "SATELLITE_FOOTPRINT_EXPAND_M must be finite and non-negative, "
            f"got {SATELLITE_FOOTPRINT_EXPAND_M}"
        )
    margin = float(SATELLITE_FOOTPRINT_EXPAND_M) + max(0.0, float(extra_margin_m))
    min_e = float(bounds_utm["min_e"]) - margin
    max_e = float(bounds_utm["max_e"]) + margin
    min_n = float(bounds_utm["min_n"]) - margin
    max_n = float(bounds_utm["max_n"]) + margin
    map_h, map_w = ref_map_full.shape[:2]
    col1_raw = int(math.floor((min_e - init_e) / resolution))
    col2_raw = int(math.ceil((max_e - init_e) / resolution)) + 1
    row1_raw = int(math.floor((init_n - max_n) / resolution))
    row2_raw = int(math.ceil((init_n - min_n) / resolution)) + 1
    col1 = int(np.clip(col1_raw, 0, map_w))
    col2 = int(np.clip(col2_raw, 0, map_w))
    row1 = int(np.clip(row1_raw, 0, map_h))
    row2 = int(np.clip(row2_raw, 0, map_h))
    if col2 <= col1 or row2 <= row1:
        raise ValueError("BEV-footprint satellite crop is empty")
    crop = ref_map_full[row1:row2, col1:col2].copy()
    crop_origin_e = init_e + col1 * resolution
    crop_origin_n = init_n - row1 * resolution
    center_e = 0.5 * (float(bounds_utm["min_e"]) + float(bounds_utm["max_e"]))
    center_n = 0.5 * (float(bounds_utm["min_n"]) + float(bounds_utm["max_n"]))
    center_px = (
        int(round((center_e - crop_origin_e) / resolution)),
        int(round((crop_origin_n - center_n) / resolution)),
    )
    fully_inside = bool(
        col1_raw >= 0 and row1_raw >= 0 and col2_raw <= map_w and row2_raw <= map_h
    )
    crop_bounds = {
        "min_e": float(crop_origin_e),
        "max_e": float(crop_origin_e + crop.shape[1] * resolution),
        "min_n": float(crop_origin_n - crop.shape[0] * resolution),
        "max_n": float(crop_origin_n),
        "width_m": float(crop.shape[1] * resolution),
        "height_m": float(crop.shape[0] * resolution),
        "expand_each_side_m": margin,
    }
    return crop, crop_origin_e, crop_origin_n, resolution, center_px, fully_inside, crop_bounds


def match_bev_anchor(
    legacy: Any,
    matcher: Any,
    bev_bgr: np.ndarray,
    bev_mask: np.ndarray,
    bev_meta: dict[str, Any],
    ref_map_full: np.ndarray,
    region_config: dict[str, Any],
    out_dir: Path,
    satellite_crop_prior: dict[str, Any] | None = None,
    save_visualizations: bool = True,
    extra_margin_m: float = 0.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    from scalloc.matchers.minima import MINIMA_match

    if save_visualizations:
        out_dir.mkdir(parents=True, exist_ok=True)
    bounds = bev_meta["bounds_utm"]
    center_e = 0.5 * (float(bounds["min_e"]) + float(bounds["max_e"]))
    center_n = 0.5 * (float(bounds["min_n"]) + float(bounds["max_n"]))
    center_source = "bev_orthophoto_footprint_center"
    sat, sat_origin_e, sat_origin_n, sat_gsd, center_px, inside, sat_crop_bounds = (
        crop_satellite_to_bev_footprint(
            ref_map_full, region_config, bounds, extra_margin_m=extra_margin_m
        )
    )
    bev_input = legacy.make_raw_thermal_bgr(bev_bgr, bev_mask)
    sat_input = legacy.make_satellite_thermal_gray(sat)
    if save_visualizations:
        cv2.imwrite(str(out_dir / "bev_prior.png"), bev_bgr)
        cv2.imwrite(str(out_dir / "bev_mask.png"), bev_mask)
    if save_visualizations:
        cv2.imwrite(str(out_dir / "satellite_crop.jpg"), sat)
        cv2.imwrite(str(out_dir / "bev_match_input.jpg"), bev_input)
        cv2.imwrite(str(out_dir / "satellite_match_input.jpg"), sat_input)
        crop_debug = save_crop_alignment_debug(
            sat, sat_origin_e, sat_origin_n, sat_gsd, bev_meta, out_dir / "satellite_crop_debug.jpg"
        )
        save_crop_alignment_debug(
            sat,
            sat_origin_e,
            sat_origin_n,
            sat_gsd,
            bev_meta,
            out_dir / "satellite_crop_debug_prior.jpg",
        )
    else:
        crop_debug = {}
    raw0, raw1, raw_conf = MINIMA_match(
        bev_input, sat_input, matcher, str(out_dir), "minima_raw_matches.jpg", need_ransac=False, show_matches=False
    )
    p0, p1, conf = [], [], []
    h0, w0 = bev_input.shape[:2]
    h1, w1 = sat_input.shape[:2]
    for a, b, c in zip(raw0, raw1, raw_conf):
        x0, y0 = int(round(float(a[0]))), int(round(float(a[1])))
        x1, y1 = int(round(float(b[0]))), int(round(float(b[1])))
        if float(c) < MINIMA_MIN_CONF:
            continue
        if not (0 <= x0 < w0 and 0 <= y0 < h0 and 0 <= x1 < w1 and 0 <= y1 < h1):
            continue
        if bev_mask[y0, x0] <= 127:
            continue
        p0.append(a)
        p1.append(b)
        conf.append(float(c))
    p0 = np.asarray(p0, dtype=np.float64).reshape(-1, 2)
    p1 = np.asarray(p1, dtype=np.float64).reshape(-1, 2)
    conf = np.asarray(conf, dtype=np.float64)
    if save_visualizations:
        save_matches(bev_input, sat_input, p0, p1, out_dir / "minima_filtered_matches.jpg")
    if len(p0) < MINIMA_MIN_MATCHES:
        return None, {"accepted": False, "reason": "too_few_matches", "valid_matches": int(len(p0))}
    src_utm = pixels_to_utm(p0, bev_meta["origin_e"], bev_meta["origin_n"], bev_meta["gsd"])
    dst_utm = pixels_to_utm(p1, sat_origin_e, sat_origin_n, sat_gsd)
    residual, inliers, stats = similarity_ransac(
        src_utm, dst_utm, conf, MINIMA_RANSAC_THRESHOLD_M, MINIMA_MIN_MATCHES
    )
    if save_visualizations and residual is not None:
        save_matches(bev_input, sat_input, p0[inliers], p1[inliers], out_dir / "minima_inlier_matches.jpg")
    accepted = residual is not None
    reasons = []
    scale_observation_only = False
    if not accepted:
        reasons.append(str(stats.get("reason", "similarity_failed")))
    else:
        if (
            MINIMA_INLIER_RATIO_GATE_ENABLED
            and float(stats.get("inlier_ratio", 0.0)) < MINIMA_MIN_INLIER_RATIO
        ):
            reasons.append("low_inlier_ratio")
        if float(stats.get("rmse", float("inf"))) > MINIMA_MAX_RMSE_M:
            reasons.append("high_rmse")
        scale = float(stats.get("scale", 0.0))
        normal_scale = MINIMA_RESIDUAL_SCALE_MIN <= scale <= MINIMA_RESIDUAL_SCALE_MAX
        scale_observation_range = MINIMA_SCALE_OBSERVATION_MIN <= scale <= MINIMA_SCALE_OBSERVATION_MAX
        if not normal_scale:
            if not reasons and scale_observation_range:
                scale_observation_only = True
            else:
                reasons.append("residual_scale_out_of_range")
    accepted = accepted and not reasons and not scale_observation_only
    if save_visualizations:
        crop_debug = save_crop_alignment_debug(
            sat,
            sat_origin_e,
            sat_origin_n,
            sat_gsd,
            bev_meta,
            out_dir / "satellite_crop_debug.jpg",
            bev_bgr=bev_input,
            bev_mask=bev_mask,
            residual_sim2=residual,
            residual_accepted=bool(accepted or scale_observation_only),
            overlay_alpha=SATELLITE_BEV_OVERLAY_ALPHA,
        )
    summary = {
        "accepted": bool(accepted),
        "scale_observation_only": bool(scale_observation_only),
        "scale_observation_range_min": float(MINIMA_SCALE_OBSERVATION_MIN),
        "scale_observation_range_max": float(MINIMA_SCALE_OBSERVATION_MAX),
        "reason": "scale_observation_only" if scale_observation_only else ";".join(reasons),
        "raw_matches": int(len(raw0)),
        "valid_matches": int(len(p0)),
        "satellite_crop_center_utm": [float(center_e), float(center_n)],
        "satellite_crop_center_source": center_source,
        "satellite_crop_width_m": float(sat_crop_bounds["width_m"]),
        "satellite_crop_height_m": float(sat_crop_bounds["height_m"]),
        "satellite_footprint_expand_each_side_m": float(
            sat_crop_bounds["expand_each_side_m"]
        ),
        "bev_rescue_extra_margin_m": float(max(0.0, extra_margin_m)),
        "satellite_crop_bounds_utm": sat_crop_bounds,
        "middle_frame_pitch_crop_prior": satellite_crop_prior,
        "satellite_center_inside": bool(inside),
        "satellite_projected_center_px": [int(center_px[0]), int(center_px[1])],
        "minima_inlier_ratio_gate_enabled": bool(MINIMA_INLIER_RATIO_GATE_ENABLED),
        "minima_min_inlier_ratio": float(MINIMA_MIN_INLIER_RATIO),
        "minima_max_rmse_m": float(MINIMA_MAX_RMSE_M),
        "bev_orientation": str(bev_meta.get("orientation", "unknown")),
        "prior_sim2_theta_deg": float(bev_meta.get("prior_sim2_theta_deg", float("nan"))),
        "camera_to_footprint_center_distance_m": float(
            bev_meta.get("camera_to_footprint_center_distance_m", float("nan"))
        ),
        **crop_debug,
        **stats,
    }
    if residual is not None:
        summary["raw_residual_sim2"] = residual.tolist()
    # The caller distinguishes a full pose anchor from a scale-only
    # observation using summary['scale_observation_only'].
    return residual if (accepted or scale_observation_only) else None, summary


def _template_structure_gray(img: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if valid_mask is not None and np.any(valid_mask > 127):
        mask = valid_mask > 127
        gray = gray.copy()
        gray[~mask] = int(np.median(gray[mask]))
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    max_val = float(mag.max())
    if max_val <= 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / max_val).astype(np.float32)


def _rotate_template_expanded(img: np.ndarray, mask: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    center = (w * 0.5, h * 0.5)
    matrix = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cos_v = abs(matrix[0, 0])
    sin_v = abs(matrix[0, 1])
    new_w = int(round(h * sin_v + w * cos_v))
    new_h = int(round(h * cos_v + w * sin_v))
    matrix[0, 2] += new_w * 0.5 - center[0]
    matrix[1, 2] += new_h * 0.5 - center[1]
    rotated = cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_mask = cv2.warpAffine(
        mask,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated, rotated_mask, matrix


def _extract_template_peaks(score_map: np.ndarray, topk: int, nms_radius_px: int) -> list[dict[str, float]]:
    score = np.asarray(score_map, dtype=np.float32).copy()
    peaks: list[dict[str, float]] = []
    radius = max(1, int(nms_radius_px))
    for _ in range(max(1, int(topk))):
        _, max_val, _, max_loc = cv2.minMaxLoc(score)
        if not np.isfinite(max_val) or max_val <= -1e5:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        y1, y2 = max(0, y - radius), min(score.shape[0], y + radius + 1)
        x1, x2 = max(0, x - radius), min(score.shape[1], x + radius + 1)
        local = np.asarray(score_map[y1:y2, x1:x2], dtype=np.float32)
        local_mean = float(local.mean()) if local.size else float(max_val)
        work = score.copy()
        work[y1:y2, x1:x2] = -np.inf
        next_val = float(np.nanmax(work)) if np.any(np.isfinite(work)) else float(max_val)
        peaks.append({
            "x": float(x),
            "y": float(y),
            "score": float(max_val),
            "peak_margin": float(max_val - next_val),
            "peak_sharpness": float(max_val - local_mean),
        })
        score[y1:y2, x1:x2] = -np.inf
    return peaks


def _save_template_anchor_overlay(
    sat: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
    top_left_xy: tuple[float, float],
    center_px: tuple[int, int],
    matched_center_px: tuple[int, int],
    out_path: Path,
    alpha: float = 0.55,
) -> None:
    vis = sat.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    tmpl = template
    if tmpl.ndim == 2:
        tmpl = cv2.cvtColor(tmpl, cv2.COLOR_GRAY2BGR)
    x0 = int(round(float(top_left_xy[0])))
    y0 = int(round(float(top_left_xy[1])))
    x1, y1 = max(0, x0), max(0, y0)
    x2 = min(vis.shape[1], x0 + tmpl.shape[1])
    y2 = min(vis.shape[0], y0 + tmpl.shape[0])
    if x2 > x1 and y2 > y1:
        sx1 = x1 - x0
        sy1 = y1 - y0
        sx2 = sx1 + (x2 - x1)
        sy2 = sy1 + (y2 - y1)
        mask_crop = (mask[sy1:sy2, sx1:sx2] > 127).astype(np.float32)
        mask_crop = cv2.GaussianBlur(mask_crop, (0, 0), sigmaX=1.2)
        a = (mask_crop * float(alpha))[..., None]
        roi = vis[y1:y2, x1:x2].astype(np.float32)
        blend = roi * (1.0 - a) + tmpl[sy1:sy2, sx1:sx2].astype(np.float32) * a
        vis[y1:y2, x1:x2] = np.clip(blend, 0, 255).astype(np.uint8)
    cv2.drawMarker(vis, center_px, (0, 140, 255), cv2.MARKER_CROSS, 42, 4, cv2.LINE_AA)
    cv2.putText(vis, "crop center", (center_px[0] + 10, center_px[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 140, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(vis, matched_center_px, (255, 255, 0), cv2.MARKER_STAR, 42, 4, cv2.LINE_AA)
    cv2.putText(vis, "template matched center", (matched_center_px[0] + 10, matched_center_px[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), vis)


def cleanup_template_anchor_outputs(
    out_dir: Path,
    save_visualizations: bool,
) -> list[str]:
    if save_visualizations:
        return []
    keep_names = {
        "template_optimizer_bev.png",
        "template_optimizer_bev_mask.png",
        "template_optimizer_bev_meta.json",
        "satellite_crop.jpg",
        "match_result_overlay.jpg",
        "match_result.json",
        "refined_candidates.json",
        "window_result.json",
        "pointcloud_local.ply",
        "pointcloud_utm.ply",
        "window_gt_vggt_attitude.csv",
        "window_vggt_intrinsics.csv",
    }
    removable_suffixes = {".jpg", ".jpeg", ".png", ".npy", ".npz"}
    removed: list[str] = []
    for path in out_dir.iterdir():
        if not path.is_file() or path.name in keep_names:
            continue
        if path.suffix.lower() not in removable_suffixes:
            continue
        try:
            path.unlink()
            removed.append(path.name)
        except OSError:
            pass
    return removed


def match_bev_anchor_template(
    legacy: Any,
    bev_bgr: np.ndarray,
    bev_mask: np.ndarray,
    bev_meta: dict[str, Any],
    ref_map_full: np.ndarray,
    region_config: dict[str, Any],
    out_dir: Path,
    satellite_crop_prior: dict[str, Any] | None = None,
    template_module: Any | None = None,
    dinov2_model: Any | None = None,
    device: torch.device | None = None,
    save_visualizations: bool = True,
    extra_margin_m: float = 0.0,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if template_module is None:
        template_module = load_module_from_path(
            "template_match_optimizer_peakshape_runtime",
            template_match_optimizer_path(),
        )
    bounds = bev_meta["bounds_utm"]
    center_e = 0.5 * (float(bounds["min_e"]) + float(bounds["max_e"]))
    center_n = 0.5 * (float(bounds["min_n"]) + float(bounds["max_n"]))
    src_center_px = mask_bbox_center_xy(bev_mask, bev_bgr.shape)
    src_center_utm = pixels_to_utm(
        np.asarray([src_center_px], dtype=np.float64),
        float(bev_meta["origin_e"]),
        float(bev_meta["origin_n"]),
        float(bev_meta["gsd"]),
    )[0]

    template_ortho_path = out_dir / "template_optimizer_bev.png"
    template_mask_path = out_dir / "template_optimizer_bev_mask.png"
    cv2.imwrite(str(template_ortho_path), bev_bgr)
    cv2.imwrite(str(template_mask_path), bev_mask)
    with open(out_dir / "template_optimizer_bev_meta.json", "w", encoding="utf-8") as file:
        json.dump({
            "gsd_m": float(bev_meta["gsd"]),
            "orientation": str(bev_meta.get("orientation", "north_up_utm")),
            "origin_e": float(bev_meta["origin_e"]),
            "origin_n": float(bev_meta["origin_n"]),
        }, file, indent=2, ensure_ascii=False)
    cv2.imwrite(str(out_dir / "bev_prior.png"), bev_bgr)
    cv2.imwrite(str(out_dir / "bev_mask.png"), bev_mask)

    sat_gsd = float(region_config[f"{REGION_NAME}_{REF_TYPE}_REF_resolution"])
    # The rasterized BEV template is already north-up in UTM coordinates.
    yaw_prior = 0.0
    sat, sat_origin_e, sat_origin_n, sat_gsd, center_px, inside, sat_crop_bounds = (
        crop_satellite_to_bev_footprint(
            ref_map_full,
            region_config,
            bounds,
            extra_margin_m=extra_margin_m,
        )
    )
    # Same crop as the MINIMA branch. The template optimizer receives the
    # already-cropped satellite patch; search_margin_m is kept only for its
    # internal radius bookkeeping and result metadata.
    search_margin_m = max(
        float(sat_crop_bounds["width_m"]),
        float(sat_crop_bounds["height_m"]),
        32.0 * sat_gsd,
    )
    args = SimpleNamespace(
        yaml="config.yaml",
        dataset_root="",
        metadata_root="",
        metadata_json="",
        split="",
        modality="Thermal",
        place="",
        image_index=0,
        single_image=True,
        test_interval=0,
        uav_path="",
        run_dir="",
        ortho_path=str(template_ortho_path),
        global_pose_json="",
        csv_path="",
        output_dir=str(out_dir),
        _output_dir_timestamped=True,
        region=REGION_NAME,
        ref_type=REF_TYPE,
        center_utm_x=float(center_e),
        center_utm_y=float(center_n),
        yaw_deg=float(yaw_prior),
        no_yaw_prior=False,
        noise_yaw=0.0,
        crop_center_offset_range_m=0.0,
        seed=GLOBAL_RANDOM_SEED,
        project_tail_center=False,
        search_margin_m=float(search_margin_m),
        dino_repo="DINOv2",
        dino_model="dinov2_vitb14",
        dino_downsample_factor=1.0,
        dino_chunk_size=1008,
        dino_chunk_overlap_ratio=0.3,
        pca_dim=8,
        sr_pca_sample_stride=1,
        ref_preprocess="satellite_thermal_gray",
        template_preprocess="raw_thermal",
        save_preprocess_debug=bool(save_visualizations),
        rotate_feature_kernel=True,
        yaw_min=float(TEMPLATE_YAW_MIN_DEG),
        yaw_max=float(TEMPLATE_YAW_MAX_DEG),
        yaw_step=float(TEMPLATE_YAW_STEP_DEG),
        fine_yaw_radius=3.0,
        fine_yaw_step=0.75,
        fine_search_radius_m=float(max(120.0, search_margin_m)),
        fine_refine=True,
        ncc_mode="same",
        ncc_min_overlap_ratio=0.25,
        topk_per_yaw=int(TEMPLATE_TOPK_PER_YAW),
        coarse_topm=20,
        nms_radius_m=float(TEMPLATE_NMS_RADIUS_M),
        reliability_margin_weight=0.2,
        peak_shape_rerank=True,
        peak_shape_window_radius_m=80.0,
        peak_shape_delta=0.03,
        peak_shape_area_weight=0.05,
        peak_shape_anisotropy_weight=0.03,
        peak_shape_curvature_weight=0.10,
        peak_shape_contrast_weight=0.0,
        overlay_alpha=float(SATELLITE_BEV_OVERLAY_ALPHA),
        show_visuals=True,
        pose_meta=None,
        region_config=region_config,
        ref_map_full=ref_map_full,
        sat_gsd=sat_gsd,
        satellite_crop_override=sat,
        satellite_crop_origin_utm=[float(sat_origin_e), float(sat_origin_n)],
        satellite_crop_gsd=float(sat_gsd),
        satellite_crop_center_px=[int(center_px[0]), int(center_px[1])],
        device_obj=device,
        dinov2_model=dinov2_model,
        selected_uav_path=str(template_ortho_path),
        selected_csv_path="",
        selected_split="",
        selected_modality="BEV",
        selected_place="VGGT_WINDOW",
        run_timestamp="",
    )
    try:
        tm_result = template_module.run_match(args)
    except Exception as exc:
        removed_debug_files = cleanup_template_anchor_outputs(
            out_dir, save_visualizations
        )
        summary = {
            "accepted": False,
            "reason": f"template_optimizer_exception:{exc}",
            "match_method": "template_optimizer_peakshape",
            "satellite_crop_center_utm": [float(center_e), float(center_n)],
            "middle_frame_pitch_crop_prior": satellite_crop_prior,
            "template_debug_cleanup": {
                "full_debug_outputs": bool(save_visualizations),
                "removed_files": removed_debug_files,
            },
        }
        return None, summary
    removed_debug_files = cleanup_template_anchor_outputs(
        out_dir, save_visualizations
    )

    matched_center = tm_result.get("matched_pointcloud_center_utm")
    if matched_center is None:
        matched_center = tm_result.get("center_utm")
    if matched_center is None:
        summary = {
            "accepted": False,
            "reason": "template_optimizer_no_matched_center",
            "match_method": "template_optimizer_peakshape",
            "template_optimizer_result": tm_result,
            "satellite_crop_center_utm": [float(center_e), float(center_n)],
            "middle_frame_pitch_crop_prior": satellite_crop_prior,
            "template_debug_cleanup": {
                "full_debug_outputs": bool(save_visualizations),
                "removed_files": removed_debug_files,
            },
        }
        return None, summary

    matched_center_e = float(matched_center[0])
    matched_center_n = float(matched_center[1])
    template_yaw_correction_deg = float(tm_result.get("yaw_correction_deg", 0.0))
    image_yaw_offset_deg = float(
        tm_result.get(
            "best_image_rotation_offset_deg",
            tm_result.get("best_yaw_offset_deg", -template_yaw_correction_deg),
        )
    )
    # The template matcher reports image/yaw correction in UAV yaw convention.
    # The EN-plane Sim(2) matrix uses the opposite positive rotation convention
    # for `yaw_from_forward()`, so only the template branch converts the sign here.
    sim2_theta_deg = -template_yaw_correction_deg
    theta = math.radians(sim2_theta_deg)
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]],
        dtype=np.float64,
    )
    src_center = np.asarray(src_center_utm, dtype=np.float64)
    dst_center = np.array([matched_center_e, matched_center_n], dtype=np.float64)
    residual = np.zeros((2, 3), dtype=np.float64)
    residual[:, :2] = rotation
    residual[:, 2] = dst_center - rotation @ src_center

    crop_debug = {}
    sat_path = out_dir / "satellite_crop.jpg"
    sat = cv2.imread(str(sat_path)) if save_visualizations else None
    if sat is not None:
        sat_origin = tm_result.get("crop_origin_utm")
        if sat_origin is not None:
            crop_debug = save_crop_alignment_debug(
                sat,
                float(sat_origin[0]),
                float(sat_origin[1]),
                sat_gsd,
                bev_meta,
                out_dir / "satellite_crop_debug.jpg",
                bev_bgr=bev_bgr,
                bev_mask=bev_mask,
                residual_sim2=residual,
                residual_accepted=True,
                overlay_alpha=SATELLITE_BEV_OVERLAY_ALPHA,
            )

    main_score = float(tm_result.get("main_score", tm_result.get("dino_score", float("nan"))))
    accepted = bool(np.isfinite(main_score) and main_score >= float(TEMPLATE_MIN_NCC_SCORE))
    summary = {
        "accepted": bool(accepted),
        "reason": "" if accepted else "template_score_too_low",
        "match_method": "template_optimizer_peakshape",
        "raw_matches": int(tm_result.get("multi_hypothesis", {}).get("num_coarse_candidates_total", 0)),
        "valid_matches": int(tm_result.get("multi_hypothesis", {}).get("num_refined_candidates", 0)),
        "inlier_count": int(max(1, round(300.0 * max(0.0, main_score if np.isfinite(main_score) else 0.0)))),
        "inlier_ratio": 1.0,
        "rmse": 1.0,
        "scale": 1.0,
        "theta_deg": float(sim2_theta_deg),
        "template_yaw_correction_deg": float(template_yaw_correction_deg),
        "image_yaw_offset_deg": float(image_yaw_offset_deg),
        "yaw_offset_deg": float(tm_result.get("best_yaw_offset_deg", image_yaw_offset_deg)),
        "ncc_score": main_score,
        "peak_margin": float(tm_result.get("multi_hypothesis", {}).get("best_peak_margin", 0.0)),
        "peak_sharpness": float(tm_result.get("multi_hypothesis", {}).get("best_peak_sharpness", 0.0)),
        "matched_center_utm": [float(matched_center_e), float(matched_center_n)],
        "source_center_utm": [float(src_center[0]), float(src_center[1])],
        "template_top_left_px": tm_result.get("top_left_px"),
        "satellite_crop_center_utm": [float(center_e), float(center_n)],
        "satellite_crop_center_source": "bev_orthophoto_footprint_center",
        "satellite_crop_width_m": float(tm_result.get("satellite_crop_width_m", sat_crop_bounds["width_m"])),
        "satellite_crop_height_m": float(tm_result.get("satellite_crop_height_m", sat_crop_bounds["height_m"])),
        "satellite_footprint_expand_each_side_m": float(
            sat_crop_bounds.get(
                "expand_each_side_m",
                float(SATELLITE_FOOTPRINT_EXPAND_M) + max(0.0, float(extra_margin_m)),
            )
        ),
        "bev_rescue_extra_margin_m": float(max(0.0, extra_margin_m)),
        "satellite_template_square_side_m": float(search_margin_m),
        "satellite_template_size_policy": "same_as_minima_bev_footprint_crop",
        "satellite_crop_bounds_utm": sat_crop_bounds,
        "middle_frame_pitch_crop_prior": satellite_crop_prior,
        "satellite_center_inside": bool(inside),
        "satellite_projected_center_px": [int(center_px[0]), int(center_px[1])],
        "bev_orientation": str(bev_meta.get("orientation", "unknown")),
        "prior_sim2_theta_deg": float(bev_meta.get("prior_sim2_theta_deg", float("nan"))),
        "camera_to_footprint_center_distance_m": float(
            bev_meta.get("camera_to_footprint_center_distance_m", float("nan"))
        ),
        "template_optimizer_result": tm_result,
        "template_debug_cleanup": {
            "full_debug_outputs": bool(save_visualizations),
            "removed_files": removed_debug_files,
        },
        "raw_residual_sim2": residual.tolist(),
        **crop_debug,
    }
    return residual if accepted else None, summary


def make_bev_rescue_snapshot(
    window_idx: int,
    frame_indices: list[int],
    geometry: GroundGeometry,
    prior: np.ndarray,
    bev_residual: np.ndarray,
) -> dict[str, Any]:
    """Represent one BEV hypothesis on its actual global frame indices."""
    candidate = compose_sim2(
        np.asarray(bev_residual, dtype=np.float64).reshape(2, 3),
        np.asarray(prior, dtype=np.float64).reshape(2, 3),
    )
    positions = apply_sim2(candidate, geometry.camera_xy)
    yaws = yaw_from_forward(candidate, geometry.camera_forward_xy)
    residual_scale, residual_yaw = sim2_scale_yaw(bev_residual)
    return {
        "window_idx": int(window_idx),
        "positions_by_frame": {
            int(global_idx): np.asarray(positions[local_idx], dtype=np.float64).copy()
            for local_idx, global_idx in enumerate(frame_indices)
        },
        "yaws_by_frame": {
            int(global_idx): float(yaws[local_idx])
            for local_idx, global_idx in enumerate(frame_indices)
        },
        "residual_scale": float(residual_scale),
        "residual_yaw_deg": float(residual_yaw),
    }


def compare_bev_rescue_snapshots(
    pending: dict[str, Any],
    current: dict[str, Any],
) -> dict[str, Any]:
    """Validate two moving-window BEV hypotheses on their shared frames."""
    pending_positions = pending.get("positions_by_frame", {})
    current_positions = current.get("positions_by_frame", {})
    shared = sorted(set(pending_positions).intersection(current_positions))
    result: dict[str, Any] = {
        "previous_window_idx": int(pending.get("window_idx", -1)),
        "current_window_idx": int(current.get("window_idx", -1)),
        "window_age": int(
            int(current.get("window_idx", -1)) - int(pending.get("window_idx", -1))
        ),
        "overlap_frame_count": int(len(shared)),
        "required_overlap_frame_count": int(BEV_RESCUE_CONFIRM_MIN_OVERLAP_FRAMES),
        "confirmed": False,
    }
    if len(shared) < int(BEV_RESCUE_CONFIRM_MIN_OVERLAP_FRAMES):
        result["reason"] = "insufficient_overlap_frames"
        return result

    position_errors = np.asarray(
        [
            np.linalg.norm(
                np.asarray(pending_positions[index], dtype=np.float64)
                - np.asarray(current_positions[index], dtype=np.float64)
            )
            for index in shared
        ],
        dtype=np.float64,
    )
    pending_yaws = pending.get("yaws_by_frame", {})
    current_yaws = current.get("yaws_by_frame", {})
    yaw_errors = np.asarray(
        [
            abs(wrap_deg(float(pending_yaws[index]) - float(current_yaws[index])))
            for index in shared
            if index in pending_yaws
            and index in current_yaws
            and np.isfinite(float(pending_yaws[index]))
            and np.isfinite(float(current_yaws[index]))
        ],
        dtype=np.float64,
    )
    position_median = float(np.median(position_errors))
    position_p95 = float(np.percentile(position_errors, 95.0))
    yaw_median = float(np.median(yaw_errors)) if len(yaw_errors) else float("inf")
    scale_log_difference = abs(math.log(
        max(float(current.get("residual_scale", 0.0)), 1e-12)
        / max(float(pending.get("residual_scale", 0.0)), 1e-12)
    ))
    confirmed = bool(
        position_median <= float(BEV_RESCUE_CONFIRM_POSITION_M)
        and yaw_median <= float(BEV_RESCUE_CONFIRM_YAW_DEG)
        and scale_log_difference <= float(BEV_RESCUE_CONFIRM_SCALE_LOG)
    )
    result.update({
        "confirmed": confirmed,
        "reason": "confirmed" if confirmed else "cross_window_disagreement",
        "position_median_difference_m": position_median,
        "position_p95_difference_m": position_p95,
        "yaw_median_difference_deg": yaw_median,
        "scale_log_difference": float(scale_log_difference),
        "position_threshold_m": float(BEV_RESCUE_CONFIRM_POSITION_M),
        "yaw_threshold_deg": float(BEV_RESCUE_CONFIRM_YAW_DEG),
        "scale_log_threshold": float(BEV_RESCUE_CONFIRM_SCALE_LOG),
    })
    return result


def build_middle_frame_pitch_crop_prior(
    geometry: GroundGeometry,
    height_scale_m_per_local: float,
    camera_utm: np.ndarray,
    predicted_ypr: list[dict[str, float]],
    frame_names: list[str],
) -> dict[str, Any]:
    middle_idx = len(frame_names) // 2
    ypr = predicted_ypr[middle_idx]
    yaw_deg = float(ypr["yaw_deg"])
    pitch_deg = float(ypr["pitch_deg"])
    height_m = (
        abs(float(geometry.camera_height[middle_idx]))
        * float(height_scale_m_per_local)
    )
    down_deg = abs(pitch_deg)
    valid = bool(
        np.isfinite(yaw_deg)
        and np.isfinite(down_deg)
        and np.isfinite(height_m)
        and 0.1 < down_deg < 89.9
        and height_m > 0.0
    )
    camera_center = np.asarray(camera_utm[middle_idx], dtype=np.float64)
    if valid:
        ground_range_m = height_m / max(math.tan(math.radians(down_deg)), 1e-12)
        ground_range_m = min(float(ground_range_m), float(SATELLITE_MAX_PITCH_GROUND_RANGE_M))
        yaw_rad = math.radians(yaw_deg)
        ground_center = camera_center + np.array([
            math.sin(yaw_rad) * ground_range_m,
            math.cos(yaw_rad) * ground_range_m,
        ], dtype=np.float64)
        source = "middle_frame_vggt_position_yaw_pitch_ground_intersection"
    else:
        ground_range_m = float("nan")
        ground_center = camera_center.copy()
        source = "invalid_middle_frame_pitch_camera_position_fallback"
    return {
        "valid": bool(valid),
        "source": source,
        "middle_local_idx": int(middle_idx),
        "frame": str(frame_names[middle_idx]),
        "camera_utm": camera_center.tolist(),
        "center_utm": ground_center.tolist(),
        "pred_yaw_deg": float(yaw_deg),
        "pred_pitch_deg": float(pitch_deg),
        "pred_roll_deg": float(ypr["roll_deg"]),
        "height_m": float(height_m),
        "height_scale_m_per_local": float(height_scale_m_per_local),
        "height_scale_source": "first_window_height_scale_reused_for_all_windows",
        "ground_range_m": float(ground_range_m) if np.isfinite(ground_range_m) else None,
        "ground_range_cap_m": float(SATELLITE_MAX_PITCH_GROUND_RANGE_M),
    }
