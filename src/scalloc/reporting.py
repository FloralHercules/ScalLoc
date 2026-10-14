"""Reporting from the validated fast ScalLoc pipeline."""
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


def record_stage_timing(
    totals: dict[str, dict[str, float | int]],
    stage: str,
    started_at: float,
    window_row: dict[str, Any] | None = None,
) -> float:
    """Accumulate a coarse pipeline-stage wall time."""
    elapsed = float(time.perf_counter() - started_at)
    stats = totals.setdefault(stage, {"total_sec": 0.0, "call_count": 0})
    stats["total_sec"] = float(stats["total_sec"]) + elapsed
    stats["call_count"] = int(stats["call_count"]) + 1
    if window_row is not None:
        window_row[f"{stage}_sec"] = elapsed
    return elapsed


def build_stage_timing_rows(
    totals: dict[str, dict[str, float | int]],
    sequence_total_sec: float,
    window_count: int,
    frame_count: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    denominator = max(float(sequence_total_sec), 1e-12)
    for stage, stats in totals.items():
        total_sec = float(stats["total_sec"])
        call_count = int(stats["call_count"])
        rows.append({
            "stage": stage,
            "total_sec": total_sec,
            "percent_of_sequence": 100.0 * total_sec / denominator,
            "call_count": call_count,
            "mean_sec_per_call": total_sec / max(call_count, 1),
            "mean_sec_per_window": total_sec / max(int(window_count), 1),
            "mean_sec_per_frame": total_sec / max(int(frame_count), 1),
        })
    measured_sec = float(sum(float(row["total_sec"]) for row in rows))
    unattributed_sec = max(0.0, float(sequence_total_sec) - measured_sec)
    rows.append({
        "stage": "other_unattributed_overhead",
        "total_sec": unattributed_sec,
        "percent_of_sequence": 100.0 * unattributed_sec / denominator,
        "call_count": 1,
        "mean_sec_per_call": unattributed_sec,
        "mean_sec_per_window": unattributed_sec / max(int(window_count), 1),
        "mean_sec_per_frame": unattributed_sec / max(int(frame_count), 1),
    })
    rows.sort(key=lambda row: float(row["total_sec"]), reverse=True)
    return rows


def save_window_attitude_csv(
    path: Path,
    gt_df: pd.DataFrame,
    global_indices: list[int],
    frame_names: list[str],
    predicted_ypr: list[dict[str, float]],
    predicted_height_m: np.ndarray,
) -> int:
    gt_by_stem = {
        Path(str(row["filename"])).stem: row
        for _, row in gt_df.iterrows()
        if "filename" in row and pd.notna(row["filename"])
    }
    rows = []
    for local_idx, (global_idx, frame_name, pred, pred_height) in enumerate(
        zip(global_indices, frame_names, predicted_ypr, predicted_height_m)
    ):
        gt_row = gt_by_stem.get(Path(frame_name).stem)
        if gt_row is None and 0 <= int(global_idx) < len(gt_df):
            gt_row = gt_df.iloc[int(global_idx)]
        rows.append({
            "frame": str(frame_name),
            "gt_yaw_deg": float(gt_row["yaw"]) if gt_row is not None else float("nan"),
            "gt_pitch_deg": float(gt_row["pitch"]) if gt_row is not None else float("nan"),
            "gt_roll_deg": float(gt_row["roll"]) if gt_row is not None else float("nan"),
            "gt_rel_height_m": float(gt_row["rel_height"]) if gt_row is not None else float("nan"),
            "gt_abs_height_m": float(gt_row["abs_height"]) if gt_row is not None else float("nan"),
            "vggt_yaw_deg": float(pred["yaw_deg"]),
            "vggt_pitch_deg": float(pred["pitch_deg"]),
            "vggt_roll_deg": float(pred["roll_deg"]),
            "vggt_height_m": float(pred_height),
        })
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return int(len(rows))


def _vggt_pad_preprocess_affine(
    source_width: int,
    source_height: int,
    target_size: int,
) -> tuple[float, float, int, int, int, int]:
    """Return source->VGGT-pad scaling and padding for the helper's pad mode."""
    if source_width >= source_height:
        resized_width = int(target_size)
        resized_height = int(round(source_height * (resized_width / source_width) / 14) * 14)
    else:
        resized_height = int(target_size)
        resized_width = int(round(source_width * (resized_height / source_height) / 14) * 14)
    pad_left = (int(target_size) - resized_width) // 2
    pad_top = (int(target_size) - resized_height) // 2
    return (
        float(resized_width) / float(source_width),
        float(resized_height) / float(source_height),
        int(pad_left),
        int(pad_top),
        int(resized_width),
        int(resized_height),
    )


def save_window_vggt_intrinsics_csv(
    path: Path,
    image_paths: list[Path],
    frame_names: list[str],
    intrinsics: np.ndarray,
    input_images_hw: tuple[int, int],
    target_size: int,
) -> int:
    """Save VGGT's predicted K in model and original-image pixel coordinates.

    VGGT predicts focal lengths from each frame's FoV and assumes the principal
    point is the centre of its padded model input.  The *_source_px fields undo
    the exact resize/pad transform used by ``load_vggt_images_and_masks``;
    they remain VGGT estimates, not a physical camera calibration.
    """
    matrices = np.asarray(intrinsics, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3):
        raise ValueError(f"Unexpected VGGT intrinsics shape: {matrices.shape}")
    if len(matrices) != len(frame_names) or len(image_paths) != len(frame_names):
        raise ValueError("VGGT intrinsics, paths, and frame names must have equal length")
    input_height, input_width = (int(input_images_hw[0]), int(input_images_hw[1]))
    rows: list[dict[str, Any]] = []
    for image_path, frame_name, matrix in zip(image_paths, frame_names, matrices):
        source = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if source is None:
            source_height = source_width = 0
        else:
            source_height, source_width = (int(source.shape[0]), int(source.shape[1]))
        row: dict[str, Any] = {
            "frame": str(frame_name),
            "source_width_px": int(source_width),
            "source_height_px": int(source_height),
            "vggt_input_width_px": int(input_width),
            "vggt_input_height_px": int(input_height),
            "vggt_fx_input_px": float(matrix[0, 0]),
            "vggt_fy_input_px": float(matrix[1, 1]),
            "vggt_cx_input_px": float(matrix[0, 2]),
            "vggt_cy_input_px": float(matrix[1, 2]),
            "preprocess": "aspect_ratio_resize_to_multiple_of_14_then_center_pad",
        }
        if source_width > 0 and source_height > 0:
            sx, sy, pad_left, pad_top, resized_width, resized_height = _vggt_pad_preprocess_affine(
                source_width, source_height, int(target_size)
            )
            row.update({
                "resize_scale_x": float(sx),
                "resize_scale_y": float(sy),
                "pad_left_px": int(pad_left),
                "pad_top_px": int(pad_top),
                "resized_width_px": int(resized_width),
                "resized_height_px": int(resized_height),
                "vggt_fx_source_px": float(matrix[0, 0] / sx),
                "vggt_fy_source_px": float(matrix[1, 1] / sy),
                "vggt_cx_source_px": float((matrix[0, 2] - pad_left) / sx),
                "vggt_cy_source_px": float((matrix[1, 2] - pad_top) / sy),
            })
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return int(len(rows))


def _metrics_summary_row(metrics: dict[str, Any]) -> dict[str, Any]:
    position = metrics.get("position", {})
    yaw = metrics.get("yaw", {})
    recall = position.get("threshold_recall", {})
    covariance = position.get("delta_east_north_covariance_m2")
    row = {
        "label": metrics.get("label"),
        "sample_count": metrics.get("sample_count"),
        "runtime_sec": metrics.get("runtime_sec"),
        "position_mean_m": position.get("error_mean_m"),
        "position_variance_m2": position.get("error_variance_m2"),
        "position_median_m": position.get("error_median_m"),
        "position_max_m": position.get("error_max_m"),
        "position_p95_m": position.get("error_p95_m"),
        "recall_5m": recall.get("5m"),
        "recall_10m": recall.get("10m"),
        "recall_20m": recall.get("20m"),
        "recall_30m": recall.get("30m"),
        "yaw_sample_count": yaw.get("sample_count"),
        "yaw_signed_mean_deg": yaw.get("signed_error_mean_deg"),
        "yaw_signed_variance_deg2": yaw.get("signed_error_variance_deg2"),
        "yaw_abs_mean_deg": yaw.get("abs_error_mean_deg"),
        "yaw_abs_median_deg": yaw.get("abs_error_median_deg"),
        "yaw_abs_max_deg": yaw.get("abs_error_max_deg"),
        "yaw_abs_p95_deg": yaw.get("abs_error_p95_deg"),
    }
    if isinstance(covariance, list) and len(covariance) == 2:
        row.update({
            "position_cov_ee_m2": covariance[0][0],
            "position_cov_en_m2": covariance[0][1],
            "position_cov_ne_m2": covariance[1][0],
            "position_cov_nn_m2": covariance[1][1],
        })
    return row


def collect_window_pitch_comparison(
    window_summaries: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Fuse per-window VGGT pitch diagnostics into one value per source frame."""
    observations: dict[str, dict[str, list[float]]] = {}
    for summary in window_summaries:
        attitude = summary.get("attitude_comparison", {}) or {}
        csv_value = attitude.get("csv")
        if not csv_value:
            continue
        csv_path = Path(str(csv_value))
        if not csv_path.exists():
            continue
        try:
            table = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            continue
        required = {"frame", "gt_pitch_deg", "vggt_pitch_deg"}
        if not required.issubset(table.columns):
            continue
        for row in table.loc[:, ["frame", "gt_pitch_deg", "vggt_pitch_deg"]].itertuples(index=False):
            frame = str(row.frame)
            record = observations.setdefault(frame, {"gt": [], "vggt": []})
            for key, value in (("gt", row.gt_pitch_deg), ("vggt", row.vggt_pitch_deg)):
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(numeric):
                    record[key].append(numeric)

    fused: dict[str, dict[str, float | int]] = {}
    for frame, values in observations.items():
        gt_values = values["gt"]
        vggt_values = values["vggt"]
        gt_pitch = float(np.median(gt_values)) if gt_values else float("nan")
        vggt_pitch = float(np.median(vggt_values)) if vggt_values else float("nan")
        pitch_error = (
            float(vggt_pitch - gt_pitch)
            if np.isfinite(gt_pitch) and np.isfinite(vggt_pitch)
            else float("nan")
        )
        fused[frame] = {
            "gt_pitch_deg": gt_pitch,
            "vggt_pitch_deg": vggt_pitch,
            "pitch_error_deg": pitch_error,
            "pitch_window_observation_count": int(len(vggt_values)),
        }
    return fused


def pitch_error_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    errors = np.asarray([row.get("pitch_error_deg", float("nan")) for row in rows], dtype=np.float64)
    errors = errors[np.isfinite(errors)]
    if len(errors) == 0:
        return {
            "pitch_sample_count": 0,
            "pitch_signed_mean_deg": float("nan"),
            "pitch_signed_variance_deg2": float("nan"),
            "pitch_abs_mean_deg": float("nan"),
            "pitch_abs_median_deg": float("nan"),
            "pitch_abs_max_deg": float("nan"),
            "pitch_abs_p95_deg": float("nan"),
        }
    abs_errors = np.abs(errors)
    return {
        "pitch_sample_count": int(len(errors)),
        "pitch_signed_mean_deg": float(np.mean(errors)),
        "pitch_signed_variance_deg2": float(np.var(errors)),
        "pitch_abs_mean_deg": float(np.mean(abs_errors)),
        "pitch_abs_median_deg": float(np.median(abs_errors)),
        "pitch_abs_max_deg": float(np.max(abs_errors)),
        "pitch_abs_p95_deg": float(np.percentile(abs_errors, 95)),
    }


def save_comparison_csv_reports(
    uncorrected_metrics: dict[str, Any],
    corrected_metrics: dict[str, Any],
    out_dir: Path,
    pitch_by_frame: dict[str, dict[str, float | int]] | None = None,
) -> tuple[Path, Path]:
    summary_path = out_dir / "trajectory_metrics_summary.csv"
    pitch_by_frame = pitch_by_frame or {}

    uncorrected_by_frame = {
        str(item.get("pred_key")): item for item in uncorrected_metrics.get("per_frame", [])
    }
    corrected_by_frame = {
        str(item.get("pred_key")): item for item in corrected_metrics.get("per_frame", [])
    }
    ordered_frames = list(dict.fromkeys([
        *uncorrected_by_frame.keys(),
        *corrected_by_frame.keys(),
    ]))
    rows = []
    for frame in ordered_frames:
        unc = uncorrected_by_frame.get(frame, {})
        cor = corrected_by_frame.get(frame, {})
        gt = cor.get("gt_utm") or unc.get("gt_utm") or [None, None]
        unc_xy = unc.get("pred_utm") or [None, None]
        cor_xy = cor.get("pred_utm") or [None, None]
        pitch = pitch_by_frame.get(frame, {})
        rows.append({
            "frame": frame,
            "frame_index": cor.get("label_idx", unc.get("label_idx")),
            "gt_easting": gt[0],
            "gt_northing": gt[1],
            "gt_yaw_deg": cor.get("gt_yaw_deg", unc.get("gt_yaw_deg")),
            "gt_pitch_deg": pitch.get("gt_pitch_deg", float("nan")),
            "vggt_pitch_deg": pitch.get("vggt_pitch_deg", float("nan")),
            "pitch_error_deg": pitch.get("pitch_error_deg", float("nan")),
            "pitch_window_observation_count": pitch.get("pitch_window_observation_count", 0),
            "uncorrected_easting": unc_xy[0],
            "uncorrected_northing": unc_xy[1],
            "uncorrected_delta_east_m": unc.get("delta_east_m"),
            "uncorrected_delta_north_m": unc.get("delta_north_m"),
            "uncorrected_position_error_m": unc.get("position_error_m"),
            "uncorrected_yaw_deg": unc.get("pred_yaw_deg"),
            "uncorrected_yaw_error_deg": unc.get("yaw_error_deg"),
            "corrected_easting": cor_xy[0],
            "corrected_northing": cor_xy[1],
            "corrected_delta_east_m": cor.get("delta_east_m"),
            "corrected_delta_north_m": cor.get("delta_north_m"),
            "corrected_position_error_m": cor.get("position_error_m"),
            "corrected_yaw_deg": cor.get("pred_yaw_deg"),
            "corrected_yaw_error_deg": cor.get("yaw_error_deg"),
        })
    per_frame_path = out_dir / "trajectory_per_frame_comparison.csv"
    pd.DataFrame(rows).to_csv(per_frame_path, index=False, encoding="utf-8-sig")
    pitch_summary = pitch_error_summary(rows)
    summary_rows = [
        _metrics_summary_row(uncorrected_metrics),
        _metrics_summary_row(corrected_metrics),
    ]
    # Pitch is inferred before the 2-D satellite correction, so it is shared
    # by corrected and uncorrected trajectory reports.
    for summary in summary_rows:
        summary.update(pitch_summary)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path, per_frame_path


def print_metrics_comparison(uncorrected: dict[str, Any], corrected: dict[str, Any]) -> None:
    def line(title: str, metrics: dict[str, Any]) -> None:
        pos = metrics.get("position", {})
        yaw = metrics.get("yaw", {})
        recall = pos.get("threshold_recall", {})
        print(
            f"{title}: samples={metrics.get('sample_count', 0)}, "
            f"position mean/median/p95/max="
            f"{pos.get('error_mean_m', float('nan')):.3f}/"
            f"{pos.get('error_median_m', float('nan')):.3f}/"
            f"{pos.get('error_p95_m', float('nan')):.3f}/"
            f"{pos.get('error_max_m', float('nan')):.3f} m, "
            f"yaw abs mean={yaw.get('abs_error_mean_deg', float('nan')):.3f} deg, "
            f"R@5/10/20/30="
            f"{100.0 * float(recall.get('5m', 0.0)):.1f}/"
            f"{100.0 * float(recall.get('10m', 0.0)):.1f}/"
            f"{100.0 * float(recall.get('20m', 0.0)):.1f}/"
            f"{100.0 * float(recall.get('30m', 0.0)):.1f}%"
        )

    print("\n================ Trajectory comparison ================")
    line("[NO ANCHOR]", uncorrected)
    line("[CORRECTED]", corrected)
    print("=======================================================")


def _anchor_scalar(value: Any) -> Any:
    if value is None:
        return float("nan")
    if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
        return value
    return float("nan")


def _anchor_xy_fields(prefix: str, value: Any) -> dict[str, float]:
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
        return {
            f"{prefix}_e": float(value[0]),
            f"{prefix}_n": float(value[1]),
        }
    return {f"{prefix}_e": float("nan"), f"{prefix}_n": float("nan")}


def save_anchor_diagnostics_csv(window_summaries: list[dict[str, Any]], out_path: Path) -> int:
    rows: list[dict[str, Any]] = []
    for summary in window_summaries:
        anchor = dict(summary.get("anchor", {}) or {})
        frame_names = summary.get("frame_names", []) or []
        center_local_idx = len(frame_names) // 2 if frame_names else -1
        correction_norm = summary.get("camera_correction_norm_m", []) or []
        height_scale = summary.get("height_scale", {}) or {}
        framewise_blend = anchor.get("framewise_anchor_blend", {}) or {}
        dual_fusion = anchor.get("dual_anchor_fusion", {}) or {}
        bev_branch = dual_fusion.get("bev", {}) or {}
        multiframe_branch = dual_fusion.get("multiframe", {}) or {}
        bev_rescue = anchor.get("bev_rescue", {}) or {}
        guided_ly_retry = (
            bev_rescue.get("guided_ly_retry", {})
            or anchor.get("bev_guided_ly_retry", {})
            or multiframe_branch.get("bev_guided_ly_retry", {})
            or {}
        )
        temporal_confirmation = (
            bev_rescue.get("temporal_confirmation", {}) or {}
        )
        row: dict[str, Any] = {
            "window_idx": int(summary.get("window_idx", -1)),
            "start_idx": int(summary.get("start_idx", -1)),
            "end_idx": int(summary.get("end_idx", -1)),
            "center_local_idx": int(center_local_idx),
            "center_frame": frame_names[center_local_idx] if 0 <= center_local_idx < len(frame_names) else "",
            "anchor_match_method": summary.get("anchor_match_method", ""),
            "anchor_accepted": bool(summary.get("anchor_accepted", False)),
            "reason": anchor.get("reason", ""),
            "match_method": anchor.get("match_method", ""),
            "dual_anchor_enabled": bool(dual_fusion.get("enabled", False)),
            "dual_anchor_decision": dual_fusion.get("decision", anchor.get("dual_anchor_decision", "")),
            "dual_anchor_agreement": _anchor_scalar(dual_fusion.get("agreement")),
            "dual_position_difference_m": _anchor_scalar(dual_fusion.get("position_difference_m")),
            "dual_yaw_difference_deg": _anchor_scalar(dual_fusion.get("yaw_difference_deg")),
            "dual_scale_log_difference": _anchor_scalar(dual_fusion.get("scale_log_difference")),
            "dual_bev_quality_score": _anchor_scalar(dual_fusion.get("bev_quality_score")),
            "dual_multiframe_quality_score": _anchor_scalar(dual_fusion.get("multiframe_quality_score")),
            "bev_rescue_triggered": bool(
                bev_rescue.get("triggered", False)
            ),
            "bev_rescue_lost_mode": bool(
                bev_rescue.get("lost_mode", False)
            ),
            "bev_rescue_relocalized": bool(
                bev_rescue.get("relocalized", False)
            ),
            "bev_rescue_extra_margin_m": _anchor_scalar(
                bev_rescue.get("extra_margin_m")
            ),
            "bev_rescue_rejections_before": _anchor_scalar(
                bev_rescue.get("consecutive_rejections_before")
            ),
            "bev_rescue_rejections_after": _anchor_scalar(
                bev_rescue.get("consecutive_rejections_after")
            ),
            "bev_rescue_temporal_enabled": bool(
                temporal_confirmation.get("enabled", False)
            ),
            "bev_rescue_temporal_status": temporal_confirmation.get("status", ""),
            "bev_rescue_temporal_confirmed": bool(
                temporal_confirmation.get("confirmed", False)
                and temporal_confirmation.get("status") == "confirmed_and_lost"
            ),
            "bev_rescue_temporal_pending_before": bool(
                temporal_confirmation.get("pending_before", False)
            ),
            "bev_rescue_temporal_pending_after": bool(
                temporal_confirmation.get("pending_after", False)
            ),
            "bev_rescue_temporal_previous_window": _anchor_scalar(
                temporal_confirmation.get("previous_window_idx")
            ),
            "bev_rescue_temporal_overlap_frames": _anchor_scalar(
                temporal_confirmation.get("overlap_frame_count")
            ),
            "bev_rescue_temporal_position_m": _anchor_scalar(
                temporal_confirmation.get("position_median_difference_m")
            ),
            "bev_rescue_temporal_yaw_deg": _anchor_scalar(
                temporal_confirmation.get("yaw_median_difference_deg")
            ),
            "bev_rescue_temporal_scale_log": _anchor_scalar(
                temporal_confirmation.get("scale_log_difference")
            ),
            "bev_guided_ly_retry_enabled": bool(
                guided_ly_retry.get("enabled", False)
            ),
            "bev_guided_ly_retry_requested": bool(
                guided_ly_retry.get("requested", False)
            ),
            "bev_guided_ly_retry_executed": bool(
                guided_ly_retry.get("executed", False)
            ),
            "bev_guided_ly_retry_accepted": bool(
                guided_ly_retry.get("accepted", False)
            ),
            "bev_guided_ly_retry_reason": guided_ly_retry.get("reason", ""),
            "bev_guided_ly_retry_bev_quality_ok": bool(
                guided_ly_retry.get("bev_quality_ok", False)
            ),
            "bev_guided_ly_retry_bev_inlier_count": _anchor_scalar(
                guided_ly_retry.get("bev_inlier_count")
            ),
            "bev_guided_ly_retry_inlier_count": _anchor_scalar(
                guided_ly_retry.get("inlier_count")
            ),
            "bev_guided_ly_retry_inlier_ratio": _anchor_scalar(
                guided_ly_retry.get("inlier_ratio")
            ),
            "bev_guided_ly_retry_rmse_m": _anchor_scalar(
                guided_ly_retry.get("rmse_m")
            ),
            "bev_branch_accepted": bool(bev_branch.get("accepted", False)),
            "bev_branch_reason": bev_branch.get("reason", ""),
            "bev_branch_inlier_count": _anchor_scalar(bev_branch.get("inlier_count")),
            "bev_branch_inlier_ratio": _anchor_scalar(bev_branch.get("inlier_ratio")),
            "bev_branch_rmse_m": _anchor_scalar(bev_branch.get("rmse")),
            "multiframe_branch_accepted": bool(multiframe_branch.get("accepted", False)),
            "multiframe_branch_reason": multiframe_branch.get("reason", ""),
            "multiframe_selected_keyframe_count": int(
                len(multiframe_branch.get("selected_local_indices", []))
            ),
            "multiframe_required_inlier_frame_count": _anchor_scalar(
                multiframe_branch.get("required_inlier_frame_count")
            ),
            "multiframe_candidate_count": _anchor_scalar(multiframe_branch.get("candidate_count")),
            "multiframe_mapped_frame_count": _anchor_scalar(multiframe_branch.get("mapped_frame_count")),
            "multiframe_inlier_frame_count": _anchor_scalar(multiframe_branch.get("inlier_frame_count")),
            "multiframe_inlier_count": _anchor_scalar(multiframe_branch.get("inlier_count")),
            "multiframe_inlier_ratio": _anchor_scalar(multiframe_branch.get("inlier_ratio")),
            "multiframe_rmse_m": _anchor_scalar(multiframe_branch.get("rmse")),
            "multiframe_residual_scale": _anchor_scalar(multiframe_branch.get("residual_scale")),
            "multiframe_residual_yaw_deg": _anchor_scalar(multiframe_branch.get("residual_yaw_deg")),
            "multiframe_selected_frames": json.dumps(
                multiframe_branch.get("selected_frames", []), ensure_ascii=False
            ),
            "continuity_gate_enabled": bool(anchor.get("continuity_gate_enabled", False)),
            "continuity_gate_skipped": bool(anchor.get("continuity_gate_skipped", False)),
            "continuity_gate_method": anchor.get("continuity_gate_method", ""),
            "position_innovation_warning_threshold_m": _anchor_scalar(
                anchor.get("position_innovation_warning_threshold_m")
            ),
            "large_position_innovation": bool(anchor.get("large_position_innovation", False)),
            "large_yaw_innovation": bool(anchor.get("large_yaw_innovation", False)),
            "yaw_gate_softened_by_dual_anchor": bool(
                anchor.get("continuity_gate_softened_by_dual_anchor", False)
            ),
            "large_position_innovation_blend": _anchor_scalar(
                anchor.get("large_position_innovation_blend")
            ),
            "applied_anchor_mode": anchor.get("applied_anchor_mode", ""),
            "applied_anchor_blend": _anchor_scalar(anchor.get("applied_anchor_blend")),
            "framewise_anchor_blend_mode": framewise_blend.get("mode", ""),
            "framewise_anchor_blend_start": _anchor_scalar(framewise_blend.get("start")),
            "framewise_anchor_blend_end": _anchor_scalar(framewise_blend.get("end")),
            "yaw_gate_threshold_deg": _anchor_scalar(anchor.get("continuity_gate_yaw_threshold_deg")),
            "extrapolation_gate_enabled": bool(anchor.get("extrapolation_gate_enabled", False)),
            "extrapolation_gate_skipped": bool(anchor.get("extrapolation_gate_skipped", False)),
            "extrapolation_history_count": _anchor_scalar(anchor.get("extrapolation_history_count")),
            "extrapolation_position_threshold_m": _anchor_scalar(
                anchor.get("extrapolation_gate_position_threshold_m")
            ),
            "extrapolation_yaw_threshold_deg": _anchor_scalar(
                anchor.get("extrapolation_gate_yaw_threshold_deg")
            ),
            "extrapolated_position_error_m": _anchor_scalar(anchor.get("extrapolated_position_error_m")),
            "extrapolated_yaw_error_deg": _anchor_scalar(anchor.get("extrapolated_yaw_error_deg")),
            "raw_matches": _anchor_scalar(anchor.get("raw_matches")),
            "valid_matches": _anchor_scalar(anchor.get("valid_matches")),
            "inlier_count": _anchor_scalar(anchor.get("inlier_count")),
            "inlier_ratio": _anchor_scalar(anchor.get("inlier_ratio")),
            "rmse_m": _anchor_scalar(anchor.get("rmse")),
            "median_residual_m": _anchor_scalar(anchor.get("median_residual")),
            "scale": _anchor_scalar(anchor.get("scale")),
            "theta_deg": _anchor_scalar(anchor.get("theta_deg", anchor.get("yaw_deg"))),
            "satellite_raw_scale": _anchor_scalar(anchor.get("satellite_raw_scale")),
            "satellite_applied_scale": _anchor_scalar(anchor.get("satellite_applied_scale")),
            "satellite_raw_yaw_deg": _anchor_scalar(anchor.get("satellite_raw_yaw_deg")),
            "satellite_applied_yaw_deg": _anchor_scalar(anchor.get("satellite_applied_yaw_deg")),
            "satellite_apply_scale": bool(anchor.get("satellite_apply_scale", False)),
            "satellite_apply_yaw": bool(anchor.get("satellite_apply_yaw", False)),
            "scale_observation_only": bool(anchor.get("scale_observation_only", False)),
            "sequential_scale_filter_enabled": bool(
                anchor.get("sequential_anchor_scale_filter_enabled", False)
            ),
            "sequential_scale_filter_active": bool(
                anchor.get("sequential_anchor_scale_filter_active", False)
            ),
            "sequential_scale_multiplier_before": _anchor_scalar(
                anchor.get("sequential_anchor_scale_multiplier_before")
            ),
            "sequential_scale_multiplier_after": _anchor_scalar(
                anchor.get("sequential_anchor_scale_multiplier_after")
            ),
            "sequential_scale_kalman_gain": _anchor_scalar(
                (anchor.get("sequential_anchor_scale_update", {}) or {}).get("kalman_gain")
            ),
            "sequential_scale_measurement_std_log": _anchor_scalar(
                (anchor.get("sequential_anchor_scale_update", {}) or {}).get("measurement_std_log")
            ),
            "sequential_scale_update_reason": (
                (anchor.get("sequential_anchor_scale_update", {}) or {}).get("reason", "")
            ),
            "robust_component_fusion_enabled": bool(
                anchor.get("robust_component_fusion_enabled", False)
            ),
            "scale_consensus_history": json.dumps(anchor.get("scale_consensus_history", [])),
            "scale_consensus_reference": _anchor_scalar(anchor.get("scale_consensus_reference")),
            "scale_consensus_relative_deviation": _anchor_scalar(
                anchor.get("scale_consensus_relative_deviation")
            ),
            "scale_consensus_passed": bool(anchor.get("scale_consensus_passed", False)),
            "yaw_consensus_history_deg": json.dumps(anchor.get("yaw_consensus_history_deg", [])),
            "yaw_consensus_reference_deg": _anchor_scalar(anchor.get("yaw_consensus_reference_deg")),
            "yaw_consensus_deviation_deg": _anchor_scalar(anchor.get("yaw_consensus_deviation_deg")),
            "yaw_consensus_passed": bool(anchor.get("yaw_consensus_passed", False)),
            "satellite_suppressed_components": json.dumps(
                anchor.get("satellite_suppressed_components", [])
            ),
            "template_yaw_correction_deg": _anchor_scalar(anchor.get("template_yaw_correction_deg")),
            "image_yaw_offset_deg": _anchor_scalar(anchor.get("image_yaw_offset_deg")),
            "yaw_offset_deg": _anchor_scalar(anchor.get("yaw_offset_deg")),
            "ncc_score": _anchor_scalar(anchor.get("ncc_score")),
            "peak_margin": _anchor_scalar(anchor.get("peak_margin")),
            "peak_sharpness": _anchor_scalar(anchor.get("peak_sharpness")),
            "position_innovation_m": _anchor_scalar(anchor.get("position_innovation_m")),
            "yaw_innovation_deg": _anchor_scalar(anchor.get("yaw_innovation_deg")),
            "gt_center_frame": anchor.get("gt_center_frame", ""),
            "gt_center_source": anchor.get("gt_center_source", ""),
            "prior_center_to_gt_m": _anchor_scalar(anchor.get("prior_center_to_gt_m")),
            "candidate_center_to_gt_m": _anchor_scalar(anchor.get("candidate_center_to_gt_m")),
            "candidate_gt_diagnostic": anchor.get("candidate_gt_diagnostic", ""),
            "center_correction_norm_m": (
                float(correction_norm[center_local_idx])
                if 0 <= center_local_idx < len(correction_norm)
                else float("nan")
            ),
            "satellite_crop_width_m": _anchor_scalar(anchor.get("satellite_crop_width_m")),
            "satellite_crop_height_m": _anchor_scalar(anchor.get("satellite_crop_height_m")),
            "satellite_footprint_expand_each_side_m": _anchor_scalar(
                anchor.get("satellite_footprint_expand_each_side_m")
            ),
            "satellite_center_inside": _anchor_scalar(anchor.get("satellite_center_inside")),
            "camera_to_footprint_center_distance_m": _anchor_scalar(
                anchor.get("camera_to_footprint_center_distance_m")
            ),
            "height_scale_m_per_local": _anchor_scalar(height_scale.get("m_per_local")),
            "horizontal_sim2_scale_m_per_local": _anchor_scalar(
                height_scale.get("horizontal_sim2_scale_m_per_local")
            ),
        }
        row.update(_anchor_xy_fields("source_center", anchor.get("source_center_utm")))
        row.update(_anchor_xy_fields("matched_center", anchor.get("matched_center_utm")))
        row.update(_anchor_xy_fields("prior_center", anchor.get("prior_center_utm")))
        row.update(_anchor_xy_fields("candidate_center", anchor.get("candidate_center_utm")))
        row.update(_anchor_xy_fields("gt_center", anchor.get("gt_center_utm")))
        row.update(_anchor_xy_fields("satellite_crop_center", anchor.get("satellite_crop_center_utm")))
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    return int(len(rows))


def aggregate_per_frame_metrics(items: list[dict[str, Any]], label: str) -> dict[str, Any]:
    errors = np.asarray([float(x.get("position_error_m", np.nan)) for x in items], dtype=np.float64)
    delta_e = np.asarray([float(x.get("delta_east_m", np.nan)) for x in items], dtype=np.float64)
    delta_n = np.asarray([float(x.get("delta_north_m", np.nan)) for x in items], dtype=np.float64)
    yaw_errors = np.asarray([float(x.get("yaw_error_deg", np.nan)) for x in items], dtype=np.float64)
    valid = np.isfinite(errors)
    errors = errors[valid]
    delta_e = delta_e[valid]
    delta_n = delta_n[valid]
    yaw_errors = yaw_errors[valid & np.isfinite(yaw_errors)] if len(yaw_errors) == len(valid) else np.asarray([])
    if len(errors) == 0:
        return {"label": label, "sample_count": 0}
    deltas = np.column_stack([delta_e, delta_n])
    covariance = np.cov(deltas, rowvar=False) if len(errors) > 1 else np.zeros((2, 2), dtype=np.float64)
    recalls = {f"recall_{thr}m": float(np.mean(errors <= float(thr))) for thr in (5, 10, 20, 30, 50)}
    accs = {f"acc_{thr}m": value for thr, value in zip((5, 10, 20, 30, 50), recalls.values())}
    row = {
        "label": label,
        "sample_count": int(len(errors)),
        "position_mean_m": float(np.mean(errors)),
        "position_variance_m2": float(np.var(errors)),
        "position_std_m": float(np.std(errors)),
        "position_median_m": float(np.median(errors)),
        "position_p95_m": float(np.percentile(errors, 95)),
        "position_max_m": float(np.max(errors)),
        "delta_east_mean_m": float(np.mean(delta_e)),
        "delta_north_mean_m": float(np.mean(delta_n)),
        "position_cov_ee_m2": float(covariance[0, 0]),
        "position_cov_en_m2": float(covariance[0, 1]),
        "position_cov_ne_m2": float(covariance[1, 0]),
        "position_cov_nn_m2": float(covariance[1, 1]),
        **recalls,
        **accs,
    }
    if len(yaw_errors) > 0:
        yaw_accs = {
            f"yaw_acc_{threshold}deg": float(np.mean(np.abs(yaw_errors) <= float(threshold)))
            for threshold in (5, 10, 20)
        }
        row.update({
            "yaw_abs_mean_deg": float(np.mean(np.abs(yaw_errors))),
            "yaw_abs_median_deg": float(np.median(np.abs(yaw_errors))),
            "yaw_abs_p95_deg": float(np.percentile(np.abs(yaw_errors), 95)),
            "yaw_abs_max_deg": float(np.max(np.abs(yaw_errors))),
            "yaw_signed_mean_deg": float(np.mean(yaw_errors)),
            "yaw_signed_variance_deg2": float(np.var(yaw_errors)),
            **yaw_accs,
        })
    return row


def flatten_sequence_metric(result: dict[str, Any], metrics: dict[str, Any], kind: str) -> dict[str, Any]:
    row = _metrics_summary_row(metrics)
    per_frame = metrics.get("per_frame", [])
    if per_frame:
        errors = np.asarray([float(x.get("position_error_m", np.nan)) for x in per_frame], dtype=np.float64)
        errors = errors[np.isfinite(errors)]
        if len(errors) > 0:
            for thr in (5, 10, 20, 30, 50):
                value = float(np.mean(errors <= float(thr)))
                row[f"recall_{thr}m"] = value
                row[f"acc_{thr}m"] = value
        yaw_errors = np.asarray([float(x.get("yaw_error_deg", np.nan)) for x in per_frame], dtype=np.float64)
        yaw_errors = yaw_errors[np.isfinite(yaw_errors)]
        if len(yaw_errors) > 0:
            for threshold in (5, 10, 20):
                row[f"yaw_acc_{threshold}deg"] = float(
                    np.mean(np.abs(yaw_errors) <= float(threshold))
                )
    row["dataset_name"] = result["dataset_name"]
    row["kind"] = kind
    row["save_dir"] = result["save_dir"]
    row["window_count"] = result["window_count"]
    row["accepted_anchor_count"] = result["accepted_anchor_count"]
    row["frame_count"] = result["frame_count"]
    row["runtime_sec"] = float(result.get("runtime_sec", float("nan")))
    timed_stages = [
        timing for timing in result.get("timing_rows", [])
        if timing.get("stage") != "other_unattributed_overhead"
    ]
    if timed_stages:
        top_stage = max(timed_stages, key=lambda timing: float(timing["total_sec"]))
        row["runtime_top_stage"] = str(top_stage["stage"])
        row["runtime_top_stage_sec"] = float(top_stage["total_sec"])
        row["runtime_top_stage_percent"] = float(top_stage["percent_of_sequence"])
    return row


def save_batch_reports(batch_root: Path, results: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    sequence_rows: list[dict[str, Any]] = []
    corrected_frames: list[dict[str, Any]] = []
    uncorrected_frames: list[dict[str, Any]] = []
    for result in results:
        sequence_rows.append(flatten_sequence_metric(result, result["uncorrected_metrics"], "uncorrected"))
        sequence_rows.append(flatten_sequence_metric(result, result["corrected_metrics"], "corrected"))
        for item in result["uncorrected_metrics"].get("per_frame", []):
            row = dict(item)
            row["dataset_name"] = result["dataset_name"]
            uncorrected_frames.append(row)
        for item in result["corrected_metrics"].get("per_frame", []):
            row = dict(item)
            row["dataset_name"] = result["dataset_name"]
            corrected_frames.append(row)

    sequence_path = batch_root / "batch_sequence_metrics.csv"
    pd.DataFrame(sequence_rows).to_csv(sequence_path, index=False, encoding="utf-8-sig")
    summary_rows = [
        aggregate_per_frame_metrics(uncorrected_frames, "overall_uncorrected"),
        aggregate_per_frame_metrics(corrected_frames, "overall_corrected"),
    ]
    summary_path = batch_root / "batch_metrics_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    timing_detail_rows: list[dict[str, Any]] = []
    for result in results:
        for timing in result.get("timing_rows", []):
            timing_detail_rows.append({
                "dataset_name": result["dataset_name"],
                "sequence_runtime_sec": float(result.get("runtime_sec", float("nan"))),
                "window_count": int(result["window_count"]),
                "frame_count": int(result["frame_count"]),
                **timing,
            })
    timing_detail_path = batch_root / "batch_runtime_stage_breakdown.csv"
    pd.DataFrame(timing_detail_rows).to_csv(
        timing_detail_path, index=False, encoding="utf-8-sig"
    )
    timing_summary_path = batch_root / "batch_runtime_stage_summary.csv"
    if timing_detail_rows:
        timing_detail_df = pd.DataFrame(timing_detail_rows)
        timing_summary_df = (
            timing_detail_df.groupby("stage", as_index=False)
            .agg(
                total_sec=("total_sec", "sum"),
                call_count=("call_count", "sum"),
                sequence_count=("dataset_name", "nunique"),
            )
        )
        timing_summary_df["percent_of_measured_batch"] = (
            100.0 * timing_summary_df["total_sec"]
            / max(float(timing_summary_df["total_sec"].sum()), 1e-12)
        )
        timing_summary_df["mean_sec_per_call"] = (
            timing_summary_df["total_sec"]
            / timing_summary_df["call_count"].clip(lower=1)
        )
        timing_summary_df = timing_summary_df.sort_values(
            "total_sec", ascending=False
        )
    else:
        timing_summary_df = pd.DataFrame(columns=[
            "stage", "total_sec", "call_count", "sequence_count",
            "percent_of_measured_batch", "mean_sec_per_call",
        ])
    timing_summary_df.to_csv(
        timing_summary_path, index=False, encoding="utf-8-sig"
    )
    batch_window_timing_rows: list[dict[str, Any]] = []
    for result in results:
        for window_timing in result.get("window_timing_rows", []):
            batch_window_timing_rows.append({
                "dataset_name": result["dataset_name"],
                **window_timing,
            })
    timing_window_path = batch_root / "batch_runtime_window_breakdown.csv"
    pd.DataFrame(batch_window_timing_rows).to_csv(
        timing_window_path, index=False, encoding="utf-8-sig"
    )
    scale_recommendations = {
        str(result["dataset_name"]): float(calibration["recommended_multiplier"])
        for result in results
        for calibration in [result.get("sequence_metric_scale_calibration", {})]
        if bool(calibration.get("available", False))
        and int(calibration.get("kept_sample_count", 0)) >= 3
    }
    scale_recommendations_path = batch_root / "batch_sequence_metric_scale_recommendations.json"
    with open(scale_recommendations_path, "w", encoding="utf-8") as file:
        json.dump(scale_recommendations, file, indent=2, ensure_ascii=False)
    if failures:
        pd.DataFrame(failures).to_csv(batch_root / "batch_failures.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=["dataset_name", "error"]).to_csv(
            batch_root / "batch_failures.csv", index=False, encoding="utf-8-sig"
        )

    if not BATCH_EXPERIMENT_COMPACT_MODE:
        print("\n================ Batch Summary ================")
        print(pd.DataFrame(summary_rows).to_string(index=False))
        print(f"[BATCH] sequence metrics: {sequence_path}")
        print(f"[BATCH] overall metrics:  {summary_path}")
        print(f"[BATCH] scale map:        {scale_recommendations_path}")
        print(f"[BATCH] runtime detail:   {timing_detail_path}")
        print(f"[BATCH] runtime summary:  {timing_summary_path}")
        print(f"[BATCH] runtime windows:  {timing_window_path}")
        print(f"[BATCH] failures:         {batch_root / 'batch_failures.csv'}")
        print("================================================")
