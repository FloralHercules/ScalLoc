from __future__ import annotations

import gc
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import cv2
from PIL import Image

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore


LOGGER = logging.getLogger(__name__)


@dataclass
class VGGTPrediction:
    points: np.ndarray
    confidences: np.ndarray
    colors_rgb: np.ndarray
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    images_hw: tuple[int, int]
    stats: dict[str, Any]
    vggt_tokens: "torch.Tensor | None" = None
    point_frame_ids: np.ndarray | None = None
    point_ys: np.ndarray | None = None
    point_xs: np.ndarray | None = None
    tracks: np.ndarray | None = None
    track_visibility: np.ndarray | None = None
    track_confidences: np.ndarray | None = None
    track_query_points: np.ndarray | None = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required. Activate the ObliqueLoc Python environment.")


def inference_dtype(device: "torch.device") -> "torch.dtype":
    require_torch()
    if device.type != "cuda":
        return torch.float32
    major, _minor = torch.cuda.get_device_capability(device)
    return torch.bfloat16 if major >= 8 else torch.float16


def autocast_context(device: "torch.device", dtype: "torch.dtype"):
    require_torch()
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda")


def _extract_state_dict(checkpoint: Any) -> dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")
    return {str(key).removeprefix("module.").removeprefix("model."): value for key, value in checkpoint.items()}


def resolve_device(device: str | "torch.device" = "auto") -> "torch.device":
    require_torch()
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_vggt_model(
    vggt_root: str | Path,
    weights: str | Path,
    device: str | "torch.device" = "auto",
    enable_track: bool = False,
) -> "torch.nn.Module":
    require_torch()
    sys.path.insert(0, str(Path(vggt_root)))
    from vggt.models.vggt import VGGT

    device_obj = resolve_device(device)
    model = VGGT(enable_track=enable_track)
    checkpoint = torch.load(Path(weights), map_location="cpu")
    result = model.load_state_dict(_extract_state_dict(checkpoint), strict=False)
    bad_missing = [key for key in result.missing_keys if enable_track or not key.startswith("track_head.")]
    bad_unexpected = [key for key in result.unexpected_keys if enable_track or not key.startswith("track_head.")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(f"VGGT checkpoint mismatch: missing={bad_missing[:8]} unexpected={bad_unexpected[:8]}")
    model.eval().to(device_obj)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    LOGGER.info("[VGGT] loaded frozen model on %s", device_obj)
    return model


def _load_rgb_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    if image.mode == "RGBA":
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)
    return image.convert("RGB")


def _pil_to_tensor(image: Image.Image) -> "torch.Tensor":
    array = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0


def load_vggt_images_and_masks(
    image_paths: Sequence[str | Path],
    preprocess_mode: str,
    device: "torch.device",
    target_size: int = 518,
) -> tuple["torch.Tensor", np.ndarray]:
    require_torch()
    if preprocess_mode not in {"crop", "pad"}:
        raise ValueError("preprocess_mode must be 'crop' or 'pad'")
    images: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    shapes: set[tuple[int, int]] = set()
    for image_path in image_paths:
        image = _load_rgb_image(image_path)
        width, height = image.size
        if preprocess_mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = _pil_to_tensor(image)
        mask = torch.ones((new_height, new_width), dtype=torch.bool)
        if preprocess_mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            tensor = tensor[:, start_y : start_y + target_size, :]
            mask = mask[start_y : start_y + target_size, :]
        if preprocess_mode == "pad":
            h_pad = target_size - tensor.shape[1]
            w_pad = target_size - tensor.shape[2]
            if h_pad > 0 or w_pad > 0:
                pad_top = h_pad // 2
                pad_bottom = h_pad - pad_top
                pad_left = w_pad // 2
                pad_right = w_pad - pad_left
                tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
                mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=False)
        shapes.add((tensor.shape[1], tensor.shape[2]))
        images.append(tensor)
        masks.append(mask)
    if len(shapes) > 1:
        max_h = max(shape[0] for shape in shapes)
        max_w = max(shape[1] for shape in shapes)
        padded_images = []
        padded_masks = []
        for tensor, mask in zip(images, masks):
            h_pad = max_h - tensor.shape[1]
            w_pad = max_w - tensor.shape[2]
            if h_pad > 0 or w_pad > 0:
                pad_top = h_pad // 2
                pad_bottom = h_pad - pad_top
                pad_left = w_pad // 2
                pad_right = w_pad - pad_left
                tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0)
                mask = F.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=False)
            padded_images.append(tensor)
            padded_masks.append(mask)
        images = padded_images
        masks = padded_masks
    return torch.stack(images).to(device, non_blocking=True), torch.stack(masks).cpu().numpy().astype(bool)


def make_homogeneous_extrinsics(extrinsics_3x4: np.ndarray) -> np.ndarray:
    out = np.tile(np.eye(4, dtype=np.float64), (len(extrinsics_3x4), 1, 1))
    out[:, :3, :4] = extrinsics_3x4.astype(np.float64)
    return out


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    rotations = extrinsics[:, :3, :3]
    translations = extrinsics[:, :3, 3]
    return (-np.einsum("nij,nj->ni", np.transpose(rotations, (0, 2, 1)), translations)).astype(np.float64)


def _input_intrinsics_from_source_calibration(
    source_calibration: dict[str, float],
    source_width: int,
    source_height: int,
    input_width: int,
    input_height: int,
    preprocess: str,
    target_size: int,
) -> np.ndarray:
    """Map a source-image K into the exact resize/pad coordinates fed to VGGT."""
    if preprocess != "pad":
        raise ValueError("Fixed-intrinsics ray correction currently requires preprocess='pad'")
    if input_width != int(target_size) or input_height != int(target_size):
        raise ValueError("Unexpected VGGT input size for fixed-intrinsics ray correction")
    if source_width >= source_height:
        resized_width = int(target_size)
        resized_height = round(source_height * (resized_width / source_width) / 14) * 14
    else:
        resized_height = int(target_size)
        resized_width = round(source_width * (resized_height / source_height) / 14) * 14
    scale_x = float(resized_width) / float(source_width)
    scale_y = float(resized_height) / float(source_height)
    pad_left = (input_width - resized_width) // 2
    pad_top = (input_height - resized_height) // 2
    return np.array([
        [float(source_calibration["fx"]) * scale_x, 0.0,
         float(source_calibration["cx"]) * scale_x + pad_left],
        [0.0, float(source_calibration["fy"]) * scale_y,
         float(source_calibration["cy"]) * scale_y + pad_top],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rectify_world_points_with_fixed_intrinsics(
    world_points: np.ndarray,
    extrinsics: np.ndarray,
    predicted_intrinsics: np.ndarray,
    image_paths: Sequence[str | Path],
    source_calibration: dict[str, float],
    preprocess: str,
    target_size: int,
    output_extrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recast each VGGT point ray through a calibrated K, keeping its predicted depth.

    VGGT does not accept K as an input. This is therefore an explicit post-model
    geometry correction, not a claim that the network itself was calibrated. When
    ``output_extrinsics`` is supplied, depths are recovered with ``extrinsics``
    but rays are placed with the optimized output poses. This is the required
    dense counterpart of fixed-K track BA.
    """
    points = np.asarray(world_points, dtype=np.float64)
    depth_ext = np.asarray(extrinsics, dtype=np.float64)
    output_ext = depth_ext if output_extrinsics is None else np.asarray(output_extrinsics, dtype=np.float64)
    predicted_k = np.asarray(predicted_intrinsics, dtype=np.float64)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError(f"Expected world points [frames,height,width,3], got {points.shape}")
    frame_count, input_height, input_width, _ = points.shape
    if (
        len(image_paths) != frame_count
        or depth_ext.shape[0] != frame_count
        or output_ext.shape[0] != frame_count
        or predicted_k.shape[0] != frame_count
    ):
        raise ValueError("Fixed-intrinsics correction inputs disagree on frame count")

    corrected = points.copy()
    fixed_ks: list[np.ndarray] = []
    per_frame: list[dict[str, float]] = []
    yy, xx = np.indices((input_height, input_width), dtype=np.float64)
    for frame_idx, image_path in enumerate(image_paths):
        with Image.open(image_path) as image:
            source_width, source_height = image.size
        fixed_k = _input_intrinsics_from_source_calibration(
            source_calibration, source_width, source_height, input_width,
            input_height, preprocess, target_size,
        )
        fixed_ks.append(fixed_k)
        depth_rotation = depth_ext[frame_idx, :3, :3]
        depth_translation = depth_ext[frame_idx, :3, 3]
        output_rotation = output_ext[frame_idx, :3, :3]
        output_translation = output_ext[frame_idx, :3, 3]
        camera_xyz = points[frame_idx] @ depth_rotation.T + depth_translation
        depth = camera_xyz[..., 2]
        valid = np.isfinite(camera_xyz).all(axis=-1) & (np.abs(depth) > 1e-8)
        calibrated_camera_xyz = np.empty_like(camera_xyz)
        calibrated_camera_xyz[..., 0] = (xx - fixed_k[0, 2]) / fixed_k[0, 0] * depth
        calibrated_camera_xyz[..., 1] = (yy - fixed_k[1, 2]) / fixed_k[1, 1] * depth
        calibrated_camera_xyz[..., 2] = depth
        corrected_frame = (calibrated_camera_xyz - output_translation) @ output_rotation
        corrected[frame_idx, valid] = corrected_frame[valid]
        per_frame.append({
            "predicted_fx_input_px": float(predicted_k[frame_idx, 0, 0]),
            "predicted_fy_input_px": float(predicted_k[frame_idx, 1, 1]),
            "fixed_fx_input_px": float(fixed_k[0, 0]),
            "fixed_fy_input_px": float(fixed_k[1, 1]),
            "valid_point_fraction": float(np.mean(valid)),
        })
    median_predicted_fx = float(np.median(predicted_k[:, 0, 0]))
    median_predicted_fy = float(np.median(predicted_k[:, 1, 1]))
    median_fixed_fx = float(np.median([k[0, 0] for k in fixed_ks]))
    median_fixed_fy = float(np.median([k[1, 1] for k in fixed_ks]))
    return corrected.astype(world_points.dtype, copy=False), {
        "mode": (
            "per_point_predicted_depth_fixed_k_optimized_pose_recast"
            if output_extrinsics is not None else "per_point_same_depth_ray_recast"
        ),
        "source_calibration": {key: float(value) for key, value in source_calibration.items()},
        "median_predicted_fx_input_px": median_predicted_fx,
        "median_predicted_fy_input_px": median_predicted_fy,
        "median_fixed_fx_input_px": median_fixed_fx,
        "median_fixed_fy_input_px": median_fixed_fy,
        "median_fx_ratio_fixed_over_predicted": median_fixed_fx / max(median_predicted_fx, 1e-12),
        "median_fy_ratio_fixed_over_predicted": median_fixed_fy / max(median_predicted_fy, 1e-12),
        "depth_pose_equals_output_pose": bool(output_extrinsics is None),
        "per_frame": per_frame,
    }


def make_track_query_grid(
    width: int,
    height: int,
    columns: int = 18,
    rows: int = 14,
    margin_px: int = 42,
) -> np.ndarray:
    """A deterministic, interior grid suitable for VGGT's native track head."""
    xs = np.linspace(float(margin_px), float(width - 1 - margin_px), int(columns))
    ys = np.linspace(float(margin_px), float(height - 1 - margin_px), int(rows))
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def _project_points_pinhole(points: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    camera = np.asarray(points, dtype=np.float64) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    depth = camera[:, 2]
    image = np.column_stack([
        intrinsic[0, 0] * camera[:, 0] / np.maximum(depth, 1e-12) + intrinsic[0, 2],
        intrinsic[1, 1] * camera[:, 1] / np.maximum(depth, 1e-12) + intrinsic[1, 2],
    ])
    return image, depth


def _triangulate_track_landmarks(
    tracks: np.ndarray,
    visible: np.ndarray,
    extrinsics: np.ndarray,
    intrinsic: np.ndarray,
    fallback: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linear multi-view triangulation for every track using fixed calibrated K."""
    refined = np.asarray(fallback, dtype=np.float64).copy()
    observation_counts = np.zeros(len(refined), dtype=np.int32)
    for track_idx in range(len(refined)):
        rows: list[np.ndarray] = []
        rhs: list[float] = []
        for frame_idx in np.flatnonzero(visible[:, track_idx]):
            u, v = tracks[frame_idx, track_idx]
            if not np.isfinite([u, v]).all():
                continue
            x = (float(u) - float(intrinsic[0, 2])) / float(intrinsic[0, 0])
            y = (float(v) - float(intrinsic[1, 2])) / float(intrinsic[1, 1])
            rotation = extrinsics[frame_idx, :3, :3]
            translation = extrinsics[frame_idx, :3, 3]
            rows.extend([x * rotation[2] - rotation[0], y * rotation[2] - rotation[1]])
            rhs.extend([translation[0] - x * translation[2], translation[1] - y * translation[2]])
            observation_counts[track_idx] += 1
        if observation_counts[track_idx] >= 2:
            try:
                point, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
                if np.isfinite(point).all():
                    refined[track_idx] = point
            except np.linalg.LinAlgError:
                pass
    return refined, observation_counts


def refine_extrinsics_with_fixed_k_tracks(
    world_points: np.ndarray,
    extrinsics: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    confidences: np.ndarray | None,
    query_points: np.ndarray,
    fixed_intrinsics: np.ndarray,
    iterations: int = 2,
    min_visibility: float = 0.45,
    min_confidence: float = 0.25,
    retriangulate_landmarks: bool = True,
    preserve_initial_camera_centers: bool = False,
    rotation_only_max_final_rms_px: float | None = None,
    rotation_only_max_rms_ratio: float | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Alternating fixed-K multi-view bundle adjustment (PnP and triangulation).

    The first camera stays fixed as the local gauge. Query landmarks are initialized
    from its VGGT depth but cast through the true K; later poses are repeatedly
    recovered with RANSAC PnP and all landmarks are triangulated from all visible
    track observations. This preserves multi-view consistency, unlike per-frame
    point-ray recasting.
    """
    points = np.asarray(world_points, dtype=np.float64)
    initial_ext = np.asarray(extrinsics, dtype=np.float64)
    refined_ext = initial_ext.copy()
    track_xy = np.asarray(tracks, dtype=np.float64)
    vis = np.asarray(visibility, dtype=np.float64)
    query_xy = np.asarray(query_points, dtype=np.float64)
    frame_count, track_count, _ = track_xy.shape
    if points.shape[0] != frame_count or len(query_xy) != track_count:
        raise ValueError("Track BA inputs disagree on frame or track count")
    valid = np.isfinite(track_xy).all(axis=-1) & (vis >= float(min_visibility))
    if confidences is not None:
        valid &= np.asarray(confidences, dtype=np.float64) >= float(min_confidence)
    valid[0] = np.isfinite(track_xy[0]).all(axis=-1)

    qx = np.clip(np.rint(query_xy[:, 0]).astype(np.int32), 0, points.shape[2] - 1)
    qy = np.clip(np.rint(query_xy[:, 1]).astype(np.int32), 0, points.shape[1] - 1)
    seed_world = points[0, qy, qx]
    rotation0 = initial_ext[0, :3, :3]
    translation0 = initial_ext[0, :3, 3]
    seed_camera = seed_world @ rotation0.T + translation0
    seed_depth = seed_camera[:, 2]
    seed_camera[:, 0] = (query_xy[:, 0] - fixed_intrinsics[0, 2]) / fixed_intrinsics[0, 0] * seed_depth
    seed_camera[:, 1] = (query_xy[:, 1] - fixed_intrinsics[1, 2]) / fixed_intrinsics[1, 1] * seed_depth
    landmarks = (seed_camera - translation0) @ rotation0
    seed_valid = np.isfinite(landmarks).all(axis=1) & (np.abs(seed_depth) > 1e-8)
    valid &= seed_valid[None, :]
    centers_initial = camera_centers_from_extrinsics(initial_ext)

    def reprojection_rms(current_ext: np.ndarray, current_points: np.ndarray) -> float | None:
        errors: list[np.ndarray] = []
        for frame_idx in range(frame_count):
            ids = np.flatnonzero(valid[frame_idx])
            if len(ids) == 0:
                continue
            projected, depth = _project_points_pinhole(current_points[ids], current_ext[frame_idx], fixed_intrinsics)
            ok = depth > 1e-7
            if np.any(ok):
                errors.append(np.linalg.norm(projected[ok] - track_xy[frame_idx, ids][ok], axis=1))
        if not errors:
            return None
        return float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))

    rms_before = reprojection_rms(refined_ext, landmarks)
    per_iteration: list[dict[str, Any]] = []
    for iteration in range(max(1, int(iterations))):
        pnp_frames = 0
        pnp_inliers = 0
        for frame_idx in range(1, frame_count):
            ids = np.flatnonzero(valid[frame_idx])
            if len(ids) < 12:
                continue
            image_points = track_xy[frame_idx, ids].reshape(-1, 1, 2).astype(np.float64)
            object_points = landmarks[ids].reshape(-1, 1, 3).astype(np.float64)
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                fixed_intrinsics,
                None,
                iterationsCount=120,
                reprojectionError=3.0,
                confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success or inliers is None or len(inliers) < 10:
                continue
            inlier_ids = ids[inliers.reshape(-1)]
            try:
                rvec, tvec = cv2.solvePnPRefineLM(
                    landmarks[inlier_ids].reshape(-1, 1, 3).astype(np.float64),
                    track_xy[frame_idx, inlier_ids].reshape(-1, 1, 2).astype(np.float64),
                    fixed_intrinsics,
                    None,
                    rvec,
                    tvec,
                )
            except cv2.error:
                pass
            refined_rotation = cv2.Rodrigues(rvec)[0]
            refined_ext[frame_idx, :3, :3] = refined_rotation
            if preserve_initial_camera_centers:
                refined_ext[frame_idx, :3, 3] = (
                    -refined_rotation @ centers_initial[frame_idx]
                )
            else:
                refined_ext[frame_idx, :3, 3] = tvec.reshape(3)
            pnp_frames += 1
            pnp_inliers += int(len(inliers))
        if retriangulate_landmarks:
            landmarks, observation_counts = _triangulate_track_landmarks(
                track_xy, valid, refined_ext, fixed_intrinsics, landmarks
            )
        else:
            observation_counts = np.count_nonzero(valid, axis=0)
        per_iteration.append({
            "iteration": int(iteration + 1),
            "pnp_refined_frame_count": int(pnp_frames),
            "pnp_inlier_count": int(pnp_inliers),
            "triangulated_track_count": int(np.count_nonzero(observation_counts >= 2)),
            "reprojection_rms_px": reprojection_rms(refined_ext, landmarks),
        })
    candidate_final_rms = reprojection_rms(refined_ext, landmarks)
    quality_reasons: list[str] = []
    if preserve_initial_camera_centers and candidate_final_rms is not None:
        if (
            rotation_only_max_final_rms_px is not None
            and candidate_final_rms > float(rotation_only_max_final_rms_px)
        ):
            quality_reasons.append("final_reprojection_rms_too_large")
        if (
            rotation_only_max_rms_ratio is not None
            and rms_before is not None
            and candidate_final_rms / max(rms_before, 1e-12)
            > float(rotation_only_max_rms_ratio)
        ):
            quality_reasons.append("reprojection_rms_ratio_too_large")
    quality_accepted = not quality_reasons
    if not quality_accepted:
        refined_ext = initial_ext.copy()

    centers_refined = camera_centers_from_extrinsics(refined_ext)
    rotation_delta_deg = []
    for frame_idx in range(frame_count):
        delta_rotation = refined_ext[frame_idx, :3, :3] @ initial_ext[frame_idx, :3, :3].T
        cosine = np.clip((np.trace(delta_rotation) - 1.0) * 0.5, -1.0, 1.0)
        rotation_delta_deg.append(float(np.degrees(np.arccos(cosine))))
    if preserve_initial_camera_centers:
        mode = (
            "alternating_triangulation_pnp_rotation_only_fixed_k"
            if retriangulate_landmarks
            else "fixed_first_frame_landmarks_pnp_rotation_only_fixed_k"
        )
    else:
        mode = (
            "alternating_pnp_multiview_triangulation_fixed_k"
            if retriangulate_landmarks
            else "fixed_first_frame_landmarks_pnp_only_fixed_k"
        )
    return refined_ext.astype(extrinsics.dtype, copy=False), {
        "mode": mode,
        "retriangulate_landmarks": bool(retriangulate_landmarks),
        "preserve_initial_camera_centers": bool(preserve_initial_camera_centers),
        "quality_gate_accepted": bool(quality_accepted),
        "quality_gate_reasons": quality_reasons,
        "rotation_only_max_final_rms_px": rotation_only_max_final_rms_px,
        "rotation_only_max_rms_ratio": rotation_only_max_rms_ratio,
        "track_count": int(track_count),
        "valid_observation_count": int(np.count_nonzero(valid)),
        "initial_reprojection_rms_px": rms_before,
        "candidate_final_reprojection_rms_px": candidate_final_rms,
        "final_reprojection_rms_px": reprojection_rms(refined_ext, landmarks),
        "camera_center_shift_local_median": float(np.median(np.linalg.norm(centers_refined - centers_initial, axis=1))),
        "camera_center_shift_local_max": float(np.max(np.linalg.norm(centers_refined - centers_initial, axis=1))),
        "camera_rotation_change_deg_median": float(np.median(rotation_delta_deg)),
        "camera_rotation_change_deg_max": float(np.max(rotation_delta_deg)),
        "iterations": per_iteration,
    }


def run_vggt_reconstruction(
    model: "torch.nn.Module",
    image_paths: Sequence[str | Path],
    device: str | "torch.device" = "auto",
    preprocess: str = "pad",
    target_size: int = 518,
    conf_percentile: float = 35.0,
    max_points: int = 300000,
    seed: int = 42,
    use_depth_unproject: bool = True,
    fixed_source_intrinsics: dict[str, float] | None = None,
    fixed_k_track_ba: bool = False,
    fixed_k_track_ba_dense_reproject: bool = False,
    fixed_k_track_ba_retriangulate_landmarks: bool = True,
    fixed_k_track_ba_preserve_camera_centers: bool = False,
    fixed_k_track_ba_rotation_only_max_final_rms_px: float | None = None,
    fixed_k_track_ba_rotation_only_max_rms_ratio: float | None = None,
    track_query_columns: int = 18,
    track_query_rows: int = 14,
    track_query_points_override: np.ndarray | None = None,
) -> VGGTPrediction:
    require_torch()
    device_obj = resolve_device(device)
    dtype = inference_dtype(device_obj)
    images, valid_mask = load_vggt_images_and_masks(image_paths, preprocess, device_obj, target_size=int(target_size))

    vggt_tokens_holder: list[torch.Tensor] = []

    def _capture_aggregator(_module, _input, output):
        tokens_list, _patch_start = output
        vggt_tokens_holder.append(tokens_list[-1].detach())

    handle = model.aggregator.register_forward_hook(_capture_aggregator)
    query_points_np = None
    query_points_t = None
    if fixed_k_track_ba:
        if track_query_points_override is None:
            query_points_np = make_track_query_grid(
                int(images.shape[-1]), int(images.shape[-2]),
                columns=int(track_query_columns), rows=int(track_query_rows),
            )
        else:
            query_points_np = np.asarray(
                track_query_points_override, dtype=np.float32
            ).reshape(-1, 2)
            if len(query_points_np) == 0 or not np.isfinite(query_points_np).all():
                raise ValueError(
                    "track_query_points_override must contain finite Nx2 points"
                )
            query_points_np[:, 0] = np.clip(
                query_points_np[:, 0], 0.0, float(images.shape[-1] - 1)
            )
            query_points_np[:, 1] = np.clip(
                query_points_np[:, 1], 0.0, float(images.shape[-2] - 1)
            )
        query_points_t = torch.from_numpy(query_points_np).to(device_obj)
    with torch.no_grad():
        with autocast_context(device_obj, dtype):
            predictions = model(images, query_points=query_points_t)
    handle.remove()
    vggt_tokens = vggt_tokens_holder[0].squeeze(0) if vggt_tokens_holder else None
    world_points = predictions["world_points"].squeeze(0).detach().float().cpu().numpy()
    point_conf = predictions["world_points_conf"].squeeze(0).detach().float().cpu().numpy()
    rgb = (images.detach().float().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)

    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    extrinsic_t, intrinsic_t = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    extrinsics = make_homogeneous_extrinsics(extrinsic_t.squeeze(0).detach().float().cpu().numpy())
    initial_extrinsics = extrinsics.copy()
    intrinsics = intrinsic_t.squeeze(0).detach().float().cpu().numpy()

    tracks = None
    track_visibility = None
    track_confidences = None
    track_ba_stats: dict[str, Any] | None = None
    if query_points_np is not None:
        if not all(key in predictions for key in ("track", "vis", "conf")):
            raise RuntimeError("VGGT track head was not enabled but fixed-K track BA was requested")
        tracks = predictions["track"].squeeze(0).detach().float().cpu().numpy()
        track_visibility = predictions["vis"].squeeze(0).detach().float().cpu().numpy()
        track_confidences = predictions["conf"].squeeze(0).detach().float().cpu().numpy()

    fixed_intrinsics_stats: dict[str, Any] | None = None
    if fixed_source_intrinsics is not None:
        source_width, source_height = _load_rgb_image(image_paths[0]).size
        fixed_k = _input_intrinsics_from_source_calibration(
            fixed_source_intrinsics,
            source_width, source_height,
            int(images.shape[-1]), int(images.shape[-2]), preprocess, int(target_size),
        )
        if fixed_k_track_ba:
            # Do not apply the old dense per-frame ray recast here: that makes
            # overlapping point maps inconsistent. BA only changes poses and its
            # sparse landmarks are jointly constrained by all track observations.
            fixed_intrinsics_stats = {
                "mode": "fixed_k_used_by_multiview_track_ba_no_dense_point_recast",
                "median_predicted_fx_input_px": float(np.median(intrinsics[:, 0, 0])),
                "median_predicted_fy_input_px": float(np.median(intrinsics[:, 1, 1])),
                "median_fixed_fx_input_px": float(fixed_k[0, 0]),
                "median_fixed_fy_input_px": float(fixed_k[1, 1]),
            }
            extrinsics, track_ba_stats = refine_extrinsics_with_fixed_k_tracks(
                world_points,
                extrinsics,
                tracks,
                track_visibility,
                track_confidences,
                query_points_np,
                fixed_k,
                retriangulate_landmarks=fixed_k_track_ba_retriangulate_landmarks,
                preserve_initial_camera_centers=fixed_k_track_ba_preserve_camera_centers,
                rotation_only_max_final_rms_px=(
                    fixed_k_track_ba_rotation_only_max_final_rms_px
                ),
                rotation_only_max_rms_ratio=(
                    fixed_k_track_ba_rotation_only_max_rms_ratio
                ),
            )
            if fixed_k_track_ba_dense_reproject:
                world_points, dense_reprojection_stats = rectify_world_points_with_fixed_intrinsics(
                    world_points,
                    initial_extrinsics,
                    intrinsics,
                    image_paths,
                    fixed_source_intrinsics,
                    preprocess,
                    int(target_size),
                    output_extrinsics=extrinsics,
                )
                fixed_intrinsics_stats["dense_reprojection"] = dense_reprojection_stats
        else:
            world_points, fixed_intrinsics_stats = rectify_world_points_with_fixed_intrinsics(
                world_points,
                extrinsics,
                intrinsics,
                image_paths,
                fixed_source_intrinsics,
                preprocess,
                int(target_size),
            )

    depth_dense_points = None
    depth_dense_colors = None
    if use_depth_unproject:
        try:
            if "depth" in predictions and "depth_conf" in predictions:
                from vggt.utils.geometry import unproject_depth_map_to_point_map

                depth_np = predictions["depth"].squeeze(0).detach().float().cpu().numpy()
                depth_conf_np = predictions["depth_conf"].squeeze(0).detach().float().cpu().numpy()
                depth_valid = (depth_conf_np > 1.1) & valid_mask
                if np.any(depth_valid):
                    ext_3x4 = extrinsic_t.squeeze(0).detach().float().cpu().numpy()
                    int_3x3 = intrinsic_t.squeeze(0).detach().float().cpu().numpy()
                    depth_world = unproject_depth_map_to_point_map(depth_np, ext_3x4, int_3x3)
                    depth_dense_points = depth_world[depth_valid].astype(np.float32)
                    depth_dense_colors = rgb[depth_valid].astype(np.uint8)
                    LOGGER.info("[VGGT] depth_unproject_points=%d", len(depth_dense_points))
        except Exception:
            LOGGER.debug("[VGGT] depth unprojection skipped", exc_info=True)

    finite = np.isfinite(world_points).all(axis=-1) & np.isfinite(point_conf) & valid_mask
    if not np.any(finite):
        raise ValueError("VGGT produced no finite 3D points")
    threshold = float(np.percentile(point_conf[finite], conf_percentile))
    keep = finite & (point_conf >= threshold)
    if np.count_nonzero(keep) < 1000:
        threshold = float(np.percentile(point_conf[finite], 10.0))
        keep = finite & (point_conf >= threshold)

    frame_count, height, width = world_points.shape[:3]
    frame_idx_grid, y_grid, x_grid = np.meshgrid(
        np.arange(frame_count, dtype=np.int32),
        np.arange(height, dtype=np.int32),
        np.arange(width, dtype=np.int32),
        indexing="ij",
    )
    flat_points = world_points[keep].astype(np.float32)
    flat_conf = point_conf[keep].astype(np.float32)
    flat_rgb = rgb[keep].astype(np.uint8)
    flat_frame_ids = frame_idx_grid[keep].astype(np.int32)
    flat_ys = y_grid[keep].astype(np.int32)
    flat_xs = x_grid[keep].astype(np.int32)

    if depth_dense_points is not None and depth_dense_colors is not None and len(depth_dense_points) > 0:
        flat_points = np.concatenate([flat_points, depth_dense_points], axis=0)
        flat_conf = np.concatenate([flat_conf, np.ones(len(depth_dense_points), dtype=np.float32) * threshold], axis=0)
        flat_rgb = np.concatenate([flat_rgb, depth_dense_colors], axis=0)
        flat_frame_ids = np.concatenate([flat_frame_ids, -np.ones(len(depth_dense_points), dtype=np.int32)], axis=0)
        flat_ys = np.concatenate([flat_ys, -np.ones(len(depth_dense_points), dtype=np.int32)], axis=0)
        flat_xs = np.concatenate([flat_xs, -np.ones(len(depth_dense_points), dtype=np.int32)], axis=0)

    points_before_sampling = len(flat_points)
    if max_points > 0 and points_before_sampling > max_points:
        rng = np.random.default_rng(seed)
        ids = rng.choice(points_before_sampling, size=max_points, replace=False)
        flat_points = flat_points[ids]
        flat_conf = flat_conf[ids]
        flat_rgb = flat_rgb[ids]
        flat_frame_ids = flat_frame_ids[ids]
        flat_ys = flat_ys[ids]
        flat_xs = flat_xs[ids]

    stats = {
        "image_count": len(image_paths),
        "images_hw": tuple(int(x) for x in images.shape[-2:]),
        "raw_finite_points": int(np.count_nonzero(finite)),
        "confidence_percentile": float(conf_percentile),
        "confidence_threshold": float(threshold),
        "points_after_confidence": int(points_before_sampling),
        "points_used": int(len(flat_points)),
    }
    if fixed_intrinsics_stats is not None:
        stats["fixed_intrinsics_ray_correction"] = fixed_intrinsics_stats
    if track_ba_stats is not None:
        stats["fixed_k_track_ba"] = track_ba_stats
    LOGGER.info("[VGGT] point_cloud=%d conf_threshold=%.4f", len(flat_points), threshold)
    del images, predictions
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return VGGTPrediction(
        flat_points,
        flat_conf,
        flat_rgb,
        extrinsics,
        intrinsics,
        tuple(stats["images_hw"]),
        stats,
        vggt_tokens=vggt_tokens,
        point_frame_ids=flat_frame_ids,
        point_ys=flat_ys,
        point_xs=flat_xs,
        tracks=tracks,
        track_visibility=track_visibility,
        track_confidences=track_confidences,
        track_query_points=query_points_np,
    )
