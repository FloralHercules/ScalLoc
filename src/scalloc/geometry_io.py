import os

# 必须在导入 cv2 之前解除 OpenCV 对超大图片的像素限制！
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
os.environ.setdefault("PYTHONHASHSEED", "42")

import cv2
import json
import yaml
import glob
import shutil
import math
import pickle
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
from plyfile import PlyData, PlyElement

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

# 导入你的可视化与坐标系工具函数
from scalloc.map_utils import (
    deg2utm, utm2deg,
    load_config_parameters_new, 
    crop_geo_images_with_offset, 
    dumpRotateImage,
    compute_Mk_to_M0_transform,
    apply_sim3_transform_to_poses,
    project_point_cloud_sim3
)
import random

GLOBAL_RANDOM_SEED = 42
STRICT_TORCH_DETERMINISM = False
DISABLE_TF32_FOR_REPRODUCIBILITY = False
OPENCV_SINGLE_THREAD_FOR_REPRODUCIBILITY = False

def seed_everything(
        seed=42,
        strict_torch_determinism=STRICT_TORCH_DETERMINISM,
        disable_tf32=DISABLE_TF32_FOR_REPRODUCIBILITY,
        opencv_single_thread=OPENCV_SINGLE_THREAD_FOR_REPRODUCIBILITY,
    ):
    """
    固定随机源。默认不强行切换 PyTorch/CUDA/OpenCV 的算法路径，
    避免影响速度、数值路径或匹配结果；需要严格复现实验时再打开上面的开关。
    """
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = bool(strict_torch_determinism)
    torch.backends.cudnn.benchmark = not bool(strict_torch_determinism)
    if strict_torch_determinism:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    torch.backends.cudnn.allow_tf32 = not bool(disable_tf32)
    torch.backends.cuda.matmul.allow_tf32 = not bool(disable_tf32)
    
    # 如果涉及到 OpenCV 的随机操作（如某些特征匹配算法）
    cv2.setRNGSeed(seed)
    if opencv_single_thread:
        cv2.setNumThreads(1)


# ==========================================
# 辅助函数：点云读写
# ==========================================
def load_ply_points(ply_path):
    if not os.path.exists(ply_path): return None
    plydata = PlyData.read(ply_path)
    x = np.array(plydata['vertex']['x'])
    y = np.array(plydata['vertex']['y'])
    z = np.array(plydata['vertex']['z'])
    
    colors = None
    if 'red' in plydata['vertex']:
        r = np.array(plydata['vertex']['red'])
        g = np.array(plydata['vertex']['green'])
        b = np.array(plydata['vertex']['blue'])
        colors = np.stack([r, g, b], axis=-1)
        
    points = np.stack([x, y, z], axis=-1)
    return points, colors

def save_unified_ply(points, colors, out_path):
    if colors is not None:
        vertex = np.core.records.fromarrays(
            [points[:,0], points[:,1], points[:,2], colors[:,0], colors[:,1], colors[:,2]],
            names='x, y, z, red, green, blue',
            formats='f4, f4, f4, u1, u1, u1'
        )
    else:
        vertex = np.core.records.fromarrays(
            [points[:,0], points[:,1], points[:,2]],
            names='x, y, z',
            formats='f4, f4, f4'
        )
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el], text=False).write(out_path)

# ==========================================
# 核心算法模块：点云投影与找平
# ==========================================
def auto_level_point_cloud(points, up_axis='y', up_direction=-1):
    print("正在使用 NumPy 实现的 RANSAC 提取地面平面，以自动找平点云...")

    n_points = points.shape[0]
    if n_points < 3:
        return points, np.eye(3), np.zeros(3)

    # 为了加速，最多只采样十万个点进行 RANSAC
    max_pts = 100000
    if n_points > max_pts:
        indices = np.random.choice(n_points, max_pts, replace=False)
        pts = points[indices]
    else:
        pts = points

    n_pts = pts.shape[0]

    # 动态计算距离阈值
    bbox_extent = np.max(pts, axis=0) - np.min(pts, axis=0)
    scale = np.linalg.norm(bbox_extent)
    distance_threshold = scale * 0.005 # 0.5% scale

    best_inliers_count = 0
    best_normal = None
    best_inliers_mask = None

    num_iterations = 2000
    for _ in range(num_iterations):
        # 随机选 3 个点
        idx = np.random.choice(n_pts, 3, replace=False)
        p1, p2, p3 = pts[idx]

        # 计算法线
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue

        normal = normal / norm
        d = -np.dot(normal, p1)

        # 计算所有点到平面的距离
        distances = np.abs(np.dot(pts, normal) + d)
        inliers_mask = distances < distance_threshold
        inliers_count = np.sum(inliers_mask)

        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_normal = normal
            best_inliers_mask = inliers_mask

    if best_normal is None:
        print("未找到有效平面，跳过自动找平。")
        return points, np.eye(3), np.zeros(3)

    # 使用所有 inliers 重新拟合平面 (SVD) 以获得更精确的法线
    inlier_pts = pts[best_inliers_mask]
    if len(inlier_pts) >= 3:
        centroid = np.mean(inlier_pts, axis=0)
        centered_pts = inlier_pts - centroid
        _, _, vh = np.linalg.svd(centered_pts, full_matrices=False)
        best_normal = vh[2, :]

    normal = best_normal

    # 定义期望的“上”向量
    target_up = np.array([0.0, 0.0, 0.0])
    if up_axis == 'x': target_up[0] = up_direction
    elif up_axis == 'y': target_up[1] = up_direction
    elif up_axis == 'z': target_up[2] = up_direction

    # 确保法线方向朝上
    if np.dot(normal, target_up) < 0:
        normal = -normal

    # 计算旋转矩阵
    v = np.cross(normal, target_up)
    s = np.linalg.norm(v)
    c_val = np.dot(normal, target_up)

    if s < 1e-6:
        print("点云已水平，无需旋转。")
        return points, np.eye(3), np.zeros(3)

    v_skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])   
    R_mat = np.eye(3) + v_skew + (v_skew @ v_skew) * ((1 - c_val) / (s**2))

    # 应用刚体变换：绕点云质心旋转（包含平移项）
    center_orig = points.mean(axis=0)
    t_align = center_orig - (R_mat @ center_orig)
    points_rotated = (R_mat @ points.T).T + t_align
    print(f"自动找平完成 (内点数: {best_inliers_count}/{n_pts})")       
    return points_rotated, R_mat, t_align

def _orthophoto_axes(points, up_axis, up_direction):
    if up_axis == 'z':
        u, v, h = points[:, 0], points[:, 1], points[:, 2]
    elif up_axis == 'y':
        u, v, h = points[:, 0], points[:, 2], points[:, 1]
    elif up_axis == 'x':
        u, v, h = points[:, 1], points[:, 2], points[:, 0]
    else:
        raise ValueError("Invalid up_axis")
    return u, v, h * up_direction

def _resolve_first_frame_name(frame_names, first_frame_prior):
    if first_frame_prior is not None and first_frame_prior.get("name"):
        prior_name = os.path.splitext(os.path.basename(str(first_frame_prior["name"])))[0]
        for name in frame_names:
            name_base = os.path.splitext(os.path.basename(str(name)))[0]
            if name == prior_name or name_base == prior_name:
                return name
    return sorted(frame_names)[0]

def estimate_local_to_meter_from_first_frame(
        points,
        R_align,
        t_align,
        up_axis,
        up_direction,
        extri_path,
        first_frame_prior,
        ground_percentile=5.0
    ):
    rel_height_m = _read_float(first_frame_prior, ["rel_height", "alt"])
    if rel_height_m is None or rel_height_m <= 0:
        return None
    if extri_path is None or not os.path.exists(extri_path):
        return None

    try:
        frame_names, local_poses = load_scal3r_poses(extri_path)
    except Exception as e:
        print(f"[BEV scale] Failed to read first-frame extrinsics: {e}")
        return None
    valid_names = [name for name in frame_names if name in local_poses]
    if not valid_names:
        return None

    first_name = _resolve_first_frame_name(valid_names, first_frame_prior)
    pose0 = local_poses[first_name]
    cam_center_0 = -pose0["R"].T @ pose0["T"]
    cam_leveled_0 = R_align @ cam_center_0 + t_align

    p_u, p_v, p_h = _orthophoto_axes(points, up_axis, up_direction)
    c_u, c_v, c_h = _orthophoto_axes(cam_leveled_0.reshape(1, 3), up_axis, up_direction)
    c_u, c_v, c_h = float(c_u[0]), float(c_v[0]), float(c_h[0])

    finite = np.isfinite(p_u) & np.isfinite(p_v) & np.isfinite(p_h)
    p_u, p_v, p_h = p_u[finite], p_v[finite], p_h[finite]
    if p_h.size < 16:
        return None

    dist2 = (p_u - c_u) ** 2 + (p_v - c_v) ** 2
    candidate_count = min(p_h.size, max(500, p_h.size // 20))
    if candidate_count < p_h.size:
        near_idx = np.argpartition(dist2, candidate_count - 1)[:candidate_count]
        cand_h = p_h[near_idx]
    else:
        cand_h = p_h

    below_cam = cand_h[cand_h < c_h]
    if below_cam.size >= 16:
        cand_h = below_cam

    low_h = float(np.percentile(cand_h, ground_percentile))
    high_h = float(np.percentile(cand_h, 100.0 - ground_percentile))
    height_to_low = c_h - low_h
    height_to_high = high_h - c_h
    if height_to_low > 1e-6:
        local_height = height_to_low
        ground_h = low_h
    elif height_to_high > 1e-6:
        local_height = height_to_high
        ground_h = high_h
    else:
        local_height = max(abs(c_h - low_h), abs(c_h - high_h))
        ground_h = low_h if abs(c_h - low_h) >= abs(c_h - high_h) else high_h

    if local_height <= 1e-6:
        return None

    local_to_meter = float(rel_height_m / local_height)
    if not np.isfinite(local_to_meter) or local_to_meter <= 0:
        return None

    return {
        "local_to_meter": local_to_meter,
        "first_frame_name": first_name,
        "first_frame_rel_height_m": float(rel_height_m),
        "first_frame_local_height": float(local_height),
        "first_frame_camera_h": float(c_h),
        "first_frame_ground_h": float(ground_h),
        "scale_source": "first_frame_height_prior",
        "candidate_point_count": int(cand_h.size),
    }

def interpolate_orthophoto_for_matching(
        img_bgr,
        mask_img,
        inpaint_radius=3,
        blur_sigma=0.75,
        blend_alpha=0.55
    ):
    """
    Densify and gently smooth the point-cloud orthomosaic for matching.
    The operation is constrained to the valid footprint so the black background
    and footprint boundary do not become stronger matching cues.
    """
    if img_bgr is None or mask_img is None:
        return img_bgr

    mask_bool = mask_img > 127
    if not np.any(mask_bool):
        return img_bgr

    out = img_bgr.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY) if out.ndim == 3 else out.copy()
    valid_gray = gray[mask_bool]
    fill_value = int(np.median(valid_gray)) if valid_gray.size > 0 else 0

    work = out.copy()
    work[~mask_bool] = fill_value

    # Reconstruct tiny empty raster cells inside the valid footprint first.
    dark_holes = (gray <= 2) & mask_bool
    if np.any(dark_holes):
        inpaint_mask = dark_holes.astype(np.uint8) * 255
        work = cv2.inpaint(work, inpaint_mask, inpaint_radius, cv2.INPAINT_TELEA)

    # Sub-pixel interpolation pass: up/down sampling removes hard raster cells
    # while preserving the orthomosaic size and its GSD metadata.
    h, w = work.shape[:2]
    up = cv2.resize(work, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    interp = cv2.resize(up, (w, h), interpolation=cv2.INTER_AREA)

    if blur_sigma is not None and blur_sigma > 0:
        smooth = cv2.GaussianBlur(interp, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    else:
        smooth = interp

    blended = cv2.addWeighted(work, 1.0 - blend_alpha, smooth, blend_alpha, 0)
    out[mask_bool] = blended[mask_bool]
    out[~mask_bool] = 0
    return out

def postprocess_projected_pointcloud_for_matching(
        img_bgr,
        mask_img,
        dilate_kernel_size=7,
        close_kernel_size=13,
        inpaint_radius=5,
        blur_sigma=0.75,
        blend_alpha=0.55,
    ):
    """
    Use the same valid-mask idea as orthophoto generation to turn sparse projected
    point-cloud pixels into a denser matching image.
    """
    if img_bgr is None or mask_img is None:
        return img_bgr, mask_img

    mask_img = (mask_img > 0).astype(np.uint8) * 255
    if not np.any(mask_img > 0):
        return img_bgr, mask_img

    img = img_bgr.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    valid = mask_img > 0
    valid_gray = gray[valid]
    if valid_gray.size > 0:
        p_min, p_max = np.percentile(valid_gray, [1, 99])
        if p_max > p_min:
            gray_norm = np.clip(
                (gray.astype(np.float32) - p_min) / (p_max - p_min) * 255.0,
                0,
                255,
            ).astype(np.uint8)
            img = cv2.merge([gray_norm, gray_norm, gray_norm])
            img[~valid] = 0

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (int(dilate_kernel_size), int(dilate_kernel_size)),
    )
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (int(close_kernel_size), int(close_kernel_size)),
    )
    mask_dilated = cv2.dilate(mask_img, kernel, iterations=1)
    mask_final = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)
    hole_mask = cv2.bitwise_and(mask_final, cv2.bitwise_not(mask_img))

    img_closed = cv2.bitwise_and(img, img, mask=mask_img)
    if np.any(hole_mask > 0):
        img_closed = cv2.inpaint(
            img_closed,
            hole_mask,
            inpaintRadius=int(inpaint_radius),
            flags=cv2.INPAINT_TELEA,
        )
    img_closed[mask_final == 0] = 0

    img_interp = interpolate_orthophoto_for_matching(
        img_closed,
        mask_final,
        inpaint_radius=3,
        blur_sigma=blur_sigma,
        blend_alpha=blend_alpha,
    )
    return img_interp, mask_final

def _densify_point_idx_map_for_matching(point_idx_map, match_mask, max_distance_px=10.0):
    """
    The rendered image may be densified for matching, but PnP still needs each
    matched pixel to refer to a nearby real projected 3D point.  Fill a lookup
    map by nearest raw projected point and reject pixels too far from any source.
    """
    point_idx_map = np.asarray(point_idx_map)
    source_valid = point_idx_map >= 0
    dense = np.full(point_idx_map.shape, -1, dtype=np.int32)
    dist_map = np.full(point_idx_map.shape, np.inf, dtype=np.float32)
    if not np.any(source_valid):
        return dense, dist_map

    try:
        from scipy.ndimage import distance_transform_edt
        dist, nearest = distance_transform_edt(
            (~source_valid).astype(np.uint8),
            return_indices=True,
        )
        nearest_idx = point_idx_map[nearest[0], nearest[1]]
        allowed = (np.asarray(match_mask) > 0) & (dist <= float(max_distance_px)) & (nearest_idx >= 0)
        dense[allowed] = nearest_idx[allowed].astype(np.int32)
        dist_map[allowed] = dist[allowed].astype(np.float32)
        dense[source_valid] = point_idx_map[source_valid].astype(np.int32)
        dist_map[source_valid] = 0.0
        return dense, dist_map
    except Exception:
        kernel_radius = max(1, int(math.ceil(float(max_distance_px))))
        for y, x in np.argwhere(source_valid):
            y1, y2 = max(0, y - kernel_radius), min(point_idx_map.shape[0], y + kernel_radius + 1)
            x1, x2 = max(0, x - kernel_radius), min(point_idx_map.shape[1], x + kernel_radius + 1)
            yy, xx = np.mgrid[y1:y2, x1:x2]
            d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2)
            update = (d <= float(max_distance_px)) & (d < dist_map[y1:y2, x1:x2]) & (np.asarray(match_mask)[y1:y2, x1:x2] > 0)
            dense_view = dense[y1:y2, x1:x2]
            dist_view = dist_map[y1:y2, x1:x2]
            dense_view[update] = int(point_idx_map[y, x])
            dist_view[update] = d[update].astype(np.float32)
        return dense, dist_map

def generate_orthophoto(
        ply_path,
        out_path,
        max_resolution=2048,
        up_axis='y',
        up_direction=-1,
        use_morphology=False,
        auto_level=False,
        target_gsd_m=None,
        first_frame_prior=None,
        extri_path=None,
        max_output_pixels=80000000,
        interpolate_for_matching=False
    ):
    print(f"正在加载点云文件 {ply_path}...")
    plydata = PlyData.read(ply_path)

    # 提取坐标
    x = np.array(plydata['vertex']['x'])
    y = np.array(plydata['vertex']['y'])
    z = np.array(plydata['vertex']['z'])

    # 提取颜色（如果存在）
    if 'red' in plydata['vertex']:
        r = np.array(plydata['vertex']['red'])
        g = np.array(plydata['vertex']['green'])
        b = np.array(plydata['vertex']['blue'])
    else:
        r = np.full_like(x, 200, dtype=np.uint8)
        g = np.full_like(x, 200, dtype=np.uint8)
        b = np.full_like(x, 200, dtype=np.uint8)

    points = np.stack([x, y, z], axis=-1)
    colors = np.stack([b, g, r], axis=-1) # OpenCV默认顺序是BGR

    R_align = np.eye(3)
    t_align = np.zeros(3)
    if auto_level:
        points, R_align, t_align = auto_level_point_cloud(points, up_axis, up_direction)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Select the BEV plane in the leveled local coordinate system.
    u, v, h = _orthophoto_axes(points, up_axis, up_direction)

    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()

    v = -v
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()

    u_range = u_max - u_min
    v_range = v_max - v_min
    max_range = max(u_range, v_range)

    scale_meta = None
    local_to_meter = None
    resolution_source = "max_resolution"
    target_gsd_m = float(target_gsd_m) if target_gsd_m is not None else None

    if target_gsd_m is not None and target_gsd_m > 0:
        scale_meta = estimate_local_to_meter_from_first_frame(
            points=points,
            R_align=R_align,
            t_align=t_align,
            up_axis=up_axis,
            up_direction=up_direction,
            extri_path=extri_path,
            first_frame_prior=first_frame_prior,
        )
        if scale_meta is not None:
            local_to_meter = scale_meta["local_to_meter"]
            resolution = target_gsd_m / local_to_meter
            resolution_source = "first_frame_prior_target_gsd"
            print(
                f"[BEV scale] 1 local unit = {local_to_meter:.6f} m, "
                f"target_gsd={target_gsd_m:.4f} m/px, local_resolution={resolution:.6f}"
            )
        else:
            print("[BEV scale] First-frame prior scale failed; fallback to max_resolution.")

    if local_to_meter is None:
        if max_range <= 0:
            raise ValueError("Point cloud has zero BEV extent; cannot generate orthophoto.")
        resolution = max_range / max_resolution

    if resolution <= 0 or not np.isfinite(resolution):
        raise ValueError(f"Invalid orthophoto resolution: {resolution}")

    width = max(1, int(np.ceil((u_max - u_min) / resolution)))
    height = max(1, int(np.ceil((v_max - v_min) / resolution)))

    pixel_cap_applied = False
    if max_output_pixels is not None and width * height > max_output_pixels:
        cap_scale = np.sqrt((width * height) / float(max_output_pixels))
        resolution *= cap_scale
        width = max(1, int(np.ceil((u_max - u_min) / resolution)))
        height = max(1, int(np.ceil((v_max - v_min) / resolution)))
        pixel_cap_applied = True
        resolution_source += "_pixel_cap"
        actual_gsd = resolution * local_to_meter if local_to_meter is not None else None
        print(f"[BEV scale] Output capped to {width}x{height}; actual_gsd={actual_gsd}")

    print(f"坐标系参数: up_axis={up_axis}, up_direction={up_direction}")    
    print(f"生成的正射图像尺寸预估为 {width} x {height} ...")

    u_idx = np.floor((u - u_min) / resolution).astype(int)
    v_idx = np.floor((v - v_min) / resolution).astype(int)

    u_idx = np.clip(u_idx, 0, width - 1)
    v_idx = np.clip(v_idx, 0, height - 1)

    sort_idx = np.argsort(h)
    u_idx = u_idx[sort_idx]
    v_idx = v_idx[sort_idx]
    colors = colors[sort_idx]

    img = np.zeros((height, width, 3), dtype=np.uint8)
    mask_img = np.zeros((height, width), dtype=np.uint8)

    img[v_idx, u_idx] = colors
    mask_img[v_idx, u_idx] = 255

    is_gray = np.all(r == g) and np.all(g == b)
    if is_gray:
        print("检测到点云色彩为极暗的单通道灰度！正对其进行对比度自适应拉伸，使其从黑底中显现出来...")
        img_gray = img[:, :, 0]

        mask = (img_gray > 0).astype(np.uint8)
        valid_pixels = img_gray[mask > 0]

        if len(valid_pixels) > 0:
            p_min, p_max = np.percentile(valid_pixels, 1), np.percentile(valid_pixels, 99)
            if p_max > p_min:
                img_norm = np.clip((img_gray.astype(np.float32) - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
            else:
                img_norm = img_gray

            img = cv2.merge([img_norm, img_norm, img_norm])
            img_closed = cv2.bitwise_and(img, img, mask=mask)
        else:
            img_closed = img
    else:
        img_closed = img

    if use_morphology:
        print("进行掩膜界限修补与图像纹理保留填充...")
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)) 

        # 1. 仅对单独的物理边界掩膜进行强力形态学膨胀和闭运算，推断出真实视野的几何轮廓
        mask_dilated = cv2.dilate(mask_img, kernel, iterations=1)
        mask_final = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, kernel_close)

        # 2. 找出需要修复的孔洞（处于有效掩膜内，但原本没有任何点云落上去的地方）
        hole_mask = cv2.bitwise_and(mask_final, cv2.bitwise_not(mask_img))

        # 3. 无损结构修复：取代以前的 cv2.dilate！
        # 以前直接在原图上做 cv2.dilate（其实是最大值池化运算），极易导致高亮像素扩张，像黑洞一样把暗色（冷源）的马路全部吞噬糊掉！
        # 现在改用专用的 Inpaint（图像修补）算法，它会根据边缘柔和地推断填补颜色，100% 保护马路线条和地面特征不变形。
        print("正在执行 Inpaint 算法无损修复空洞 (保护暗色特征)...")
        img_closed = cv2.inpaint(img_closed, hole_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

        # 将最终有效区域赋值给 mask_img 供保存
        mask_img = mask_final

    interpolation_applied = False
    if interpolate_for_matching:
        raw_path = os.path.splitext(out_path)[0] + "_raw.png"
        cv2.imwrite(raw_path, img_closed)
        print(f"[Orthophoto interpolation] Saved raw orthomosaic before interpolation: {raw_path}")
        img_closed = interpolate_orthophoto_for_matching(
            img_closed,
            mask_img,
            inpaint_radius=3,
            blur_sigma=0.75,
            blend_alpha=0.55,
        )
        interpolation_applied = True
        interp_debug_path = os.path.splitext(out_path)[0] + "_interpolated_debug.png"
        cv2.imwrite(interp_debug_path, img_closed)
        print(f"[Orthophoto interpolation] Applied valid-mask interpolation for matching: {interp_debug_path}")

    print(f"保存正射图像到 {out_path} ...")
    cv2.imwrite(out_path, img_closed)
    
    mask_path = os.path.splitext(out_path)[0] + "_mask.png"
    cv2.imwrite(mask_path, mask_img)
    print(f"保存对应点云边界遮罩(Mask)到 {mask_path} ...")

    # 保存用于坐标反算的元数据
    meta_path = os.path.splitext(out_path)[0] + "_meta.json"
    meta_data = {
        "u_min": float(u_min),
        "u_max": float(u_max),
        "v_min": float(v_min),
        "v_max": float(v_max),
        "resolution": float(resolution),
        "local_to_meter": float(local_to_meter) if local_to_meter is not None else None,
        "gsd_m": float(resolution * local_to_meter) if local_to_meter is not None else None,
        "target_gsd_m": float(target_gsd_m) if target_gsd_m is not None else None,
        "resolution_source": resolution_source,
        "pixel_cap_applied": bool(pixel_cap_applied),
        "up_axis": up_axis,
        "up_direction": up_direction,
        "width": width,
        "height": height,
        "interpolation_applied": bool(interpolation_applied),
        "interpolation_method": "masked_inpaint_lanczos_gaussian_blend" if interpolation_applied else None,
        "R_align": R_align.tolist(),
        "t_align": t_align.tolist(),
        "h_center": float((h.min() + h.max()) / 2.0)
    }
    if scale_meta is not None:
        meta_data.update(scale_meta)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4)
    print(f"保存正射图像元数据到 {meta_path} ...")

# ==========================================
# 核心算法模块：SR-PCA 与 Masked NCC
# ==========================================
def compute_sr_pca(ref_feat, d=8, valid_mask=None):
    B, C, H, W = ref_feat.shape
    feat_flat_all = ref_feat.view(C, -1).T
    if valid_mask is not None:
        mask_flat = valid_mask.view(-1) > 0.5
        feat_flat = feat_flat_all[mask_flat]
        if feat_flat.shape[0] < d:
            feat_flat = feat_flat_all
            valid_mask = None
    else:
        feat_flat = feat_flat_all

    mu_m = feat_flat.mean(dim=0, keepdim=True)
    feat_centered = feat_flat - mu_m
    U, S, V = torch.pca_lowrank(feat_centered, q=d, center=False)
    ref_proj_flat = torch.matmul(feat_flat_all - mu_m, V)
    ref_proj = ref_proj_flat.T.view(1, d, H, W)
    if valid_mask is not None:
        ref_proj = ref_proj * valid_mask
    return ref_proj, mu_m, V

def _make_dino_patch_mask(mask_img, feat_h, feat_w, padded_h, padded_w, device, min_valid_ratio=0.25):
    if mask_img is None:
        return None

    mask = (mask_img > 127).astype(np.float32)
    pad_h = max(0, padded_h - mask.shape[0])
    pad_w = max(0, padded_w - mask.shape[1])
    if pad_h > 0 or pad_w > 0:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)

    patch_valid = cv2.resize(mask, (feat_w, feat_h), interpolation=cv2.INTER_AREA)
    patch_valid = (patch_valid >= min_valid_ratio).astype(np.float32)
    if patch_valid.sum() < 1:
        return None
    return torch.from_numpy(patch_valid).to(device).view(1, 1, feat_h, feat_w)

def _normalize_dino_features(feat, valid_mask=None):
    eps = 1e-6
    if valid_mask is None:
        mean = feat.mean(dim=(2, 3), keepdim=True)
        std = feat.std(dim=(2, 3), keepdim=True) + eps
        return F.normalize((feat - mean) / std, p=2, dim=1)

    valid_count = valid_mask.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    mean = (feat * valid_mask).sum(dim=(2, 3), keepdim=True) / valid_count
    var = (((feat - mean) * valid_mask) ** 2).sum(dim=(2, 3), keepdim=True) / valid_count
    feat = ((feat - mean) / torch.sqrt(var + eps)) * valid_mask
    return F.normalize(feat, p=2, dim=1) * valid_mask

def masked_ncc_dino_pca(ref_feat, uav_feat, mask):
    eps = 1e-6
    N = mask.sum()
    if N < 1:
        raise ValueError("传入的 Mask 为空！")

    C = ref_feat.shape[1]

    uav_mean = (uav_feat * mask).sum(dim=(2, 3), keepdim=True) / N
    uav_zero_mean = (uav_feat - uav_mean) * mask
    uav_energy = torch.sqrt((uav_zero_mean ** 2).sum() + eps)

    ref_sq_sum_c = (ref_feat ** 2).sum(dim=1, keepdim=True)
    local_sq_sum = F.conv2d(ref_sq_sum_c, mask)

    mask_depthwise = mask.expand(C, 1, mask.shape[2], mask.shape[3])
    local_sum_c = F.conv2d(ref_feat, mask_depthwise, groups=C)
    local_mean_c = local_sum_c / N
    local_mean_energy = N * (local_mean_c ** 2).sum(dim=1, keepdim=True)

    local_variance = local_sq_sum - local_mean_energy
    local_variance = torch.clamp(local_variance, min=0.0)
    ref_energy = torch.sqrt(local_variance + eps)

    numerator = F.conv2d(ref_feat, uav_zero_mean)
    ncc_score_map = numerator / (uav_energy * ref_energy)
    
    return ncc_score_map

def apply_center_prior_to_score_map(score_map, prior_center_xy, template_feat_shape, feat_to_img_scale, ref_resolution_m, radius_m, weight=0.2):
    if prior_center_xy is None or radius_m <= 0:
        return score_map

    _, _, score_h, score_w = score_map.shape
    template_h, template_w = template_feat_shape
    center_x_feat = prior_center_xy[0] / feat_to_img_scale
    center_y_feat = prior_center_xy[1] / feat_to_img_scale
    target_x = center_x_feat - template_w / 2.0
    target_y = center_y_feat - template_h / 2.0

    yy, xx = torch.meshgrid(
        torch.arange(score_h, device=score_map.device, dtype=score_map.dtype),
        torch.arange(score_w, device=score_map.device, dtype=score_map.dtype),
        indexing='ij',
    )
    dist_feat = torch.sqrt((xx - target_x) ** 2 + (yy - target_y) ** 2)
    meters_per_feat = max(feat_to_img_scale * ref_resolution_m, 1e-6)
    dist_m = dist_feat * meters_per_feat

    radius_m = max(float(radius_m), meters_per_feat)
    sigma_m = max(radius_m * 0.5, meters_per_feat)
    penalty = weight * (dist_m / sigma_m) ** 2
    gated_score = score_map - penalty.view(1, 1, score_h, score_w)
    return gated_score.masked_fill((dist_m > radius_m).view(1, 1, score_h, score_w), -1e6)

def resize_image_and_mask_to_gsd(img, mask, source_gsd_m, target_gsd_m, min_size=32):
    if source_gsd_m is None or target_gsd_m is None or source_gsd_m <= 0 or target_gsd_m <= 0:
        raise ValueError(f"Invalid GSD: source={source_gsd_m}, target={target_gsd_m}")

    scale = float(source_gsd_m) / float(target_gsd_m)
    h, w = img.shape[:2]
    new_w = max(min_size, int(round(w * scale)))
    new_h = max(min_size, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    mask_resized = None
    if mask is not None:
        mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return img_resized, mask_resized, scale

def read_orthophoto_gsd_m(ortho_path, fallback_gsd_m=None):
    meta_path = os.path.splitext(ortho_path)[0] + "_meta.json"
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            gsd_m = _read_float(meta, ["gsd_m"])
            if gsd_m is None:
                resolution = _read_float(meta, ["resolution"])
                local_to_meter = _read_float(meta, ["local_to_meter"])
                if resolution is not None and local_to_meter is not None:
                    gsd_m = resolution * local_to_meter
            if gsd_m is not None and gsd_m > 0:
                return float(gsd_m), "orthophoto_meta"
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"[BEV scale] Failed to read orthophoto meta GSD: {e}")

    if fallback_gsd_m is not None and fallback_gsd_m > 0:
        return float(fallback_gsd_m), "fallback_camera_gsd"
    return None, "unknown"

def rotate_image_and_mask_fixed(img, mask, angle_deg):
    if abs(float(angle_deg)) < 1e-9:
        return img, mask
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    img_rot = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mask_rot = None
    if mask is not None:
        mask_rot = cv2.warpAffine(
            mask,
            M,
            (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return img_rot, mask_rot

def make_structure_gray_for_match(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    max_val = float(mag.max())
    if max_val <= 1e-6:
        return np.zeros_like(mag, dtype=np.float32)
    return (mag / max_val).astype(np.float32)

def subpixel_peak_2d(score_map, x, y):
    h, w = score_map.shape[:2]
    dx = 0.0
    dy = 0.0
    if 0 < x < w - 1:
        left = float(score_map[y, x - 1])
        center = float(score_map[y, x])
        right = float(score_map[y, x + 1])
        denom = left - 2.0 * center + right
        if abs(denom) > 1e-9:
            dx = float(np.clip(0.5 * (left - right) / denom, -1.0, 1.0))
    if 0 < y < h - 1:
        top = float(score_map[y - 1, x])
        center = float(score_map[y, x])
        bottom = float(score_map[y + 1, x])
        denom = top - 2.0 * center + bottom
        if abs(denom) > 1e-9:
            dy = float(np.clip(0.5 * (top - bottom) / denom, -1.0, 1.0))
    return dx, dy

def refine_template_match_pixel_ncc(ref_img, template_img, template_mask, coarse_top_left_xy, search_radius_px=96):
    th, tw = template_img.shape[:2]
    rh, rw = ref_img.shape[:2]
    if th < 8 or tw < 8 or rh < th or rw < tw:
        return None

    coarse_x, coarse_y = float(coarse_top_left_xy[0]), float(coarse_top_left_xy[1])
    radius = int(max(1, round(search_radius_px)))
    x1 = max(0, int(np.floor(coarse_x - radius)))
    y1 = max(0, int(np.floor(coarse_y - radius)))
    x2 = min(rw, int(np.ceil(coarse_x + radius + tw)))
    y2 = min(rh, int(np.ceil(coarse_y + radius + th)))
    if x2 - x1 < tw or y2 - y1 < th:
        return None

    search_gray = make_structure_gray_for_match(ref_img[y1:y2, x1:x2])
    template_gray = make_structure_gray_for_match(template_img)
    if template_mask is None:
        mask = np.ones((th, tw), dtype=np.float32)
    else:
        if template_mask.shape[:2] != (th, tw):
            template_mask = cv2.resize(template_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
        mask = (template_mask > 127).astype(np.float32)

    valid_count = float(mask.sum())
    if valid_count < max(64.0, th * tw * 0.03):
        return None

    template_mean = float((template_gray * mask).sum() / valid_count)
    template_zero = (template_gray - template_mean) * mask
    template_norm = float(np.sqrt((template_zero * template_zero).sum()))
    if template_norm <= 1e-6:
        return None

    sum_i = cv2.matchTemplate(search_gray, mask, cv2.TM_CCORR)
    sum_i2 = cv2.matchTemplate(search_gray * search_gray, mask, cv2.TM_CCORR)
    numerator = cv2.matchTemplate(search_gray, template_zero, cv2.TM_CCORR)
    variance_i = np.maximum(sum_i2 - (sum_i * sum_i) / valid_count, 1e-9)
    score = numerator / (template_norm * np.sqrt(variance_i))
    score[~np.isfinite(score)] = -np.inf
    _, max_score, _, max_loc = cv2.minMaxLoc(score)
    if not np.isfinite(max_score):
        return None

    peak_x, peak_y = max_loc
    sub_dx, sub_dy = subpixel_peak_2d(score, peak_x, peak_y)
    return {
        "x": float(x1 + peak_x + sub_dx),
        "y": float(y1 + peak_y + sub_dy),
        "score": float(max_score),
        "score_map": score,
        "search_origin": (int(x1), int(y1)),
    }

def _robust_normalize_u8(arr, valid_mask=None, low_pct=1.0, high_pct=99.0):
    arr = arr.astype(np.float32)
    if valid_mask is not None and np.any(valid_mask > 127):
        vals = arr[valid_mask > 127]
    else:
        vals = arr.reshape(-1)
    lo, hi = np.percentile(vals, [low_pct, high_pct])
    if hi <= lo + 1e-6:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr - lo) * 255.0 / (hi - lo), 0, 255)
    return out.astype(np.uint8)

def make_dino_pseudo_rgb(img_bgr, valid_mask=None, edge_mix=0.30):
    """
    DINO input for thermal/RGB matching. This keeps intensity and low-frequency
    structure, while using edges only weakly so matching is not dominated by
    outlines, mask borders, or thermal speckles.
    """
    if img_bgr.ndim == 2:
        gray = img_bgr.copy()
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    mask_bool = valid_mask > 127 if valid_mask is not None and np.any(valid_mask > 127) else None
    if mask_bool is not None:
        gray = gray.copy()
        gray[~mask_bool] = int(np.median(gray[mask_bool]))

    gray = _robust_normalize_u8(gray, valid_mask)
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=5)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(denoised)
    low_freq = cv2.GaussianBlur(clahe, (0, 0), sigmaX=2.0)

    gx = cv2.Sobel(low_freq, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(low_freq, cv2.CV_32F, 0, 1, ksize=3)
    edge = cv2.magnitude(gx, gy)
    edge = _robust_normalize_u8(edge, valid_mask, low_pct=2.0, high_pct=98.5)
    if mask_bool is not None:
        kernel = np.ones((7, 7), np.uint8)
        mask_u8 = mask_bool.astype(np.uint8)
        boundary = cv2.dilate(mask_u8, kernel) != cv2.erode(mask_u8, kernel)
        edge[boundary] = int(np.median(edge[mask_bool]))

    mixed_edge = cv2.addWeighted(low_freq, 1.0 - edge_mix, edge, edge_mix, 0)
    detail = cv2.addWeighted(clahe, 1.15, low_freq, -0.15, 0)
    pseudo_rgb = cv2.merge([mixed_edge, low_freq, detail])

    if mask_bool is not None:
        fill_color = np.median(pseudo_rgb[mask_bool], axis=0).astype(np.uint8)
        pseudo_rgb[~mask_bool] = fill_color
    return pseudo_rgb

def make_satellite_thermal_gray(img_bgr, valid_mask=None):
    """
    Convert satellite RGB to a thermal-like grayscale image. This intentionally
    avoids edge-only maps: DINO sees broad brightness/texture regions similar to
    the thermal orthomosaic, while color-specific satellite cues are removed.
    """
    if img_bgr.ndim == 2:
        gray = img_bgr.copy()
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if valid_mask is not None and np.any(valid_mask > 127):
        mask_bool = valid_mask > 127
        gray = gray.copy()
        gray[~mask_bool] = int(np.median(gray[mask_bool]))
    else:
        mask_bool = None

    gray = _robust_normalize_u8(gray, valid_mask, low_pct=1.0, high_pct=99.0)
    gray = cv2.bilateralFilter(gray, d=5, sigmaColor=30, sigmaSpace=5)
    gray = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8)

    if mask_bool is not None:
        gray[~mask_bool] = int(np.median(gray[mask_bool]))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def make_raw_thermal_bgr(img_bgr, valid_mask=None):
    """
    Keep the orthomosaic thermal appearance. Only normalize robustly and fill the
    invalid background so mask borders do not become a matching cue.
    """
    if img_bgr.ndim == 2:
        gray = img_bgr.copy()
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if valid_mask is not None and np.any(valid_mask > 127):
        mask_bool = valid_mask > 127
        gray = gray.copy()
        gray[~mask_bool] = int(np.median(gray[mask_bool]))
    else:
        mask_bool = None

    gray = _robust_normalize_u8(gray, valid_mask, low_pct=1.0, high_pct=99.0)
    if mask_bool is not None:
        gray[~mask_bool] = int(np.median(gray[mask_bool]))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

def extract_dino_features_chunked(dinov2, device, img_bgr, patch_size=14, global_scale=1.0, valid_mask=None, use_edge_enhancement=True, debug_vis_path=None, preprocess_mode=None):
    img_bgr = img_bgr.copy()

    if preprocess_mode is None:
        preprocess_mode = "pseudo_rgb" if use_edge_enhancement else "raw"

    if preprocess_mode == "pseudo_rgb":
        img_bgr = make_dino_pseudo_rgb(img_bgr, valid_mask=valid_mask, edge_mix=0.30)
    elif preprocess_mode == "satellite_thermal_gray":
        img_bgr = make_satellite_thermal_gray(img_bgr, valid_mask=valid_mask)
    elif preprocess_mode == "raw_thermal":
        img_bgr = make_raw_thermal_bgr(img_bgr, valid_mask=valid_mask)

    if debug_vis_path is not None:
        cv2.imwrite(debug_vis_path, img_bgr)
    
    img_h, img_w = img_bgr.shape[:2]
    if global_scale != 1.0:
        img_bgr = cv2.resize(img_bgr, (int(img_w * global_scale), int(img_h * global_scale)))
        if valid_mask is not None:
            valid_mask = cv2.resize(valid_mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
            
    # ------ 修复方案 1：用均值颜色填补黑边，抑制黑边带来的边缘高频高响应 (Padding Effect) ------
    if valid_mask is not None:
        mask_bool = valid_mask > 127
        if mask_bool.any():
            mean_color = img_bgr[mask_bool].mean(axis=0)
            img_bgr[~mask_bool] = mean_color
    # ----------------------------------------------------------------------------------------

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    img_tensor = normalize(img_tensor).unsqueeze(0).to(device)
    
    B, C, H, W = img_tensor.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        
    B, C, H, W = img_tensor.shape
    chunk_size = 1008
    stride = chunk_size 
    h_feat, w_feat = H // patch_size, W // patch_size
    embed_dim = getattr(dinov2, 'embed_dim', 1024)
    feat_out = torch.zeros((1, embed_dim, h_feat, w_feat), device='cpu') 
    
    with torch.no_grad():
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                y_end = min(H, y + chunk_size)
                x_end = min(W, x + chunk_size)
                
                img_chunk = img_tensor[:, :, y:y_end, x:x_end]
                ch_pad_h = (patch_size - img_chunk.shape[2] % patch_size) % patch_size
                ch_pad_w = (patch_size - img_chunk.shape[3] % patch_size) % patch_size
                if ch_pad_h > 0 or ch_pad_w > 0:
                    img_chunk = F.pad(img_chunk, (0, ch_pad_w, 0, ch_pad_h), mode='reflect')
                
                feat_dict = dinov2.forward_features(img_chunk)
                chunk_feat = feat_dict['x_norm_patchtokens']
                
                ch_feat_h, ch_feat_w = img_chunk.shape[2] // patch_size, img_chunk.shape[3] // patch_size
                chunk_feat = chunk_feat.reshape(1, ch_feat_h, ch_feat_w, -1).permute(0, 3, 1, 2)
                
                valid_h = (y_end - y) // patch_size
                valid_w = (x_end - x) // patch_size
                y_f, x_f = y // patch_size, x // patch_size
                feat_out[:, :, y_f:y_f+valid_h, x_f:x_f+valid_w] = chunk_feat[:, :, :valid_h, :valid_w].cpu()

    feat_out = feat_out.to(device)
    patch_mask = _make_dino_patch_mask(valid_mask, h_feat, w_feat, H, W, device)
    feat_out = _normalize_dino_features(feat_out, patch_mask)
    return feat_out

def _read_float(meta, keys, default=None):
    for key in keys:
        if meta is None or key not in meta:
            continue
        value = meta[key]
        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    if default is None:
        return None
    return float(default)

def get_camera_intrinsics_from_meta(meta):
    """
    从 DJI/手工配置字段恢复 OpenCV 内参。
    cam_size 按传感器宽度(mm)处理；如果没有显式 cy/cx，则默认主点在图像中心。
    """
    width = _read_float(meta, ["width", "image_width", "img_width", "cols"])
    height = _read_float(meta, ["height", "image_height", "img_height", "rows"])
    if width is None or height is None:
        raise ValueError("缺少图像 width/height，无法计算相机内参")

    fx = _read_float(meta, ["fx", "focal_x", "focal_len_px", "focal_length_px"])
    fy = _read_float(meta, ["fy", "focal_y"])

    if fx is None or fy is None:
        focal_mm = _read_float(meta, ["focal_len", "focal_length", "focal_mm"])
        sensor_w = _read_float(meta, ["sensor_width", "sensor_width_mm", "cam_size"])
        sensor_h = _read_float(meta, ["sensor_height", "sensor_height_mm"])
        if sensor_h is None and sensor_w is not None:
            sensor_h = sensor_w * height / width
        if focal_mm is None or sensor_w is None or sensor_h is None:
            raise ValueError("缺少 focal_len/cam_size 或 fx/fy，无法计算相机内参")
        fx = focal_mm / sensor_w * width
        fy = focal_mm / sensor_h * height

    cx = _read_float(meta, ["cx", "principal_x", "principal_point_x", "ppx"], width / 2.0)
    cy = _read_float(meta, ["cy", "principal_y", "principal_point_y", "ppy"], height / 2.0)
    return {
        "width": width,
        "height": height,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }

def estimate_nadir_gsd_from_meta(meta):
    rel_height = _read_float(meta, ["rel_height", "alt"], 0.0)
    focal_mm = _read_float(meta, ["focal_len", "focal_length", "focal_mm"])
    sensor_w = _read_float(meta, ["sensor_width", "sensor_width_mm", "cam_size"])
    width = _read_float(meta, ["width", "image_width", "img_width", "cols"])
    if focal_mm is None or sensor_w is None or width is None or focal_mm == 0 or width == 0:
        raise ValueError("缺少 rel_height/focal_len/cam_size/width，无法估计 GSD")
    return (rel_height * sensor_w) / (focal_mm * width)

def project_image_pixel_to_ground_utm(camera_utm_x, camera_utm_y, pose_meta, pixel_xy=None):
    """
    将图像某个像素的相机射线与局部水平地面相交，返回卫星图上的 UTM 坐标。
    约定：yaw 以北为 0 顺时针为正；DJI pitch=-90 为垂直向下，0 为看向地平线。
    """
    intr = get_camera_intrinsics_from_meta(pose_meta)
    yaw_deg = _read_float(pose_meta, ["yaw", "gb_yaw"], 0.0)
    pitch_deg = _read_float(pose_meta, ["pitch", "gb_pitch"], -90.0)
    roll_deg = _read_float(pose_meta, ["roll", "gb_roll"], 0.0)
    rel_height = _read_float(pose_meta, ["rel_height", "alt"], 0.0)

    if rel_height <= 0:
        raise ValueError(f"相对高度无效: {rel_height}")

    if pixel_xy is None:
        u = intr["width"] / 2.0
        v = intr["height"] / 2.0
    else:
        u, v = float(pixel_xy[0]), float(pixel_xy[1])

    x_norm = (u - intr["cx"]) / intr["fx"]
    y_norm = (v - intr["cy"]) / intr["fy"]

    depression_deg = -pitch_deg
    if depression_deg <= 0:
        raise ValueError(f"pitch={pitch_deg:.2f}° 未指向地面，无法投影到地面")

    yaw_rad = np.deg2rad(yaw_deg)
    dep_rad = np.deg2rad(depression_deg)

    forward = np.array([
        np.sin(yaw_rad) * np.cos(dep_rad),
        np.cos(yaw_rad) * np.cos(dep_rad),
        -np.sin(dep_rad),
    ], dtype=np.float64)
    right = np.array([np.cos(yaw_rad), -np.sin(yaw_rad), 0.0], dtype=np.float64)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-12)

    roll_rad = np.deg2rad(roll_deg)
    right_roll = np.cos(roll_rad) * right + np.sin(roll_rad) * down
    down_roll = -np.sin(roll_rad) * right + np.cos(roll_rad) * down

    ray = forward + x_norm * right_roll + y_norm * down_roll
    ray /= max(np.linalg.norm(ray), 1e-12)
    if ray[2] >= -1e-6:
        raise ValueError(f"像素射线没有打到地面: ray_z={ray[2]:.6f}")

    scale = rel_height / (-ray[2])
    east_offset = scale * ray[0]
    north_offset = scale * ray[1]
    return {
        "utm_x": float(camera_utm_x + east_offset),
        "utm_y": float(camera_utm_y + north_offset),
        "east_offset": float(east_offset),
        "north_offset": float(north_offset),
        "horizontal_distance": float(np.hypot(east_offset, north_offset)),
        "pixel": (float(u), float(v)),
        "intrinsics": intr,
        "yaw": float(yaw_deg),
        "pitch": float(pitch_deg),
        "roll": float(roll_deg),
        "rel_height": float(rel_height),
    }

def project_image_pixel_to_z_plane_offset_utm(camera_z, target_z, pose_meta, pixel_xy=None):
    """
    返回从相机中心到指定绝对高度 Z 平面的图像射线水平偏移。
    用于把匹配到的3D表面点反推出相机XY；不要把屋顶/树冠点当成地面交点。
    """
    intr = get_camera_intrinsics_from_meta(pose_meta)
    if pixel_xy is None:
        u = intr["width"] / 2.0
        v = intr["height"] / 2.0
    else:
        u, v = float(pixel_xy[0]), float(pixel_xy[1])

    x_norm = (u - intr["cx"]) / intr["fx"]
    y_norm = (v - intr["cy"]) / intr["fy"]
    right_axis, down_axis, forward_axis = _camera_basis_enu_from_pose_meta(pose_meta)
    ray = forward_axis + x_norm * right_axis + y_norm * down_axis
    ray /= max(np.linalg.norm(ray), 1e-12)
    if abs(ray[2]) < 1e-9:
        raise ValueError(f"像素射线近似平行于Z平面: ray_z={ray[2]:.9f}")

    scale = (float(target_z) - float(camera_z)) / float(ray[2])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"像素射线无法与目标Z平面形成前向交点: camera_z={float(camera_z):.3f}, "
            f"target_z={float(target_z):.3f}, ray_z={float(ray[2]):.6f}, scale={float(scale):.3f}"
        )
    east_offset = scale * ray[0]
    north_offset = scale * ray[1]
    return {
        "east_offset": float(east_offset),
        "north_offset": float(north_offset),
        "horizontal_distance": float(np.hypot(east_offset, north_offset)),
        "pixel": (float(u), float(v)),
        "camera_z": float(camera_z),
        "target_z": float(target_z),
        "ray": ray.astype(float).tolist(),
        "ray_scale": float(scale),
    }

def estimate_image_ground_footprint_size_m(camera_utm_x, camera_utm_y, pose_meta):
    metrics = estimate_image_ground_footprint_metrics_m(camera_utm_x, camera_utm_y, pose_meta)
    if metrics is None:
        return None
    return metrics["max_extent_m"]

def estimate_image_ground_footprint_metrics_m(camera_utm_x, camera_utm_y, pose_meta):
    intr = get_camera_intrinsics_from_meta(pose_meta)
    corners = [
        (0.0, 0.0),
        (intr["width"] - 1.0, 0.0),
        (intr["width"] - 1.0, intr["height"] - 1.0),
        (0.0, intr["height"] - 1.0),
    ]
    projected = []
    for pt in corners:
        try:
            projected.append(project_image_pixel_to_ground_utm(camera_utm_x, camera_utm_y, pose_meta, pt))
        except ValueError:
            continue
    if len(projected) < 4:
        return None
    xs = np.array([p["utm_x"] for p in projected], dtype=np.float64)
    ys = np.array([p["utm_y"] for p in projected], dtype=np.float64)

    def _dist(i, j):
        return float(np.hypot(xs[i] - xs[j], ys[i] - ys[j]))

    top_m = _dist(0, 1)
    right_m = _dist(1, 2)
    bottom_m = _dist(2, 3)
    left_m = _dist(3, 0)
    image_width_m = max(top_m, bottom_m)
    image_height_m = max(left_m, right_m)
    bbox_width_m = float(xs.max() - xs.min())
    bbox_height_m = float(ys.max() - ys.min())
    return {
        "image_width_m": image_width_m,
        "image_height_m": image_height_m,
        "bbox_width_m": bbox_width_m,
        "bbox_height_m": bbox_height_m,
        "max_extent_m": float(max(bbox_width_m, bbox_height_m, image_width_m, image_height_m)),
        "corners": [(float(p["utm_x"]), float(p["utm_y"])) for p in projected],
    }

def run_ortho_matching_and_get_coords(ortho_path, save_dir, region, opt, region_config, ref_map_full, dsm_map_full, truePos):
    """
    运行 DINO 特征匹配，并返回匹配后得到的绝对全球坐标 (pred_utm_x, pred_utm_y) 
    以及旋转偏移量 best_yaw_offset
    """
    seed_everything(GLOBAL_RANDOM_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"正在加载需要匹配的正射影像: {ortho_path}")
    uav_image = cv2.imread(ortho_path)
    if uav_image is None:
        raise ValueError(f"读取正射影像失败: {ortho_path}")

    mask_path = os.path.splitext(ortho_path)[0] + "_mask.png"
    uav_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if uav_mask is None:
        print(f"[警告] 未找到生成的专属 Mask 文件(推测为从旧流程读取) {mask_path}，将依赖图像灰度估计掩膜...")

    # 直接借用 temp_config 中在外面裁切或读取时可能校正过的 ref_resolution
    # 或直接使用我们在上方修改的精确值
    # 由于真正重要的是 ref_map_crop 的生成分辨率，所以我们可以回退到 temp_config
    # 之前是强行读配置
    # ref_resolution = region_config.get(f'{region}_{opt.Ref_type}_REF_resolution', 0.3)
    
    # 卫星图由于已经在外面通过 load_config_parameters_new 可能获得TIF里的精确值，并在 crop 时产生了 ref_map_crop
    # 我们可以通过 opt.ref_resolution 或这里直接计算
    sat_ref_resolution = opt.ref_resolution if hasattr(opt, "ref_resolution") else region_config.get(f'{region}_{opt.Ref_type}_REF_resolution', 0.3)
    ref_resolution = sat_ref_resolution

    print(f"裁剪参考卫星地图 (范围 {opt.crop_size_m}x{opt.crop_size_m} m)...")
    ref_map_crop, dsm_map_crop, temp_config = crop_geo_images_with_offset(
        region, region_config, opt, truePos, ref_map_full, dsm_map_full
    )
    
    if opt.pose_priori == 'yp':
        ref_map_rot, matRotation = dumpRotateImage(ref_map_crop, truePos['yaw'])
        ref_mask_rot, _ = dumpRotateImage(np.full(ref_map_crop.shape[:2], 255, dtype=np.uint8), truePos['yaw'])
    else:
        ref_map_rot = ref_map_crop
        ref_mask_rot = np.full(ref_map_crop.shape[:2], 255, dtype=np.uint8)
        matRotation = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    print(f"加载 DINOv2 模型...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device).eval()

    uav_h, uav_w = uav_image.shape[:2]

    # Calculate actual satellite resolution from crop spatial size and crop pixel dimensions (before any rotation)
    crop_h, crop_w = ref_map_crop.shape[:2]
    sat_res = opt.crop_size_m / max(crop_h, crop_w)

    fallback_uav_res = (truePos['rel_height'] * truePos['cam_size']) / (truePos['focal_len'] * truePos['width'])
    uav_res, uav_res_source = read_orthophoto_gsd_m(ortho_path, fallback_uav_res)
    if uav_res is None:
        raise ValueError("Cannot infer orthophoto GSD for template matching.")
    print(
        f"[GSD] orthophoto source={uav_res_source}, source_gsd={uav_res:.4f} m/px, "
        f"satellite_gsd={sat_res:.4f} m/px"
    )

    if uav_mask is None:
        uav_mask = (cv2.cvtColor(uav_image, cv2.COLOR_BGR2GRAY) > 5).astype(np.uint8) * 255
    uav_scaled, uav_mask_scaled, scale_factor = resize_image_and_mask_to_gsd(
        uav_image,
        uav_mask,
        source_gsd_m=uav_res,
        target_gsd_m=sat_res,
    )
    print(
        f"[GSD对齐] 正射模板: source_gsd={uav_res:.4f} m/px, "
        f"sat_gsd={sat_res:.4f} m/px, resize_scale={scale_factor:.3f}, "
        f"{uav_w}x{uav_h} -> {uav_scaled.shape[1]}x{uav_scaled.shape[0]}"
    )
    
    ref_h, ref_w = ref_map_rot.shape[:2]
    global_scale = 1.0
    # ------ 修复方案 2：放宽特征提取阶段被强制缩放的约束，保留更多特征格网数目 ------
    if max(ref_h, ref_w) > 4000:
        global_scale = 4000.0 / max(ref_h, ref_w)

    ref_feat_high_dim = extract_dino_features_chunked(
        dinov2,
        device,
        ref_map_rot,
        global_scale=global_scale,
        valid_mask=ref_mask_rot,
        debug_vis_path=os.path.join(save_dir, 'DinoTemplate_Ref_Edge.jpg'),
        preprocess_mode="satellite_thermal_gray"
    )
    
    pca_dim = 8
    ref_patch_mask = _make_dino_patch_mask(
        cv2.resize(ref_mask_rot, (ref_feat_high_dim.shape[3] * 14, ref_feat_high_dim.shape[2] * 14), interpolation=cv2.INTER_NEAREST),
        ref_feat_high_dim.shape[2],
        ref_feat_high_dim.shape[3],
        ref_feat_high_dim.shape[2] * 14,
        ref_feat_high_dim.shape[3] * 14,
        device,
    )
    ref_feat, global_mu, projection_matrix = compute_sr_pca(ref_feat_high_dim, d=pca_dim, valid_mask=ref_patch_mask)
    del ref_feat_high_dim 
    torch.cuda.empty_cache()
    
    yaw_search_range = [-20, -16, -12, -9, -6, -3, 0, 3, 6, 9, 12, 16, 20]
    best_score = -1
    best_yaw_offset = 0
    best_x_feat, best_y_feat = 0, 0
    best_uav_feat_shape = None

    print(f"进行三自由度特征滑动窗口搜索...")
    for dyaw in yaw_search_range:
        if dyaw != 0:
            M_rot = cv2.getRotationMatrix2D((uav_scaled.shape[1]/2, uav_scaled.shape[0]/2), dyaw, 1.0)
            uav_rotated = cv2.warpAffine(uav_scaled, M_rot, (uav_scaled.shape[1], uav_scaled.shape[0]))
            uav_mask_rotated = cv2.warpAffine(uav_mask_scaled, M_rot, (uav_scaled.shape[1], uav_scaled.shape[0]))
        else:
            uav_rotated = uav_scaled
            uav_mask_rotated = uav_mask_scaled
        
        uav_feat_high_dim = extract_dino_features_chunked(
            dinov2,
            device,
            uav_rotated,
            global_scale=global_scale,
            valid_mask=uav_mask_rotated,
            debug_vis_path=os.path.join(save_dir, 'DinoTemplate_UAV_Edge.jpg') if dyaw == 0 else None,
            preprocess_mode="raw_thermal"
        )
        
        b, c_high, h_f, w_f = uav_feat_high_dim.shape
        uav_proj_flat = torch.matmul((uav_feat_high_dim.view(c_high, -1).T - global_mu), projection_matrix)
        uav_feat = uav_proj_flat.T.view(1, pca_dim, h_f, w_f)
        
        # 使用精准的物理边界 Mask (避免热成像中低温物体被误当作空洞丢弃)
        mask_np = (cv2.resize(uav_mask_rotated, (w_f, h_f), interpolation=cv2.INTER_NEAREST) > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_np).to(device).view(1, 1, h_f, w_f)
        
        try:
            score_map = masked_ncc_dino_pca(ref_feat, uav_feat, mask_tensor)
        except RuntimeError as e:
            continue
            
        max_val = score_map.max().item()
        if max_val > best_score:
            best_score = max_val
            best_yaw_offset = dyaw
            max_idx = score_map.argmax().item()
            best_y_feat, best_x_feat = divmod(max_idx, score_map.shape[3])
            best_uav_feat_shape = uav_feat.shape[2:]
            best_score_map = score_map.squeeze(0).squeeze(0).cpu().numpy()
            best_uav_rotated = uav_rotated
            best_uav_feat = uav_feat

    if best_score > -1:
        print(f"------ [特征级匹配结束] 最强NCC响应得分: {best_score:.4f}, Yaw局部偏移: {best_yaw_offset}度 ------")
        
        # === 新增可视化 ===
        # 1. 响应热力图 (Heatmap)
        plt.figure(figsize=(10, 8))
        plt.imshow(best_score_map, cmap='jet')
        plt.colorbar(label='NCC Similarity')
        plt.title(f'Template Matching Response (Yaw: {best_yaw_offset}°)')
        heatmap_path = os.path.join(save_dir, 'DinoTemplate_Heatmap.png')
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"已保存特征响应热力图至: {heatmap_path}")
        
        # 2. PCA降维可视化
        print("正在生成DINO特征PCA降维空间可视化...")
        def save_pca_vis(feat, out_path, pca_model=None):
            """单主成分 Viridis 配色可视化 (对标图中效果)"""
            d, h, w = feat.shape[1:]
            f_flat = feat.squeeze(0).permute(1, 2, 0).cpu().numpy().reshape(-1, d)
            
            # 1. 改为只取第 1 个主成分 (单通道)
            if pca_model is None:
                pca_model = PCA(n_components=1)
                f_pca = pca_model.fit_transform(f_flat)
            else:
                f_pca = pca_model.transform(f_flat)
            
            # 2. 鲁棒归一化 (极其重要：使用 1% 和 99% 分位数截断离群点)
            # 如果用你原来的 min 和 max，图中个别极值会让整张图显得很暗/对比度极低
            q_low, q_high = np.percentile(f_pca, [1, 99])
            f_pca_norm = np.clip((f_pca - q_low) / (q_high - q_low + 1e-6), 0, 1)
            
            # 3. 转为 0-255 的单通道灰度图矩阵
            f_pca_gray = (f_pca_norm * 255).astype(np.uint8).reshape(h, w)
            
            # 4. 放大图像 (将 INTER_NEAREST 改为 INTER_CUBIC，消除马赛克，获得图中平滑的视觉效果)
            f_pca_resized = cv2.resize(f_pca_gray, (w * 14, h * 14), interpolation=cv2.INTER_CUBIC)
            
            # 5. 应用 Viridis 伪彩色映射 (这就是你图中深蓝-绿-黄的同款配色)
            f_pca_color = cv2.applyColorMap(f_pca_resized, cv2.COLORMAP_VIRIDIS)
            
            # OpenCV 的 applyColorMap 默认输出就是 BGR 格式，直接 imwrite 保存即可
            cv2.imwrite(out_path, f_pca_color)
            
            return pca_model

        pca_model = save_pca_vis(ref_feat, os.path.join(save_dir, 'DinoTemplate_Ref_PCA.png'))
        save_pca_vis(best_uav_feat, os.path.join(save_dir, 'DinoTemplate_UAV_PCA.png'), pca_model)
        
        # 3. 将匹配红框直接绘制在底图上
        map_vis = ref_map_rot.copy()
        effective_patch_size = 14
        top_left_x_scaled = best_x_feat * effective_patch_size
        top_left_y_scaled = best_y_feat * effective_patch_size
        
        top_left_x = int(top_left_x_scaled / global_scale)
        top_left_y = int(top_left_y_scaled / global_scale)
        
        # 已经在原尺寸，无需再除非global_scale
        h_uav, w_uav = best_uav_rotated.shape[:2]
        
        cv2.rectangle(map_vis, (top_left_x, top_left_y), (top_left_x + w_uav, top_left_y + h_uav), (0, 0, 255), 5)
        cv2.circle(map_vis, (top_left_x + w_uav // 2, top_left_y + h_uav // 2), 10, (0, 255, 0), -1)
        
        match_box_path = os.path.join(save_dir, 'DinoTemplate_MatchBox.jpg')
        cv2.imwrite(match_box_path, map_vis)
        print(f"已保存底图匹配红框框选可视化至: {match_box_path}")
        # === 可视化结束 ===

        effective_patch_size = 14
        ref_x_scaled = int(best_x_feat * effective_patch_size + (best_uav_feat_shape[1] * effective_patch_size) / 2)
        ref_y_scaled = int(best_y_feat * effective_patch_size + (best_uav_feat_shape[0] * effective_patch_size) / 2)
        ref_x, ref_y = ref_x_scaled / global_scale, ref_y_scaled / global_scale
        
        M_inv = cv2.invertAffineTransform(matRotation)
        center_ref_crop_global = M_inv @ np.array([ref_x, ref_y, 1.0]).reshape(-1, 1)
        final_x, final_y = center_ref_crop_global[0][0], center_ref_crop_global[1][0]
        
        ref_origin_utm_X = temp_config[f"{region}_{opt.Ref_type}_REF_initialX"]
        ref_origin_utm_Y = temp_config[f"{region}_{opt.Ref_type}_REF_initialY"]
        
        pred_utm_x = ref_origin_utm_X + final_x * ref_resolution
        pred_utm_y = ref_origin_utm_Y - final_y * ref_resolution
        
        # 释放显存
        del dinov2, ref_feat, uav_feat
        torch.cuda.empty_cache()
        
        pred_lon, pred_lat = utm2deg(region, temp_config, pred_utm_x, pred_utm_y)
        print(f"\n[匹配完成] 得分: {best_score:.4f}, Yaw偏移: {best_yaw_offset}°")
        print(f"匹配 UTM: X={pred_utm_x:.2f}, Y={pred_utm_y:.2f} \n经纬度: Lat={pred_lat:.6f}, Lon={pred_lon:.6f}")
        
        # 保存正射匹配与定位的可视化结果
        pred_pos = {'lon': pred_lon, 'lat': pred_lat, 'yaw': truePos['yaw'] + best_yaw_offset}
        vis_path = os.path.join(save_dir, 'Localization_Visual_DinoTemplate.jpg')
        # visualize_final_localization(ref_map_crop, dsm_map_crop, uav_image, truePos, pred_pos, temp_config, region, vis_path, opt)
        print(f"DINO模板匹配可视化大图已被临时注释")
        
        # === 新增: 在底图上直观标记真实与预测坐标及朝向 ===
        map_arrow_vis = ref_map_crop.copy()
        
        # 计算真实 UTM -> 像素坐标
        true_utm_x, true_utm_y = deg2utm(region, temp_config, truePos['lon'], truePos['lat'])
        true_x = int((true_utm_x - ref_origin_utm_X) / ref_resolution)
        true_y = int((ref_origin_utm_Y - true_utm_y) / ref_resolution)
        
        # 预测坐标已经在 final_x, final_y 中 (相对 ref_map_crop 的像素)
        pred_x = int(final_x)
        pred_y = int(final_y)
        
        def draw_yaw_arrow(img, cx, cy, yaw_deg, color, label, length=400, thickness=12):
            import math
            rad = math.radians(yaw_deg)
            # 在图像系：正北为 -Y (向上), 正东为 +X (向右), 偏航 Yaw 为正北顺时针
            dx = int(length * math.sin(rad))
            dy = int(-length * math.cos(rad))
            cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), color, thickness, tipLength=0.2)
            cv2.circle(img, (cx, cy), thickness*2, color, -1)
            cv2.putText(img, label, (cx + dx + 20, cy + dy + 20), cv2.FONT_HERSHEY_SIMPLEX, 4.0, color, 6)

        # 绿色的先验真值
        draw_yaw_arrow(map_arrow_vis, true_x, true_y, truePos['yaw'], (0, 255, 0), "True")
        # 红色的预测结果
        total_yaw = truePos['yaw'] + best_yaw_offset
        draw_yaw_arrow(map_arrow_vis, pred_x, pred_y, total_yaw, (0, 0, 255), "Pred")
        # 黄色线段连接真实与预测的位置对应偏移量
        cv2.line(map_arrow_vis, (true_x, true_y), (pred_x, pred_y), (0, 255, 255), 6)
        
        arrow_path = os.path.join(save_dir, 'DinoTemplate_Direction_Compare.jpg')
        cv2.imwrite(arrow_path, map_arrow_vis)
        print(f"带偏航方向和位置标注的对比清晰图已保存至: {arrow_path}")

        # 将最佳 NCC 得分写入文件，供序贯 Pipeline 读取
        score_path = os.path.join(save_dir, 'match_score.json')
        with open(score_path, 'w') as _sf:
            json.dump({"score": float(best_score)}, _sf)

        return pred_utm_x, pred_utm_y, best_yaw_offset
    else:
        raise RuntimeError("未能计算出任何有效的特征匹配得分，匹配失败！")

# ==========================================
# 位姿恢复与纯预测推导对齐模块
# ==========================================

from scipy.spatial.transform import Rotation as R

def load_scal3r_poses(extri_path):
    """
    解析 Scal3R 输出的 OpenCV YAML 格式的外参文件 (包含多帧的相对旋转和相对平移)
    """
    fs = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise FileNotFoundError(f"无法打开外参文件: {extri_path}")
    
    names_node = fs.getNode("names")
    if names_node.empty():
        raise ValueError(f"格式错误，找不到 'names' 节点: {extri_path}")
        
    names = [names_node.at(i).string() for i in range(names_node.size())]
    
    poses = {}
    for name in names:
        rot_mat = fs.getNode(f"Rot_{name}").mat()
        t_vec = fs.getNode(f"T_{name}").mat()
        
        if rot_mat is None or t_vec is None:
            continue
            
        poses[name] = {
            "R": rot_mat,
            "T": t_vec.flatten()
        }
        
    fs.release()
    return names, poses

def _csv_frame_stem(filename):
    return os.path.splitext(os.path.basename(str(filename)))[0]

def _camera_uv_from_local_pose(local_pose, R_align, t_align, up_axis):
    local_R = local_pose["R"]
    local_T = local_pose["T"]
    cam_center = -local_R.T @ local_T
    cam_leveled = R_align @ cam_center + t_align
    if up_axis == 'y':
        u = cam_leveled[0]
        v = -cam_leveled[2]
    elif up_axis == 'z':
        u = cam_leveled[0]
        v = -cam_leveled[1]
    else:
        u = cam_leveled[1]
        v = -cam_leveled[2]
    return float(u), float(v)

def estimate_gt_ortho_center_from_sequence(
        gt_csv_path,
        local_poses,
        meta,
        scale_local_to_m,
        u_center,
        v_center,
        R_align,
        t_align,
        up_axis,
        region,
        region_config,
        fallback_yaw_deg
    ):
    """
    Estimate where the reconstructed orthomosaic center should be by using the
    GT camera positions of the same thermal sequence that was fed to Scal3R.
    Each frame gives an independent center estimate:
        gt_camera_utm - rotated(local_camera_offset_from_ortho_center).
    """
    if not gt_csv_path or not os.path.exists(gt_csv_path):
        return None

    gt_df = pd.read_csv(gt_csv_path)
    if 'num' in gt_df.columns:
        gt_df = gt_df.sort_values(by='num')
    centers = []
    yaw_samples = []

    for _, row in gt_df.iterrows():
        if 'filename' not in row or pd.isna(row['filename']):
            continue
        stem = _csv_frame_stem(row['filename'])
        if stem not in local_poses:
            continue
        if 'lon' not in row or 'lat' not in row or pd.isna(row['lon']) or pd.isna(row['lat']):
            continue

        gt_utm_x, gt_utm_y = deg2utm(region, region_config, float(row['lon']), float(row['lat']))
        yaw_deg = float(row['yaw']) if 'yaw' in row and not pd.isna(row['yaw']) else float(fallback_yaw_deg)
        yaw_samples.append(yaw_deg)
        yaw_rad = np.deg2rad(yaw_deg)
        cos_gt, sin_gt = np.cos(yaw_rad), np.sin(yaw_rad)

        u, v = _camera_uv_from_local_pose(local_poses[stem], R_align, t_align, up_axis)
        du_m = (u - u_center) * scale_local_to_m
        dv_m = (v - v_center) * scale_local_to_m

        east_offset = du_m * cos_gt + (-dv_m) * sin_gt
        north_offset = -du_m * sin_gt + (-dv_m) * cos_gt
        centers.append((gt_utm_x - east_offset, gt_utm_y - north_offset))

    if not centers:
        return None

    centers_np = np.array(centers, dtype=np.float64)
    center_xy = np.median(centers_np, axis=0)
    residuals = np.linalg.norm(centers_np - center_xy[None, :], axis=1)
    yaw_deg = float(np.median(np.array(yaw_samples, dtype=np.float64))) if yaw_samples else float(fallback_yaw_deg)

    return {
        "utm_x": float(center_xy[0]),
        "utm_y": float(center_xy[1]),
        "yaw_deg": yaw_deg,
        "frame_count": int(len(centers)),
        "rmse_m": float(np.sqrt(np.mean(residuals ** 2))),
        "max_residual_m": float(np.max(residuals)),
    }

def _wrap_angle_deg(angle_deg):
    return float((float(angle_deg) + 180.0) % 360.0 - 180.0)

def _rotate_enu_clockwise(vec_en, yaw_deg):
    yaw = np.deg2rad(float(yaw_deg))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    east, north = float(vec_en[0]), float(vec_en[1])
    return np.array([
        east * cos_y + north * sin_y,
        -east * sin_y + north * cos_y,
    ], dtype=np.float64)

def _leveled_vector_to_bev_right_up(vec_leveled, up_axis):
    vec_leveled = np.asarray(vec_leveled, dtype=np.float64).reshape(3)
    if up_axis == "y":
        return float(vec_leveled[0]), float(vec_leveled[2])
    if up_axis == "z":
        return float(vec_leveled[0]), float(vec_leveled[1])
    if up_axis == "x":
        return float(vec_leveled[1]), float(vec_leveled[2])
    raise ValueError(f"Unsupported up_axis: {up_axis}")

def _bearing_deg_from_bev_right_up(right_m, up_m):
    if abs(float(right_m)) < 1e-12 and abs(float(up_m)) < 1e-12:
        return None
    return float(np.degrees(np.arctan2(float(right_m), float(up_m))))

def _local_camera_forward_bearing_deg(local_pose, R_align, up_axis):
    local_R = local_pose["R"]
    for dir_cam in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])):
        ray_leveled = R_align @ (local_R.T @ dir_cam)
        right, up = _leveled_vector_to_bev_right_up(ray_leveled, up_axis)
        bearing = _bearing_deg_from_bev_right_up(right, up)
        if bearing is not None:
            return bearing, dir_cam.tolist(), [float(right), float(up)]
    return None, None, None

def _estimate_local_track_bearing_deg(frame_names, local_poses, R_align, t_align, up_axis):
    pts = []
    for name in frame_names:
        if name not in local_poses:
            continue
        u, v = _camera_uv_from_local_pose(local_poses[name], R_align, t_align, up_axis)
        pts.append([u, v])
    pts = np.asarray(pts, dtype=np.float64)
    if pts.shape[0] < 2:
        return None

    deltas = np.diff(pts, axis=0)
    step_len = np.linalg.norm(deltas, axis=1)
    keep = step_len > max(float(np.nanmedian(step_len)) * 0.05, 1e-9)
    if not np.any(keep):
        return None

    deltas = deltas[keep]
    step_len = step_len[keep]
    mean_delta = np.average(deltas, axis=0, weights=step_len)
    mean_bearing = _bearing_deg_from_bev_right_up(float(mean_delta[0]), float(-mean_delta[1]))
    chord = pts[-1] - pts[0]
    chord_bearing = _bearing_deg_from_bev_right_up(float(chord[0]), float(-chord[1]))
    path_len = float(np.sum(step_len))
    chord_len = float(np.linalg.norm(chord))
    chord_path_ratio = chord_len / max(path_len, 1e-12)

    bearings = np.array([
        _bearing_deg_from_bev_right_up(float(d[0]), float(-d[1]))
        for d in deltas
    ], dtype=np.float64)
    bearings_rad = np.unwrap(np.deg2rad(bearings))
    turn_span_deg = float(np.rad2deg(np.max(bearings_rad) - np.min(bearings_rad))) if bearings_rad.size else 0.0

    return {
        "bearing_deg": float(chord_bearing if chord_bearing is not None else mean_bearing),
        "mean_bearing_deg": None if mean_bearing is None else float(mean_bearing),
        "chord_bearing_deg": None if chord_bearing is None else float(chord_bearing),
        "chord_path_ratio": float(chord_path_ratio),
        "turn_span_deg": float(abs(turn_span_deg)),
        "path_len_local": float(path_len),
        "chord_len_local": float(chord_len),
    }

def _yaw_rotation_enu_clockwise_matrix(yaw_deg):
    yaw = np.deg2rad(float(yaw_deg))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cos_y, sin_y, 0.0],
        [-sin_y, cos_y, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

def _leveled_to_enu_matrix(up_axis="y", up_direction=-1):
    up_direction = float(up_direction)
    if up_axis == "y":
        # Leveled axes used by the orthophoto: x=image right/east-like,
        # z=image up/north-like, y=height axis.
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, up_direction, 0.0],
        ], dtype=np.float64)
    if up_axis == "z":
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, up_direction],
        ], dtype=np.float64)
    if up_axis == "x":
        return np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [up_direction, 0.0, 0.0],
        ], dtype=np.float64)
    raise ValueError(f"Unsupported up_axis: {up_axis}")

def _camera_to_body_rotation_opencv():
    # Body frame used for yaw/pitch/roll: X=forward(image top), Y=right, Z=down.
    return np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

def _body_to_ned_from_body_to_enu(R_body_to_enu):
    enu_to_ned = np.array([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=np.float64)
    return enu_to_ned @ R_body_to_enu

def _pose_ypr_from_body_to_enu(R_body_to_enu):
    R_body_to_ned = _body_to_ned_from_body_to_enu(R_body_to_enu)
    yaw_deg, pitch_deg, roll_deg = R.from_matrix(R_body_to_ned).as_euler("ZYX", degrees=True)
    return {
        "Yaw_deg": _wrap_angle_deg(yaw_deg),
        "Pitch_deg": _wrap_angle_deg(pitch_deg),
        "Roll_deg": _wrap_angle_deg(roll_deg),
    }

def _pred_pose_ypr_in_gt_csv_convention(R_body_to_enu):
    ypr = _pose_ypr_from_body_to_enu(R_body_to_enu)
    raw_pitch = _wrap_angle_deg(-ypr["Pitch_deg"])
    dji_like_pitch = _wrap_angle_deg(-(90.0 - abs(float(raw_pitch))))
    return {
        "Yaw_deg": ypr["Yaw_deg"],
        # Thermal_info.csv uses camera/drone pitch with nose-down as negative.
        "Pitch_deg": dji_like_pitch,
        "Roll_deg": ypr["Roll_deg"],
    }

def _apply_yaw_delta_to_global_orientation(pose, yaw_delta_deg):
    if "Ortho_Yaw_deg" in pose:
        pose["Ortho_Yaw_deg"] = float(_wrap_angle_deg(float(pose["Ortho_Yaw_deg"]) + yaw_delta_deg))
    if "Rotation_Matrix" not in pose:
        pose["Yaw_deg"] = float(_wrap_angle_deg(float(pose["Yaw_deg"]) + yaw_delta_deg))
        return
    R_body_to_enu = np.array(pose["Rotation_Matrix"], dtype=np.float64)
    R_body_to_enu = _yaw_rotation_enu_clockwise_matrix(yaw_delta_deg) @ R_body_to_enu
    ypr = _pred_pose_ypr_in_gt_csv_convention(R_body_to_enu)
    pose["Rotation_Matrix"] = R_body_to_enu.tolist()
    pose["Yaw_deg"] = float(ypr["Yaw_deg"])
    pose["Pitch_deg"] = float(ypr["Pitch_deg"])
    pose["Roll_deg"] = float(ypr["Roll_deg"])

def _camera_basis_enu_from_pose_meta(pose_meta):
    yaw_deg = _read_float(pose_meta, ["yaw", "Yaw_deg", "gb_yaw"], 0.0)
    pitch_deg = _read_float(pose_meta, ["pitch", "Pitch_deg", "gb_pitch"], -90.0)
    roll_deg = _read_float(pose_meta, ["roll", "Roll_deg", "gb_roll"], 0.0)
    depression_deg = -float(pitch_deg)
    yaw_rad = np.deg2rad(float(yaw_deg))
    dep_rad = np.deg2rad(depression_deg)

    forward = np.array([
        np.sin(yaw_rad) * np.cos(dep_rad),
        np.cos(yaw_rad) * np.cos(dep_rad),
        -np.sin(dep_rad),
    ], dtype=np.float64)
    forward /= max(np.linalg.norm(forward), 1e-12)
    right = np.array([np.cos(yaw_rad), -np.sin(yaw_rad), 0.0], dtype=np.float64)
    right /= max(np.linalg.norm(right), 1e-12)
    down = np.cross(forward, right)
    down /= max(np.linalg.norm(down), 1e-12)

    roll_rad = np.deg2rad(float(roll_deg))
    right_roll = np.cos(roll_rad) * right + np.sin(roll_rad) * down
    down_roll = -np.sin(roll_rad) * right + np.cos(roll_rad) * down
    return right_roll, down_roll, forward

def _opencv_pnp_initial_from_pose_meta(pose_meta, camera_center_enu, object_origin_enu):
    """
    Build an OpenCV object->camera pose prior from the current ENU pose.
    Object points in this file use ENU axes, usually shifted by object_origin_enu.
    OpenCV camera axes are x=right, y=down, z=forward.
    """
    right_axis, down_axis, forward_axis = _camera_basis_enu_from_pose_meta(pose_meta)
    R_obj_to_cam = np.vstack([right_axis, down_axis, forward_axis]).astype(np.float64)
    camera_center_enu = np.asarray(camera_center_enu, dtype=np.float64).reshape(3)
    object_origin_enu = np.asarray(object_origin_enu, dtype=np.float64).reshape(3)
    camera_center_obj = camera_center_enu - object_origin_enu
    tvec = -R_obj_to_cam @ camera_center_obj.reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R_obj_to_cam)
    return rvec.astype(np.float64), tvec.astype(np.float64), R_obj_to_cam


def _globalize_scal3r_pointcloud_for_pnp(ply_path, ortho_meta_path, run_dir, global_trajectory):
    points_raw, colors_rgb = load_ply_points(ply_path)
    if points_raw is None or len(points_raw) == 0:
        raise ValueError(f"Empty point cloud: {ply_path}")
    colors_bgr = None if colors_rgb is None else colors_rgb[:, ::-1].astype(np.uint8)

    with open(ortho_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _, local_poses = load_scal3r_poses(os.path.join(run_dir, "extri.yml"))
    pose_names = sorted([k for k in global_trajectory.keys() if not str(k).startswith("_")])
    if not pose_names:
        raise ValueError("global_trajectory has no camera poses.")
    first_name = pose_names[0]
    if first_name not in local_poses:
        first_name = sorted(local_poses.keys())[0]

    R_align = np.array(meta.get("R_align", np.eye(3).tolist()), dtype=np.float64)
    t_align = np.array(meta.get("t_align", [0.0, 0.0, 0.0]), dtype=np.float64)
    scale_local_to_m = _read_float(meta, ["local_to_meter"])
    if scale_local_to_m is None or scale_local_to_m <= 0:
        raise ValueError("orthophoto metadata lacks a valid local_to_meter scale.")
    up_axis = meta.get("up_axis", "y")
    up_direction = meta.get("up_direction", -1)
    first_ground_h = _read_float(meta, ["first_frame_ground_h", "h_center"], 0.0)

    points_leveled = (R_align @ points_raw.T).T + t_align
    u_raw, v_raw, h = _orthophoto_axes(points_leveled, up_axis, up_direction)
    v_img = -v_raw
    u0, v0 = _camera_uv_from_local_pose(local_poses[first_name], R_align, t_align, up_axis)

    right_m = (u_raw - float(u0)) * float(scale_local_to_m)
    up_m = -(v_img - float(v0)) * float(scale_local_to_m)
    position_yaw = global_trajectory.get("_yaw_alignment", {}).get("position_yaw_deg")
    if position_yaw is None:
        position_yaw = global_trajectory[first_name].get("Ortho_Yaw_deg", global_trajectory[first_name].get("Yaw_deg", 0.0))
    yaw = np.deg2rad(float(position_yaw))
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    first_t = np.array(global_trajectory[first_name]["Translation_UTM"], dtype=np.float64)
    first_height = _read_float(global_trajectory[first_name], ["Height"], 0.0)
    base_elevation = float(first_t[2] - first_height) if len(first_t) >= 3 else 0.0
    east = first_t[0] + right_m * cos_y + up_m * sin_y
    north = first_t[1] - right_m * sin_y + up_m * cos_y
    z_up = base_elevation + (h - float(first_ground_h)) * float(scale_local_to_m)
    global_points = np.stack([east, north, z_up], axis=1).astype(np.float64)

    return {
        "points_enu": global_points,
        "colors_bgr": colors_bgr,
        "meta": {
            "source_ply": ply_path,
            "local_to_meter": float(scale_local_to_m),
            "position_yaw_deg": float(position_yaw),
            "first_frame": first_name,
            "base_elevation": float(base_elevation),
            "point_count": int(global_points.shape[0]),
        }
    }


def _render_global_pointcloud_view(
        points_enu,
        colors_bgr,
        frame_pose,
        true_pos,
        image_shape,
        max_points=350000,
        max_range_m=650.0,
    ):
    pose_meta = _load_pose_meta_for_frame("", None, true_pos, frame_pose.get("Yaw_deg", 0.0), frame_pose=frame_pose)
    intr = get_camera_intrinsics_from_meta(pose_meta)
    h_img = int(image_shape[0])
    w_img = int(image_shape[1])
    K = np.array([[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]], dtype=np.float64)

    cam_t = np.array(frame_pose["Translation_UTM"], dtype=np.float64)
    if cam_t.size < 3:
        cam_t = np.array([cam_t[0], cam_t[1], _read_float(frame_pose, ["Height"], 0.0)], dtype=np.float64)

    rel_xy = points_enu[:, :2] - cam_t[:2][None, :]
    near_mask = np.linalg.norm(rel_xy, axis=1) <= float(max_range_m)
    candidate_idx = np.flatnonzero(near_mask)
    if candidate_idx.size == 0:
        raise ValueError("No global point-cloud points near current frame.")
    if candidate_idx.size > int(max_points):
        step = int(math.ceil(candidate_idx.size / float(max_points)))
        candidate_idx = candidate_idx[::step]

    pts = points_enu[candidate_idx]
    rel = pts - cam_t.reshape(1, 3)
    right_axis, down_axis, forward_axis = _camera_basis_enu_from_pose_meta(pose_meta)
    x = rel @ right_axis
    y = rel @ down_axis
    z = rel @ forward_axis
    valid = z > 1.0
    if not np.any(valid):
        raise ValueError("No point-cloud points project in front of current camera.")
    x, y, z = x[valid], y[valid], z[valid]
    valid_idx = candidate_idx[valid]

    u = intr["fx"] * (x / z) + intr["cx"]
    v = intr["fy"] * (y / z) + intr["cy"]
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = (ui >= 0) & (ui < w_img) & (vi >= 0) & (vi < h_img)
    if not np.any(inside):
        raise ValueError("Projected point cloud does not overlap current image.")
    ui, vi, z, valid_idx = ui[inside], vi[inside], z[inside], valid_idx[inside]

    order = np.argsort(z)[::-1]  # far to near; near points overwrite.
    ui, vi, z, valid_idx = ui[order], vi[order], z[order], valid_idx[order]
    render = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    depth = np.full((h_img, w_img), np.inf, dtype=np.float32)
    point_idx_map = np.full((h_img, w_img), -1, dtype=np.int32)
    if colors_bgr is None:
        render[vi, ui] = (180, 180, 180)
    else:
        render[vi, ui] = colors_bgr[valid_idx]
    depth[vi, ui] = z.astype(np.float32)
    point_idx_map[vi, ui] = valid_idx.astype(np.int32)
    mask = (point_idx_map >= 0).astype(np.uint8) * 255

    render_vis, mask_vis = postprocess_projected_pointcloud_for_matching(
        render,
        mask,
        dilate_kernel_size=7,
        close_kernel_size=13,
        inpaint_radius=5,
        blur_sigma=0.75,
        blend_alpha=0.55,
    )
    lookup_idx_map, lookup_dist_map = _densify_point_idx_map_for_matching(
        point_idx_map,
        mask_vis,
        max_distance_px=10.0,
    )
    return render_vis, mask_vis, depth, point_idx_map, lookup_idx_map, lookup_dist_map, K, render, mask


def _lookup_rendered_3d_points(render_pts, point_idx_map, points_enu, radius_px=2, distance_map=None):
    object_points = []
    keep_indices = []
    lookup_distances = []
    h, w = point_idx_map.shape[:2]
    for i, pt in enumerate(render_pts):
        x = int(round(float(pt[0])))
        y = int(round(float(pt[1])))
        if 0 <= x < w and 0 <= y < h and point_idx_map[y, x] >= 0:
            pidx = int(point_idx_map[y, x])
            object_points.append(points_enu[pidx])
            keep_indices.append(i)
            lookup_distances.append(0.0 if distance_map is None else float(distance_map[y, x]))
            continue
        x1, x2 = max(0, x - radius_px), min(w, x + radius_px + 1)
        y1, y2 = max(0, y - radius_px), min(h, y + radius_px + 1)
        window = point_idx_map[y1:y2, x1:x2]
        valid = np.argwhere(window >= 0)
        if valid.size == 0:
            continue
        yyxx = valid - np.array([[y - y1, x - x1]])
        nearest = valid[int(np.argmin(np.sum(yyxx * yyxx, axis=1)))]
        pidx = int(window[nearest[0], nearest[1]])
        object_points.append(points_enu[pidx])
        keep_indices.append(i)
        if distance_map is not None:
            lookup_distances.append(float(distance_map[y1 + nearest[0], x1 + nearest[1]]))
        else:
            lookup_distances.append(float(np.sqrt(np.min(np.sum(yyxx * yyxx, axis=1)))))
    if not object_points:
        return np.empty((0, 3), dtype=np.float64), [], []
    return np.asarray(object_points, dtype=np.float64), keep_indices, lookup_distances

def _draw_pnp_match_lines(
        render_img,
        render_pts,
        uav_img,
        uav_pts,
        out_path,
        title,
        max_lines=400,
        conf=None,
    ):
    if render_pts is None or uav_pts is None:
        return
    render_pts = np.asarray(render_pts, dtype=np.float64).reshape(-1, 2)
    uav_pts = np.asarray(uav_pts, dtype=np.float64).reshape(-1, 2)
    count = min(len(render_pts), len(uav_pts))
    if count <= 0:
        return
    render_vis = render_img.copy()
    uav_vis = uav_img.copy()
    if render_vis.ndim == 2:
        render_vis = cv2.cvtColor(render_vis, cv2.COLOR_GRAY2BGR)
    if uav_vis.ndim == 2:
        uav_vis = cv2.cvtColor(uav_vis, cv2.COLOR_GRAY2BGR)

    if conf is not None:
        conf_arr = np.asarray(conf, dtype=np.float64).reshape(-1)[:count]
        order = np.argsort(-conf_arr)
    else:
        order = np.arange(count)
    if count > int(max_lines):
        order = order[:int(max_lines)]

    h = max(render_vis.shape[0], uav_vis.shape[0])
    w_left = render_vis.shape[1]
    w_right = uav_vis.shape[1]
    canvas = np.zeros((h, w_left + w_right, 3), dtype=np.uint8)
    canvas[:render_vis.shape[0], :w_left] = render_vis
    canvas[:uav_vis.shape[0], w_left:w_left + w_right] = uav_vis

    for j, idx in enumerate(order):
        rp = render_pts[idx]
        up = uav_pts[idx]
        x1, y1 = int(round(rp[0])), int(round(rp[1]))
        x2, y2 = int(round(up[0] + w_left)), int(round(up[1]))
        hue = (37 * j) % 180
        color_hsv = np.uint8([[[hue, 220, 255]]])
        color = tuple(int(v) for v in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.line(canvas, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
        cv2.circle(canvas, (x1, y1), 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (x2, y2), 3, color, -1, cv2.LINE_AA)

    cv2.putText(canvas, "rendered pointcloud", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "uav frame", (w_left + 12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"{title}: {len(order)}/{count}", (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)

def _optimize_trajectory_with_pose_anchors(
        global_trajectory,
        anchor_observations,
        save_dir,
        summary_key,
        odom_constraint_sigma_m=1.0,
        anchor_constraint_sigma_m=4.0,
        lock_first_frame_prior=True,
    ):
    corrected = json.loads(json.dumps(global_trajectory))
    pose_names = sorted([k for k in global_trajectory.keys() if not str(k).startswith("_")])
    pose_graph_summary = {
        "enabled": True,
        "mode": summary_key,
        "odom_constraint_sigma_m": float(odom_constraint_sigma_m),
        "anchor_constraint_sigma_m": float(anchor_constraint_sigma_m),
        "anchor_observation_count": int(len(anchor_observations)),
    }
    if len(pose_names) < 2:
        pose_graph_summary.update({"optimized": False, "reason": "fewer_than_two_poses"})
        corrected[summary_key] = {"pose_graph": pose_graph_summary}
        return corrected, pose_graph_summary

    original_xy = np.array([global_trajectory[name]["Translation_UTM"][:2] for name in pose_names], dtype=np.float64)
    n = len(pose_names)
    name_to_idx = {name: i for i, name in enumerate(pose_names)}
    first_xy = original_xy[0].copy()
    var_count = max(0, n - 1)
    rows, rhs_e, rhs_n = [], [], []

    def _add_equation(coeffs, target_xy, weight):
        rows.append(coeffs * float(weight))
        rhs_e.append(float(target_xy[0]) * float(weight))
        rhs_n.append(float(target_xy[1]) * float(weight))

    odom_w = 1.0 / max(float(odom_constraint_sigma_m), 1e-6)
    anchor_w_base = 1.0 / max(float(anchor_constraint_sigma_m), 1e-6)
    for i in range(1, n):
        delta_xy = original_xy[i] - original_xy[i - 1]
        coeff = np.zeros(var_count, dtype=np.float64)
        coeff[i - 1] = 1.0
        target = delta_xy.copy()
        if i == 1:
            target = first_xy + delta_xy
        else:
            coeff[i - 2] -= 1.0
        _add_equation(coeff, target, odom_w)

    used_anchor_frames = []
    for obs in anchor_observations:
        idx = name_to_idx.get(obs["frame"])
        if idx is None or idx == 0:
            continue
        coeff = np.zeros(var_count, dtype=np.float64)
        coeff[idx - 1] = 1.0
        inlier_boost = math.sqrt(max(1.0, min(float(obs.get("inlier_count", 1)), 900.0)) / 300.0)
        _add_equation(coeff, np.asarray(obs["corr_xy"], dtype=np.float64), anchor_w_base * inlier_boost)
        used_anchor_frames.append(obs["frame"])

    if not used_anchor_frames:
        pose_graph_summary.update({
            "optimized": False,
            "reason": "no_effective_anchor_observations",
            "variable_pose_count": int(var_count),
            "equation_count": int(len(rows)),
            "used_anchor_frames": [],
            "max_position_delta_m": 0.0,
        })
        corrected[summary_key] = {"pose_graph": pose_graph_summary}
        return corrected, pose_graph_summary

    if not rows or var_count <= 0:
        pose_graph_summary.update({"optimized": False, "reason": "insufficient_equations_or_variables"})
        corrected[summary_key] = {"pose_graph": pose_graph_summary}
        return corrected, pose_graph_summary

    A = np.vstack(rows)
    sol_e, *_ = np.linalg.lstsq(A, np.asarray(rhs_e, dtype=np.float64), rcond=None)
    sol_n, *_ = np.linalg.lstsq(A, np.asarray(rhs_n, dtype=np.float64), rcond=None)
    optimized_xy = original_xy.copy()
    optimized_xy[0] = first_xy
    optimized_xy[1:, 0] = sol_e
    optimized_xy[1:, 1] = sol_n

    anchor_indices = [0]
    anchor_yaws = [0.0]
    for obs in anchor_observations:
        idx = name_to_idx.get(obs["frame"])
        if idx is not None:
            anchor_indices.append(int(idx))
            anchor_yaws.append(float(obs.get("yaw_delta", 0.0)))
    order = np.argsort(anchor_indices)
    anchor_indices = np.asarray(anchor_indices, dtype=np.float64)[order]
    anchor_yaws = np.asarray(anchor_yaws, dtype=np.float64)[order]

    max_pos_delta = 0.0
    for i, name in enumerate(pose_names):
        pose = corrected[name]
        old_xy = original_xy[i]
        new_xy = optimized_xy[i]
        max_pos_delta = max(max_pos_delta, float(np.linalg.norm(new_xy - old_xy)))
        new_t = list(pose["Translation_UTM"])
        new_t[0] = float(new_xy[0])
        new_t[1] = float(new_xy[1])
        pose["Translation_UTM"] = new_t
        applied_yaw_delta = float(np.interp(float(i), anchor_indices, anchor_yaws)) if len(anchor_indices) > 1 else 0.0
        _apply_yaw_delta_to_global_orientation(pose, applied_yaw_delta)
        pose["PnP_3D_Correction"] = {
            "mode": summary_key,
            "first_frame_fixed": bool(lock_first_frame_prior),
            "applied_yaw_delta_deg": float(applied_yaw_delta),
            "position_delta_east_m": float(new_xy[0] - old_xy[0]),
            "position_delta_north_m": float(new_xy[1] - old_xy[1]),
            "used_anchor_frames": used_anchor_frames,
        }

    pose_graph_summary.update({
        "optimized": True,
        "variable_pose_count": int(var_count),
        "equation_count": int(A.shape[0]),
        "used_anchor_frames": used_anchor_frames,
        "max_position_delta_m": float(max_pos_delta),
    })
    return corrected, pose_graph_summary


def _se2_transform_global_pointcloud(global_pc, pred_xy, corr_xy, yaw_delta_deg):
    pred_xy = np.asarray(pred_xy, dtype=np.float64).reshape(2)
    corr_xy = np.asarray(corr_xy, dtype=np.float64).reshape(2)
    points = np.asarray(global_pc["points_enu"], dtype=np.float64).copy()
    rel_xy = points[:, :2] - pred_xy.reshape(1, 2)
    theta = np.deg2rad(float(yaw_delta_deg))
    c, s = np.cos(theta), np.sin(theta)
    rotated_xy = np.empty_like(rel_xy)
    rotated_xy[:, 0] = rel_xy[:, 0] * c + rel_xy[:, 1] * s
    rotated_xy[:, 1] = -rel_xy[:, 0] * s + rel_xy[:, 1] * c
    points[:, :2] = corr_xy.reshape(1, 2) + rotated_xy
    out = {
        "points_enu": points,
        "colors_bgr": global_pc.get("colors_bgr"),
        "meta": json.loads(json.dumps(global_pc.get("meta", {}))),
    }
    out["meta"]["pointcloud_nailing"] = {
        "mode": "minima_orthophoto_satellite_se2",
        "pred_anchor_xy": [float(pred_xy[0]), float(pred_xy[1])],
        "corr_anchor_xy": [float(corr_xy[0]), float(corr_xy[1])],
        "yaw_delta_deg": float(yaw_delta_deg),
    }
    return out

def _pose_with_se2_initial_for_render(frame_pose, pred_xy, corr_xy, yaw_delta_deg):
    pose = json.loads(json.dumps(frame_pose))
    old_t = list(pose["Translation_UTM"])
    old_xy = np.asarray(old_t[:2], dtype=np.float64)
    new_xy = np.asarray(corr_xy, dtype=np.float64) + _rotate_enu_clockwise(
        old_xy - np.asarray(pred_xy, dtype=np.float64),
        yaw_delta_deg,
    )
    old_t[0] = float(new_xy[0])
    old_t[1] = float(new_xy[1])
    pose["Translation_UTM"] = old_t
    _apply_yaw_delta_to_global_orientation(pose, yaw_delta_deg)
    pose["_render_initial_from_minima_se2"] = {
        "pred_anchor_xy": [float(pred_xy[0]), float(pred_xy[1])],
        "corr_anchor_xy": [float(corr_xy[0]), float(corr_xy[1])],
        "yaw_delta_deg": float(yaw_delta_deg),
    }
    return pose

def _local_patch_xy_m_from_orthophoto_px(px_xy, ortho_gsd_m):
    pts = np.asarray(px_xy, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts, dtype=np.float64)
    out[:, 0] = pts[:, 0] * float(ortho_gsd_m)
    out[:, 1] = -pts[:, 1] * float(ortho_gsd_m)
    return out

def _image_pixels_to_local_ground_xy_m(
        frame_name,
        image_pts,
        local_poses,
        local_ortho_meta,
        true_pos,
        ortho_gsd_m,
    ):
    if frame_name not in local_poses:
        return np.empty((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool), {"reason": "frame_not_in_local_poses"}

    pose = local_poses[frame_name]
    local_R = pose["R"]
    local_T = pose["T"]
    cam_center = -local_R.T @ local_T

    R_align = np.array(local_ortho_meta.get("R_align", np.eye(3).tolist()), dtype=np.float64)
    t_align = np.array(local_ortho_meta.get("t_align", [0.0, 0.0, 0.0]), dtype=np.float64)
    up_axis = local_ortho_meta.get("up_axis", "y")
    up_direction = local_ortho_meta.get("up_direction", -1)
    ground_h = _read_float(local_ortho_meta, ["first_frame_ground_h", "h_center"], 0.0)

    cam_leveled = R_align @ cam_center + t_align
    _, _, cam_h_arr = _orthophoto_axes(cam_leveled.reshape(1, 3), up_axis, up_direction)
    cam_h = float(cam_h_arr[0])

    meta = dict(true_pos or {})
    intr = get_camera_intrinsics_from_meta(meta)
    pts = np.asarray(image_pts, dtype=np.float64).reshape(-1, 2)
    out_xy = np.full((pts.shape[0], 2), np.nan, dtype=np.float64)
    valid = np.zeros((pts.shape[0],), dtype=bool)
    for i, (u, v) in enumerate(pts):
        dir_cam = np.array([
            (float(u) - float(intr["cx"])) / float(intr["fx"]),
            (float(v) - float(intr["cy"])) / float(intr["fy"]),
            1.0,
        ], dtype=np.float64)
        ray_world = local_R.T @ dir_cam
        ray_leveled = R_align @ ray_world
        _, _, ray_h_arr = _orthophoto_axes(ray_leveled.reshape(1, 3), up_axis, up_direction)
        ray_h = float(ray_h_arr[0])
        if abs(ray_h) < 1e-9:
            continue
        lam = (float(ground_h) - cam_h) / ray_h
        if not np.isfinite(lam) or lam <= 0:
            continue
        ground_leveled = cam_leveled + lam * ray_leveled
        px = _leveled_point_to_orthophoto_px(ground_leveled, local_ortho_meta)
        out_xy[i] = _local_patch_xy_m_from_orthophoto_px([px], ortho_gsd_m)[0]
        valid[i] = True
    return out_xy, valid, {
        "camera_h": float(cam_h),
        "ground_h": float(ground_h),
        "valid_count": int(valid.sum()),
        "total_count": int(pts.shape[0]),
    }

def _fit_similarity_2d(src_xy, dst_xy, weights=None):
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    if src.shape[0] < 2 or dst.shape != src.shape:
        return None, float("nan"), float("nan")
    if weights is None:
        w = np.ones(src.shape[0], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape[0] != src.shape[0]:
            w = np.ones(src.shape[0], dtype=np.float64)
        w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    if float(w.sum()) <= 1e-12:
        w = np.ones(src.shape[0], dtype=np.float64)
    w = w / float(w.sum())
    mu_src = np.sum(src * w[:, None], axis=0)
    mu_dst = np.sum(dst * w[:, None], axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    var_src = float(np.sum(w * np.sum(src_c * src_c, axis=1)))
    if var_src <= 1e-12 or not np.isfinite(var_src):
        return None, float("nan"), float("nan")
    cov = (dst_c * w[:, None]).T @ src_c
    try:
        u, singular, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None, float("nan"), float("nan")
    sign = np.eye(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        sign[-1, -1] = -1.0
    rot = u @ sign @ vt
    scale = float(np.sum(singular * np.diag(sign)) / var_src)
    if not np.isfinite(scale) or scale <= 1e-12:
        return None, float("nan"), float("nan")
    t = mu_dst - scale * (rot @ mu_src)
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rot
    matrix[:, 2] = t
    yaw_deg = float(math.degrees(math.atan2(rot[1, 0], rot[0, 0])))
    return matrix, scale, yaw_deg

def _apply_similarity_2d(matrix, xy):
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    mat = np.asarray(matrix, dtype=np.float64).reshape(2, 3)
    return pts @ mat[:, :2].T + mat[:, 2]

def _apply_affine_2d(matrix_2x3, xy):
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if pts.size == 0:
        return pts.reshape(-1, 2)
    mat = np.asarray(matrix_2x3, dtype=np.float64).reshape(2, 3)
    return pts @ mat[:, :2].T + mat[:, 2]

def _spread_ratio_2d(points_xy):
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if pts.shape[0] < 3:
        return 0.0
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    try:
        singular = np.linalg.svd(centered, compute_uv=False)
    except np.linalg.LinAlgError:
        return 0.0
    if singular.size < 2 or float(singular[0]) <= 1e-12:
        return 0.0
    return float(singular[-1] / singular[0])

def _estimate_similarity_ransac(src_xy, dst_xy, conf=None, threshold_m=6.0, min_inliers=12, iterations=1200):
    src = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_xy, dtype=np.float64).reshape(-1, 2)
    n = min(src.shape[0], dst.shape[0])
    src, dst = src[:n], dst[:n]
    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    if conf is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(conf, dtype=np.float64).reshape(-1)[:n]
        weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    src = src[finite]
    dst = dst[finite]
    weights = weights[finite]
    n = int(src.shape[0])
    if n < max(3, int(min_inliers)):
        return None, np.zeros((0,), dtype=bool), {
            "reason": "too_few_correspondences",
            "candidate_count": int(n),
        }

    order = np.argsort(-weights)
    sample_pool = order[: min(n, max(80, int(min(n, 1000))))]
    rng = np.random.default_rng(int(GLOBAL_RANDOM_SEED) + 20260618 + n)
    best_matrix = None
    best_inliers = None
    best_count = -1
    best_rmse = float("inf")
    for _ in range(max(1, int(iterations))):
        if sample_pool.size < 3:
            break
        ids = rng.choice(sample_pool, size=3, replace=False)
        if _spread_ratio_2d(src[ids]) < 0.02 or _spread_ratio_2d(dst[ids]) < 0.02:
            continue
        mat, _, _ = _fit_similarity_2d(src[ids], dst[ids], weights[ids])
        if mat is None:
            continue
        res = np.linalg.norm(_apply_similarity_2d(mat, src) - dst, axis=1)
        inliers = res <= float(threshold_m)
        count = int(np.count_nonzero(inliers))
        if count <= 0:
            continue
        rmse = float(np.sqrt(np.mean(res[inliers] ** 2)))
        if count > best_count or (count == best_count and rmse < best_rmse):
            best_matrix = mat
            best_inliers = inliers
            best_count = count
            best_rmse = rmse

    if best_matrix is None or best_inliers is None or best_count < int(min_inliers):
        return None, np.zeros((n,), dtype=bool), {
            "reason": "ransac_failed_or_low_inliers",
            "candidate_count": int(n),
            "inlier_count": int(max(best_count, 0)),
        }

    refined, scale, yaw = _fit_similarity_2d(src[best_inliers], dst[best_inliers], weights[best_inliers])
    if refined is not None:
        best_matrix = refined
    res = np.linalg.norm(_apply_similarity_2d(best_matrix, src) - dst, axis=1)
    inliers = res <= float(threshold_m)
    inlier_count = int(np.count_nonzero(inliers))
    rmse = float(np.sqrt(np.mean(res[inliers] ** 2))) if inlier_count else float("nan")
    median = float(np.median(res[inliers])) if inlier_count else float("nan")
    scale = float(math.sqrt(abs(np.linalg.det(best_matrix[:, :2]))))
    yaw = float(math.degrees(math.atan2(best_matrix[1, 0], best_matrix[0, 0])))
    return best_matrix, inliers, {
        "candidate_count": int(n),
        "inlier_count": int(inlier_count),
        "inlier_ratio": float(inlier_count / max(n, 1)),
        "rmse_m": rmse,
        "median_residual_m": median,
        "scale": scale,
        "yaw_deg": yaw,
        "threshold_m": float(threshold_m),
    }

def _filter_plain_minima_matches(img0, img1, pts0, pts1, conf, min_conf=0.0):
    valid_pts0, valid_pts1, valid_conf = [], [], []
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    for p0, p1, c in zip(pts0, pts1, conf):
        if float(c) < float(min_conf):
            continue
        x0, y0 = int(round(float(p0[0]))), int(round(float(p0[1])))
        x1, y1 = int(round(float(p1[0]))), int(round(float(p1[1])))
        if not (0 <= x0 < w0 and 0 <= y0 < h0 and 0 <= x1 < w1 and 0 <= y1 < h1):
            continue
        valid_pts0.append(p0)
        valid_pts1.append(p1)
        valid_conf.append(c)
    return valid_pts0, valid_pts1, valid_conf

def _satellite_crop_pixels_to_object_points(
        sat_pts,
        crop_origin_x,
        crop_origin_y,
        ref_res,
        origin_utm,
        ground_z,
    ):
    sat_pts = np.asarray(sat_pts, dtype=np.float64).reshape(-1, 2)
    xs = float(crop_origin_x) + sat_pts[:, 0] * float(ref_res)
    ys = float(crop_origin_y) - sat_pts[:, 1] * float(ref_res)
    zs = np.full_like(xs, float(ground_z), dtype=np.float64)
    obj_global = np.stack([xs, ys, zs], axis=1)
    return obj_global - np.asarray(origin_utm, dtype=np.float64).reshape(1, 3), obj_global

def _crop_satellite_for_oblique_frame_like_orthophoto_matching(
        ref_map_full,
        region_config,
        region,
        ref_type,
        pred_t,
        pose_meta,
        search_margin_m,
        max_crop_size_m=None,
        out_dir=None,
        prefix="satellite_crop",
    ):
    center_ground = project_image_pixel_to_ground_utm(pred_t[0], pred_t[1], pose_meta)
    footprint = estimate_image_ground_footprint_metrics_m(pred_t[0], pred_t[1], pose_meta)
    footprint_m = float(footprint["max_extent_m"]) if footprint is not None else 450.0
    uncapped_crop_size_m = footprint_m + 2.0 * float(search_margin_m)
    crop_size_m = float(uncapped_crop_size_m)
    if max_crop_size_m is not None and float(max_crop_size_m) > 0:
        crop_size_m = min(crop_size_m, float(max_crop_size_m))

    sat_crop, crop_origin_x, crop_origin_y, ref_res, projected_center_px, center_inside = _crop_ref_map_at_utm(
        ref_map_full,
        region_config,
        region,
        ref_type,
        center_ground["utm_x"],
        center_ground["utm_y"],
        crop_size_m,
    )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}.jpg"), sat_crop)
        debug = sat_crop.copy()
        cv2.drawMarker(
            debug,
            (int(projected_center_px[0]), int(projected_center_px[1])),
            (0, 255, 255),
            cv2.MARKER_CROSS,
            42,
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            "image-center ground anchor",
            (int(projected_center_px[0]) + 14, int(projected_center_px[1]) + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug,
            "image-center ground anchor",
            (int(projected_center_px[0]) + 14, int(projected_center_px[1]) + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        if footprint is not None and footprint.get("corners"):
            pts = []
            for xy in footprint["corners"]:
                pts.append([
                    int(round((float(xy[0]) - float(crop_origin_x)) / float(ref_res))),
                    int(round((float(crop_origin_y) - float(xy[1])) / float(ref_res))),
                ])
            pts = np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(debug, [pts], isClosed=True, color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
        with open(os.path.join(out_dir, f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
            json.dump({
                "crop_policy": "orthophoto_matching_style_for_oblique_frame",
                "crop_center_source": "predicted_image_center_ground_intersection",
                "crop_center_utm": [float(center_ground["utm_x"]), float(center_ground["utm_y"])],
                "camera_pred_utm": [float(pred_t[0]), float(pred_t[1]), float(pred_t[2])],
                "center_ground_offset_m": [float(center_ground["east_offset"]), float(center_ground["north_offset"])],
                "footprint_extent_m": float(footprint_m),
                "search_margin_m": float(search_margin_m),
                "uncapped_crop_size_m": float(uncapped_crop_size_m),
                "max_crop_size_m": None if max_crop_size_m is None else float(max_crop_size_m),
                "crop_size_m": float(crop_size_m),
                "crop_origin_utm": [float(crop_origin_x), float(crop_origin_y)],
                "ref_res_m": float(ref_res),
                "projected_center_px": [int(projected_center_px[0]), int(projected_center_px[1])],
                "center_inside_reference": bool(center_inside),
                "footprint": footprint,
            }, f, indent=2, ensure_ascii=False)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_debug.jpg"), debug)

    return {
        "sat_crop": sat_crop,
        "crop_origin_x": float(crop_origin_x),
        "crop_origin_y": float(crop_origin_y),
        "ref_res": float(ref_res),
        "projected_center_px": projected_center_px,
        "center_inside": bool(center_inside),
        "center_ground": center_ground,
        "footprint": footprint,
        "footprint_m": float(footprint_m),
        "crop_size_m": float(crop_size_m),
    }

def _transform_points_h(points_xy, homography):
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)

def _mask_bbox_center_xy(mask, fallback_shape):
    if mask is not None and np.any(mask > 127):
        x, y, w, h = cv2.boundingRect((mask > 127).astype(np.uint8))
        return np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    h, w = fallback_shape[:2]
    return np.array([w * 0.5, h * 0.5], dtype=np.float32)

def _estimate_homography_from_matches(sen_pts, ref_pts, reproj_threshold=5.0):
    if len(sen_pts) < 4:
        return None, None
    src = np.float32(sen_pts).reshape(-1, 1, 2)
    dst = np.float32(ref_pts).reshape(-1, 1, 2)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    homography, inlier_mask = cv2.findHomography(
        src,
        dst,
        method,
        reproj_threshold,
        maxIters=4000,
        confidence=0.999,
    )
    if inlier_mask is None:
        return homography, None
    return homography, inlier_mask.ravel().astype(bool)

def _homography_quality_stats(homography, src_pts=None, dst_pts=None, inlier_mask=None):
    stats = {
        "finite": False,
        "valid_match_count": int(len(src_pts)) if src_pts is not None else 0,
        "inlier_count": 0,
        "inlier_ratio": 0.0,
        "scale": None,
        "condition": None,
        "perspective_norm": None,
        "rmse_px": None,
        "median_px": None,
    }
    if homography is None:
        stats["reason"] = "missing_homography"
        return stats
    H = np.asarray(homography, dtype=np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)):
        stats["reason"] = "invalid_homography_matrix"
        return stats
    h22 = H[2, 2]
    if abs(float(h22)) > 1e-12:
        H = H / float(h22)
    stats["finite"] = True

    A = H[:2, :2]
    try:
        singular = np.linalg.svd(A, compute_uv=False)
        if singular.size >= 2 and np.all(np.isfinite(singular)):
            stats["scale"] = float(math.sqrt(max(float(abs(np.linalg.det(A))), 0.0)))
            stats["condition"] = float(singular[0] / max(singular[-1], 1e-12))
    except np.linalg.LinAlgError:
        stats["condition"] = float("inf")
    stats["perspective_norm"] = float(np.linalg.norm(H[2, :2]))

    if src_pts is None or dst_pts is None:
        return stats
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)
    n = min(src.shape[0], dst.shape[0])
    stats["valid_match_count"] = int(n)
    if n <= 0:
        return stats
    if inlier_mask is None:
        keep = np.ones((n,), dtype=bool)
    else:
        keep = np.asarray(inlier_mask, dtype=bool).reshape(-1)[:n]
    stats["inlier_count"] = int(np.count_nonzero(keep))
    stats["inlier_ratio"] = float(stats["inlier_count"] / max(n, 1))
    if stats["inlier_count"] <= 0:
        return stats
    pred = _transform_points_h(src[:n], H)
    residual = np.linalg.norm(pred - dst[:n], axis=1)
    residual = residual[keep]
    residual = residual[np.isfinite(residual)]
    if residual.size > 0:
        stats["rmse_px"] = float(np.sqrt(np.mean(residual ** 2)))
        stats["median_px"] = float(np.median(residual))
    return stats

def _validate_minima_se2_anchor_quality(
        anchor,
        min_inliers=12,
        min_valid_matches=12,
        min_inlier_ratio=0.08,
        homography_scale_min=0.50,
        homography_scale_max=2.00,
        max_homography_condition=80.0,
        max_homography_perspective_norm=0.01,
        max_homography_rmse_px=10.0,
        min_local_valid_ratio=0.05,
    ):
    reasons = []
    raw_count = int(anchor.get("raw_match_count", 0) or 0)
    valid_count = int(anchor.get("valid_match_count", 0) or 0)
    inlier_count = int(anchor.get("inlier_count", 0) or 0)
    inlier_ratio = float(anchor.get("inlier_ratio", 0.0) or 0.0)
    if valid_count < int(min_valid_matches):
        reasons.append(f"valid_matches:{valid_count}<{int(min_valid_matches)}")
    if inlier_count < int(min_inliers):
        reasons.append(f"inliers:{inlier_count}<{int(min_inliers)}")
    if inlier_ratio < float(min_inlier_ratio):
        reasons.append(f"inlier_ratio:{inlier_ratio:.3f}<{float(min_inlier_ratio):.3f}")

    quality = anchor.get("homography_quality") or {}
    if not bool(quality.get("finite", False)):
        reasons.append("invalid_homography")
    scale = quality.get("scale")
    if scale is None or not np.isfinite(float(scale)):
        reasons.append("invalid_homography_scale")
    elif float(scale) < float(homography_scale_min) or float(scale) > float(homography_scale_max):
        reasons.append(
            f"homography_scale:{float(scale):.3f} not in "
            f"[{float(homography_scale_min):.3f},{float(homography_scale_max):.3f}]"
        )
    condition = quality.get("condition")
    if condition is None or not np.isfinite(float(condition)):
        reasons.append("invalid_homography_condition")
    elif float(condition) > float(max_homography_condition):
        reasons.append(f"homography_condition:{float(condition):.2f}>{float(max_homography_condition):.2f}")
    perspective_norm = quality.get("perspective_norm")
    if perspective_norm is None or not np.isfinite(float(perspective_norm)):
        reasons.append("invalid_homography_perspective_norm")
    elif float(perspective_norm) > float(max_homography_perspective_norm):
        reasons.append(
            f"homography_perspective_norm:{float(perspective_norm):.6f}>"
            f"{float(max_homography_perspective_norm):.6f}"
        )
    rmse = quality.get("rmse_px", anchor.get("homography_inlier_rmse_px"))
    if rmse is not None and np.isfinite(float(rmse)) and float(rmse) > float(max_homography_rmse_px):
        reasons.append(f"homography_rmse:{float(rmse):.2f}px>{float(max_homography_rmse_px):.2f}px")

    local_patch = anchor.get("local_orthophoto_patch") or {}
    valid_ratio = local_patch.get("valid_ratio")
    if valid_ratio is not None and np.isfinite(float(valid_ratio)) and float(valid_ratio) < float(min_local_valid_ratio):
        reasons.append(f"local_valid_ratio:{float(valid_ratio):.3f}<{float(min_local_valid_ratio):.3f}")

    for key in ("corrected_utm", "pred_utm"):
        xy = np.asarray(anchor.get(key, []), dtype=np.float64)
        if xy.size < 2 or not np.all(np.isfinite(xy[:2])):
            reasons.append(f"invalid_{key}")
    yaw_delta = anchor.get("yaw_delta_deg")
    if yaw_delta is None or not np.isfinite(float(yaw_delta)):
        reasons.append("invalid_yaw_delta")

    return {
        "accepted": len(reasons) == 0,
        "policy": "match_quality_and_transform_validity_only",
        "reasons": reasons,
        "raw_match_count": raw_count,
        "valid_match_count": valid_count,
        "inlier_count": inlier_count,
        "inlier_ratio": inlier_ratio,
        "thresholds": {
            "min_valid_matches": int(min_valid_matches),
            "min_inliers": int(min_inliers),
            "min_inlier_ratio": float(min_inlier_ratio),
            "homography_scale_range": [float(homography_scale_min), float(homography_scale_max)],
            "max_homography_condition": float(max_homography_condition),
            "max_homography_perspective_norm": float(max_homography_perspective_norm),
            "max_homography_rmse_px": float(max_homography_rmse_px),
            "min_local_valid_ratio": float(min_local_valid_ratio),
        },
    }

def _save_minima_matches_vis(template_img, sat_img, sen_pts, ref_pts, out_path):
    if len(sen_pts) == 0:
        return
    kpts0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in sen_pts]
    kpts1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in ref_pts]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(sen_pts))]
    vis = cv2.drawMatches(
        template_img,
        kpts0,
        sat_img,
        kpts1,
        matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(out_path, vis)

def _crop_ref_map_at_utm(ref_map_full, region_config, region, ref_type, center_x, center_y, crop_size_m):
    ref_res = float(region_config[f"{region}_{ref_type}_REF_resolution"])
    init_x = float(region_config[f"{region}_{ref_type}_REF_initialX"])
    init_y = float(region_config[f"{region}_{ref_type}_REF_initialY"])
    h, w = ref_map_full.shape[:2]

    px_size = max(32, int(round(float(crop_size_m) / ref_res)))
    center_col_raw = int(round((float(center_x) - init_x) / ref_res))
    center_row_raw = int(round((init_y - float(center_y)) / ref_res))
    center_col = int(np.clip(center_col_raw, 0, w - 1))
    center_row = int(np.clip(center_row_raw, 0, h - 1))

    col1 = max(0, center_col - px_size // 2)
    row1 = max(0, center_row - px_size // 2)
    col2 = min(w, col1 + px_size)
    row2 = min(h, row1 + px_size)
    col1 = max(0, col2 - px_size)
    row1 = max(0, row2 - px_size)
    crop = ref_map_full[row1:row2, col1:col2].copy()
    if crop.size == 0:
        raise ValueError("Satellite crop is empty for MINIMA segment correction.")

    crop_origin_x = init_x + col1 * ref_res
    crop_origin_y = init_y - row1 * ref_res
    projected_center_px = (
        int(round((float(center_x) - crop_origin_x) / ref_res)),
        int(round((crop_origin_y - float(center_y)) / ref_res)),
    )
    inside = 0 <= center_col_raw < w and 0 <= center_row_raw < h
    return crop, crop_origin_x, crop_origin_y, ref_res, projected_center_px, inside

def _crop_orthophoto_patch_at_px(
        ortho_img,
        ortho_mask,
        ortho_meta,
        center_px,
        radius_m,
        ortho_gsd_m,
        out_dir=None,
        prefix="local_orthophoto",
    ):
    if ortho_img is None or ortho_img.size == 0:
        raise ValueError("Cannot crop local orthophoto patch from an empty image.")
    if ortho_gsd_m is None or ortho_gsd_m <= 0:
        raise ValueError(f"Invalid orthophoto GSD for local patch crop: {ortho_gsd_m}")

    h, w = ortho_img.shape[:2]
    cx = int(round(float(center_px[0])))
    cy = int(round(float(center_px[1])))
    radius_px = max(32, int(math.ceil(float(radius_m) / float(ortho_gsd_m))))

    x1 = max(0, cx - radius_px)
    y1 = max(0, cy - radius_px)
    x2 = min(w, cx + radius_px + 1)
    y2 = min(h, cy + radius_px + 1)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Local orthophoto patch is empty: center=({cx},{cy}), radius_px={radius_px}, size={w}x{h}"
        )

    patch_img = ortho_img[y1:y2, x1:x2].copy()
    if ortho_mask is None:
        patch_mask = (cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY) > 5).astype(np.uint8) * 255
    else:
        patch_mask = ortho_mask[y1:y2, x1:x2].copy()

    patch_meta = dict(ortho_meta)
    res = float(ortho_meta["resolution"])
    patch_meta["u_min"] = float(ortho_meta["u_min"]) + float(x1) * res
    patch_meta["v_min"] = float(ortho_meta["v_min"]) + float(y1) * res
    patch_meta["u_max"] = patch_meta["u_min"] + float(patch_img.shape[1]) * res
    patch_meta["v_max"] = patch_meta["v_min"] + float(patch_img.shape[0]) * res
    patch_meta["width"] = int(patch_img.shape[1])
    patch_meta["height"] = int(patch_img.shape[0])
    patch_meta["parent_orthophoto_crop_xyxy"] = [int(x1), int(y1), int(x2), int(y2)]
    patch_meta["local_patch_radius_m"] = float(radius_m)
    patch_meta["local_patch_center_px_parent"] = [float(center_px[0]), float(center_px[1])]

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        patch_path = os.path.join(out_dir, f"{prefix}.png")
        mask_path = os.path.join(out_dir, f"{prefix}_mask.png")
        meta_path = os.path.join(out_dir, f"{prefix}_meta.json")
        cv2.imwrite(patch_path, patch_img)
        cv2.imwrite(mask_path, patch_mask)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(patch_meta, f, indent=2, ensure_ascii=False)
    return patch_img, patch_mask, patch_meta, np.array([x1, y1], dtype=np.float32)

def _frame_pixel_in_orthophoto(frame_name, local_poses, meta):
    if frame_name not in local_poses:
        return None
    R_align = np.array(meta.get("R_align", np.eye(3).tolist()), dtype=np.float64)
    t_align = np.array(meta.get("t_align", [0.0, 0.0, 0.0]), dtype=np.float64)
    up_axis = meta.get("up_axis", "y")
    u, v = _camera_uv_from_local_pose(local_poses[frame_name], R_align, t_align, up_axis)
    x = (u - float(meta["u_min"])) / float(meta["resolution"])
    y = (v - float(meta["v_min"])) / float(meta["resolution"])
    return np.array([x, y], dtype=np.float32)

def _leveled_point_to_orthophoto_px(point_leveled, meta):
    up_axis = meta.get("up_axis", "y")
    up_direction = meta.get("up_direction", -1)
    u, v, _ = _orthophoto_axes(np.asarray(point_leveled, dtype=np.float64).reshape(1, 3), up_axis, up_direction)
    u = float(u[0])
    v = -float(v[0])
    x = (u - float(meta["u_min"])) / float(meta["resolution"])
    y = (v - float(meta["v_min"])) / float(meta["resolution"])
    return np.array([x, y], dtype=np.float32)

def _frame_image_center_ground_pixel_in_orthophoto(frame_name, local_poses, meta):
    """
    Intersect the tail frame image-center ray with the orthophoto ground plane
    in the same leveled coordinate system that generated the orthomosaic.
    """
    if frame_name not in local_poses:
        return None, None, None

    pose = local_poses[frame_name]
    local_R = pose["R"]
    local_T = pose["T"]
    cam_center = -local_R.T @ local_T

    R_align = np.array(meta.get("R_align", np.eye(3).tolist()), dtype=np.float64)
    t_align = np.array(meta.get("t_align", [0.0, 0.0, 0.0]), dtype=np.float64)
    up_axis = meta.get("up_axis", "y")
    up_direction = meta.get("up_direction", -1)

    cam_leveled = R_align @ cam_center + t_align
    _, _, cam_h_arr = _orthophoto_axes(cam_leveled.reshape(1, 3), up_axis, up_direction)
    cam_h = float(cam_h_arr[0])
    ground_h = _read_float(meta, ["first_frame_ground_h", "h_center"], 0.0)

    best = None
    for dir_cam in (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])):
        ray_world = local_R.T @ dir_cam
        ray_leveled = R_align @ ray_world
        _, _, ray_h_arr = _orthophoto_axes(ray_leveled.reshape(1, 3), up_axis, up_direction)
        ray_h = float(ray_h_arr[0])
        if abs(ray_h) < 1e-8:
            continue
        lam = (float(ground_h) - cam_h) / ray_h
        if lam <= 0:
            continue
        ground_leveled = cam_leveled + lam * ray_leveled
        best = (ground_leveled, dir_cam.tolist(), float(lam))
        break

    if best is None:
        return None, None, {
            "reason": "image_center_ray_no_forward_ground_intersection",
            "camera_h": float(cam_h),
            "ground_h": float(ground_h),
        }

    ground_leveled, dir_cam_used, lam = best
    px = _leveled_point_to_orthophoto_px(ground_leveled, meta)
    return px, ground_leveled, {
        "camera_h": float(cam_h),
        "ground_h": float(ground_h),
        "ray_lambda": float(lam),
        "dir_cam": dir_cam_used,
    }

def _load_pose_meta_for_frame(frame_name, csv_path, true_pos, yaw_deg, frame_pose=None):
    # Only camera intrinsics/static camera parameters come from the first-frame prior.
    # Per-frame attitude and height must come from the recovered prediction, not GT CSV.
    pose_meta = {}
    for key in [
        "width", "height", "focal_len", "cam_size",
        "sensor_width", "sensor_height", "fx", "fy", "cx", "cy",
    ]:
        if true_pos is not None and key in true_pos:
            pose_meta[key] = true_pos[key]

    frame_pose = frame_pose or {}
    pose_meta["yaw"] = _read_float(frame_pose, ["Yaw_deg", "yaw", "Yaw"], yaw_deg)
    pose_meta["pitch"] = _read_float(frame_pose, ["Pitch_deg", "pitch", "Pitch"])
    pose_meta["roll"] = _read_float(frame_pose, ["Roll_deg", "roll", "Roll"])
    pose_meta["rel_height"] = _read_float(frame_pose, ["Height", "Rel_Height", "rel_height"])

    missing = [k for k in ["pitch", "roll", "rel_height"] if pose_meta.get(k) is None or not np.isfinite(float(pose_meta[k]))]
    if missing:
        raise ValueError(
            f"Frame {frame_name} lacks predicted pose metadata for MINIMA anchor: {missing}. "
            "Run recover_global_poses with predicted Pitch_deg/Roll_deg/Height enabled."
        )
    return pose_meta

def _filter_minima_matches(template_mask, sat_img, sen_pts, ref_pts, mconf, min_conf=0.0):
    valid_sen_pts, valid_ref_pts, valid_conf = [], [], []
    h0, w0 = template_mask.shape[:2]
    h1, w1 = sat_img.shape[:2]
    for pt0, pt1, conf in zip(sen_pts, ref_pts, mconf):
        if float(conf) < float(min_conf):
            continue
        x0, y0 = int(round(pt0[0])), int(round(pt0[1]))
        x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
        if not (0 <= x0 < w0 and 0 <= y0 < h0 and 0 <= x1 < w1 and 0 <= y1 < h1):
            continue
        if template_mask[y0, x0] <= 127:
            continue
        valid_sen_pts.append(pt0)
        valid_ref_pts.append(pt1)
        valid_conf.append(conf)
    return valid_sen_pts, valid_ref_pts, valid_conf

def _draw_minima_anchor_overlay(sat_crop, template_img, template_mask, homography,
                                anchor_px_sat, projected_center_px, out_path, overlay_alpha=0.55):
    h_sat, w_sat = sat_crop.shape[:2]
    warped = cv2.warpPerspective(template_img, homography, (w_sat, h_sat))
    warped_mask = cv2.warpPerspective(template_mask, homography, (w_sat, h_sat), flags=cv2.INTER_NEAREST)
    vis = sat_crop.copy()
    mask = (warped_mask > 127).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=1.5)
    alpha = (mask * float(overlay_alpha))[..., None]
    vis = np.clip(vis.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    if np.any(warped_mask > 127):
        x, y, w, h = cv2.boundingRect((warped_mask > 127).astype(np.uint8))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 4)
    cv2.circle(vis, (int(round(anchor_px_sat[0])), int(round(anchor_px_sat[1]))), 8, (0, 255, 0), -1)
    cv2.drawMarker(
        vis,
        (int(projected_center_px[0]), int(projected_center_px[1])),
        (255, 255, 0),
        cv2.MARKER_CROSS,
        40,
        4,
        cv2.LINE_AA,
    )
    cv2.imwrite(out_path, vis)

def _save_template_center_vis(template_img, template_mask, pointcloud_center_px, anchor_px, out_path, camera_anchor_px=None):
    vis = template_img.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)

    if template_mask is not None and np.any(template_mask > 127):
        x, y, w, h = cv2.boundingRect((template_mask > 127).astype(np.uint8))
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 0), 3, cv2.LINE_AA)

    pc = (int(round(pointcloud_center_px[0])), int(round(pointcloud_center_px[1])))
    cv2.drawMarker(vis, pc, (255, 255, 0), cv2.MARKER_STAR, 48, 5, cv2.LINE_AA)
    cv2.putText(vis, "PC mask bbox center", (pc[0] + 14, pc[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(vis, "PC mask bbox center", (pc[0] + 14, pc[1] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2, cv2.LINE_AA)

    if camera_anchor_px is not None:
        cp = (int(round(camera_anchor_px[0])), int(round(camera_anchor_px[1])))
        cv2.drawMarker(vis, cp, (180, 180, 180), cv2.MARKER_CROSS, 34, 4, cv2.LINE_AA)
        cv2.putText(vis, "camera center", (cp[0] + 14, cp[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(vis, "camera center", (cp[0] + 14, cp[1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 2, cv2.LINE_AA)

    if anchor_px is not None:
        ap = (int(round(anchor_px[0])), int(round(anchor_px[1])))
        cv2.drawMarker(vis, ap, (255, 0, 255), cv2.MARKER_TILTED_CROSS, 42, 5, cv2.LINE_AA)
        if camera_anchor_px is not None:
            cp = (int(round(camera_anchor_px[0])), int(round(camera_anchor_px[1])))
            cv2.line(vis, cp, ap, (255, 0, 255), 3, cv2.LINE_AA)
        cv2.putText(vis, "tail image-center ground", (ap[0] + 14, ap[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(vis, "tail image-center ground", (ap[0] + 14, ap[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2, cv2.LINE_AA)

    cv2.imwrite(out_path, vis)

def _utm_to_crop_px(utm_xy, crop_origin_x, crop_origin_y, ref_res):
    return (
        int(round((float(utm_xy[0]) - float(crop_origin_x)) / float(ref_res))),
        int(round((float(crop_origin_y) - float(utm_xy[1])) / float(ref_res))),
    )

def _draw_labeled_marker(img, pt, color, label, marker_type=cv2.MARKER_CROSS, marker_size=34):
    x, y = int(round(pt[0])), int(round(pt[1]))
    cv2.drawMarker(img, (x, y), color, markerType=marker_type, markerSize=marker_size, thickness=4, line_type=cv2.LINE_AA)
    if label:
        label_x = min(max(x + 12, 8), max(8, img.shape[1] - 520))
        label_y = min(max(y - 12, 28), max(28, img.shape[0] - 12))
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)

def _overlay_minima_pointcloud_on_crop(ref_crop, anchor, new_ref_initialX, new_ref_initialY, ref_res, alpha=0.38):
    if not anchor or not anchor.get("ok"):
        return False
    output_dir = anchor.get("output_dir")
    homography = anchor.get("homography_template_to_satellite")
    crop_origin = anchor.get("crop_origin_utm")
    if not output_dir or homography is None or crop_origin is None:
        return False

    template_path = os.path.join(output_dir, "template_scaled_north.png")
    mask_path = os.path.join(output_dir, "template_mask_scaled_north.png")
    template = cv2.imread(template_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if template is None:
        return False
    if mask is None:
        mask = (cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) > 5).astype(np.uint8) * 255

    anchor_ref_res = float(anchor.get("ref_res_m", ref_res))
    sx = anchor_ref_res / float(ref_res)
    sy = anchor_ref_res / float(ref_res)
    tx = (float(crop_origin[0]) - float(new_ref_initialX)) / float(ref_res)
    ty = (float(new_ref_initialY) - float(crop_origin[1])) / float(ref_res)
    to_compare_crop = np.array(
        [[sx, 0.0, tx],
         [0.0, sy, ty],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H = to_compare_crop @ np.asarray(homography, dtype=np.float64)
    out_size = (ref_crop.shape[1], ref_crop.shape[0])
    warped = cv2.warpPerspective(template, H, out_size)
    warped_mask = cv2.warpPerspective(mask, H, out_size, flags=cv2.INTER_NEAREST)
    valid = (warped_mask > 127).astype(np.float32)
    if not np.any(valid > 0.5):
        return False

    valid = cv2.GaussianBlur(valid, (0, 0), sigmaX=1.2)
    a = (valid * float(alpha))[..., None]
    ref_crop[:] = np.clip(ref_crop.astype(np.float32) * (1.0 - a) + warped.astype(np.float32) * a, 0, 255).astype(np.uint8)

    if np.any(warped_mask > 127):
        x, y, w, h = cv2.boundingRect((warped_mask > 127).astype(np.uint8))
        cv2.rectangle(ref_crop, (x, y), (x + w, y + h), (255, 128, 0), 2, cv2.LINE_AA)
        cv2.putText(ref_crop, "Warped PC image", (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(ref_crop, "Warped PC image", (x, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2, cv2.LINE_AA)
    return True

def _captioned_panel(img, title, size=(520, 360)):
    if img is None:
        img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out_w, out_h = size
    body_h = out_h - 34
    h, w = img.shape[:2]
    scale = min(out_w / max(w, 1), body_h / max(h, 1))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    x = (out_w - new_w) // 2
    y = 34 + (body_h - new_h) // 2
    canvas[y:y + new_h, x:x + new_w] = resized
    cv2.putText(canvas, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas

def _save_anchor_match_process_panel(frame_dir, frame_name, raw_count, valid_count, inlier_count):
    panels = []
    for filename, title in [
        ("template_center_marked.png", "Template center"),
        ("satellite_crop.jpg", "Satellite crop"),
        ("minima_filtered_matches.jpg", "Filtered matches"),
        ("minima_inlier_matches.jpg", "RANSAC inliers"),
        ("match_result_overlay.jpg", "Warp overlay"),
    ]:
        panels.append(_captioned_panel(cv2.imread(os.path.join(frame_dir, filename)), title))

    blank = np.zeros_like(panels[0])
    cv2.putText(blank, frame_name, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blank, f"raw: {raw_count}", (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(blank, f"valid: {valid_count}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(blank, f"inliers: {inlier_count}", (20, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 220, 255), 2, cv2.LINE_AA)
    panels.append(blank)

    row1 = np.hstack(panels[:3])
    row2 = np.hstack(panels[3:6])
    cv2.imwrite(os.path.join(frame_dir, "match_process_panel.jpg"), np.vstack([row1, row2]))

def save_segment_minima_match_overview(save_dir, anchors, max_cols=2):
    ok_anchors = [a for a in anchors if a.get("ok") and a.get("output_dir")]
    if not ok_anchors:
        return None
    thumbs = []
    for anchor in ok_anchors:
        panel_path = os.path.join(anchor["output_dir"], "match_process_panel.jpg")
        img = cv2.imread(panel_path)
        if img is None:
            img = cv2.imread(os.path.join(anchor["output_dir"], "match_result_overlay.jpg"))
        title = (
            f"{anchor['frame']}  "
            f"inliers={anchor.get('inlier_count', 0)}  "
            f"dYaw={anchor.get('yaw_delta_deg', 0.0):.2f}"
        )
        thumbs.append(_captioned_panel(img, title, size=(760, 520)))

    cols = max(1, int(max_cols))
    rows = int(np.ceil(len(thumbs) / cols))
    blank = np.zeros_like(thumbs[0])
    while len(thumbs) < rows * cols:
        thumbs.append(blank.copy())
    grid_rows = [np.hstack(thumbs[r * cols:(r + 1) * cols]) for r in range(rows)]
    out = np.vstack(grid_rows)
    out_path = os.path.join(save_dir, "segment_minima_match_overview.jpg")
    cv2.imwrite(out_path, out)
    return out_path

def _homography_yaw_offset_deg(homography, anchor_template_px, step_px=80.0):
    center = np.asarray(anchor_template_px, dtype=np.float32)
    north = center + np.array([0.0, -float(step_px)], dtype=np.float32)
    warped_center, warped_north = _transform_points_h([center, north], homography)
    vec = warped_north - warped_center
    if np.linalg.norm(vec) < 1e-6:
        return 0.0
    return float(math.degrees(math.atan2(vec[0], -vec[1])))

def match_segment_anchor_with_minima(
        frame_name,
        frame_pose,
        ortho_path,
        ortho_meta,
        local_poses,
        matcher,
        save_dir,
        region,
        opt,
        region_config,
        ref_map_full,
        true_pos,
        csv_path,
        search_margin_m=500.0,
        min_conf=0.0,
        min_inliers=12,
        ransac_reproj_threshold=5.0,
        overlay_alpha=0.55,
        anchor_patch_radius_m=180.0,
    ):
    from scalloc.matchers.minima import MINIMA_match

    pred_utm = np.array(frame_pose["Translation_UTM"][:2], dtype=np.float64)
    pred_yaw = float(frame_pose["Yaw_deg"])
    # The matching template is a crop from the Scal3R orthomosaic, not the raw
    # oblique UAV image.  Its image axes are governed by the global orthomosaic
    # heading; using per-frame camera yaw here breaks on turns.
    ortho_prior_yaw = _read_float(frame_pose, ["Ortho_Yaw_deg", "Orthophoto_Yaw_deg", "Map_Yaw_deg"], pred_yaw)
    ortho_prior_yaw = float(ortho_prior_yaw)

    ortho_img = cv2.imread(ortho_path)
    if ortho_img is None:
        raise ValueError(f"Cannot read orthomosaic for MINIMA correction: {ortho_path}")
    ortho_mask_path = os.path.splitext(ortho_path)[0] + "_mask.png"
    ortho_mask = cv2.imread(ortho_mask_path, cv2.IMREAD_GRAYSCALE)
    if ortho_mask is None:
        ortho_mask = (cv2.cvtColor(ortho_img, cv2.COLOR_BGR2GRAY) > 5).astype(np.uint8) * 255

    sat_res = region_config.get(f"{region}_{opt.Ref_type}_REF_resolution", 0.3)
    fallback_gsd = (true_pos["rel_height"] * true_pos["cam_size"]) / (true_pos["focal_len"] * true_pos["width"])
    ortho_gsd, _ = read_orthophoto_gsd_m(ortho_path, fallback_gsd)
    if ortho_gsd is None:
        raise ValueError("Cannot infer orthophoto GSD for MINIMA segment correction.")

    camera_px_ortho_global = _frame_pixel_in_orthophoto(frame_name, local_poses, ortho_meta)
    if camera_px_ortho_global is None:
        raise ValueError(f"Frame {frame_name} is not present in orthophoto local poses.")
    anchor_px_ortho_global, anchor_ground_leveled, anchor_ray_meta = _frame_image_center_ground_pixel_in_orthophoto(
        frame_name,
        local_poses,
        ortho_meta,
    )
    anchor_source = "local_image_center_ray_ground_intersection"
    if anchor_px_ortho_global is None:
        anchor_px_ortho_global = camera_px_ortho_global.copy()
        anchor_source = "fallback_camera_center"

    anchor_delta_px_global = anchor_px_ortho_global - camera_px_ortho_global
    anchor_delta_m_ortho = np.array([
        float(anchor_delta_px_global[0]) * float(ortho_gsd),
        -float(anchor_delta_px_global[1]) * float(ortho_gsd),
    ], dtype=np.float64)
    image_center_ground_offset_en = _rotate_enu_clockwise(anchor_delta_m_ortho, ortho_prior_yaw)
    crop_center_x = float(pred_utm[0] + image_center_ground_offset_en[0])
    crop_center_y = float(pred_utm[1] + image_center_ground_offset_en[1])
    crop_center = {
        "utm_x": crop_center_x,
        "utm_y": crop_center_y,
        "east_offset": float(image_center_ground_offset_en[0]),
        "north_offset": float(image_center_ground_offset_en[1]),
        "source": "orthophoto_camera_to_image_center_ground_delta",
        "ortho_delta_px": [float(anchor_delta_px_global[0]), float(anchor_delta_px_global[1])],
        "ortho_delta_m_before_yaw": [float(anchor_delta_m_ortho[0]), float(anchor_delta_m_ortho[1])],
        "ortho_yaw_deg": float(ortho_prior_yaw),
    }

    frame_dir = os.path.join(save_dir, "segment_minima_se2", frame_name)
    os.makedirs(frame_dir, exist_ok=True)
    local_patch_center_px = anchor_px_ortho_global
    ortho_img, ortho_mask, local_ortho_meta, patch_origin_px = _crop_orthophoto_patch_at_px(
        ortho_img=ortho_img,
        ortho_mask=ortho_mask,
        ortho_meta=ortho_meta,
        center_px=local_patch_center_px,
        radius_m=anchor_patch_radius_m,
        ortho_gsd_m=ortho_gsd,
        out_dir=frame_dir,
        prefix="local_orthophoto",
    )
    ortho_meta = local_ortho_meta
    camera_px_ortho = camera_px_ortho_global - patch_origin_px
    anchor_px_ortho = anchor_px_ortho_global - patch_origin_px
    valid_ratio = float((ortho_mask > 127).mean()) if ortho_mask is not None and ortho_mask.size > 0 else 0.0
    print(
        f"[MINIMA-SE2] {frame_name}: local orthophoto patch "
        f"{ortho_img.shape[1]}x{ortho_img.shape[0]}, radius={anchor_patch_radius_m:.1f}m, "
        f"valid_ratio={valid_ratio:.3f}"
    )

    template_scaled, mask_scaled, gsd_scale = resize_image_and_mask_to_gsd(
        ortho_img,
        ortho_mask,
        source_gsd_m=ortho_gsd,
        target_gsd_m=sat_res,
    )
    template_north, rot_mat = dumpRotateImage(template_scaled, -ortho_prior_yaw)
    mask_north, _ = dumpRotateImage(mask_scaled, -ortho_prior_yaw)

    camera_px_scaled = camera_px_ortho * float(gsd_scale)
    anchor_px_scaled = anchor_px_ortho * float(gsd_scale)
    camera_anchor_px_template = np.array([
        rot_mat[0, 0] * camera_px_scaled[0] + rot_mat[0, 1] * camera_px_scaled[1] + rot_mat[0, 2],
        rot_mat[1, 0] * camera_px_scaled[0] + rot_mat[1, 1] * camera_px_scaled[1] + rot_mat[1, 2],
    ], dtype=np.float32)
    anchor_px_template = np.array([
        rot_mat[0, 0] * anchor_px_scaled[0] + rot_mat[0, 1] * anchor_px_scaled[1] + rot_mat[0, 2],
        rot_mat[1, 0] * anchor_px_scaled[0] + rot_mat[1, 1] * anchor_px_scaled[1] + rot_mat[1, 2],
    ], dtype=np.float32)
    image_center_ground_offset_ortho_px = (anchor_px_ortho - camera_px_ortho) * float(gsd_scale)
    image_center_ground_offset_template_px = anchor_px_template - camera_anchor_px_template

    footprint_m = max(template_north.shape[:2]) * float(sat_res)
    crop_size_m = footprint_m + 2.0 * float(search_margin_m)
    sat_crop, crop_origin_x, crop_origin_y, ref_res, projected_center_px, center_inside = _crop_ref_map_at_utm(
        ref_map_full,
        region_config,
        region,
        opt.Ref_type,
        crop_center_x,
        crop_center_y,
        crop_size_m,
    )

    cv2.imwrite(os.path.join(frame_dir, "satellite_crop.jpg"), sat_crop)
    cv2.imwrite(os.path.join(frame_dir, "template_scaled_north.png"), template_north)
    cv2.imwrite(os.path.join(frame_dir, "template_mask_scaled_north.png"), mask_north)

    sen_pts, ref_pts, mconf = MINIMA_match(
        template_north,
        sat_crop,
        matcher,
        frame_dir,
        "minima_raw_matches.jpg",
        need_ransac=False,
        show_matches=False,
    )
    raw_match_count = len(sen_pts)
    sen_pts, ref_pts, mconf = _filter_minima_matches(mask_north, sat_crop, sen_pts, ref_pts, mconf, min_conf=min_conf)
    _save_minima_matches_vis(template_north, sat_crop, sen_pts, ref_pts, os.path.join(frame_dir, "minima_filtered_matches.jpg"))

    homography, inlier_mask = _estimate_homography_from_matches(sen_pts, ref_pts, reproj_threshold=ransac_reproj_threshold)
    inlier_count = 0 if inlier_mask is None else int(inlier_mask.sum())
    if homography is None or inlier_mask is None or inlier_count < int(min_inliers):
        return {
            "ok": False,
            "frame": frame_name,
            "reason": f"homography_failed_or_low_inliers:{inlier_count}",
            "raw_match_count": int(raw_match_count),
            "valid_match_count": int(len(sen_pts)),
            "inlier_count": int(inlier_count),
        }

    inlier_sen = [p for p, keep in zip(sen_pts, inlier_mask) if keep]
    inlier_ref = [p for p, keep in zip(ref_pts, inlier_mask) if keep]
    homography_quality = _homography_quality_stats(homography, sen_pts, ref_pts, inlier_mask)
    if not homography_quality.get("finite", False):
        return {
            "ok": False,
            "frame": frame_name,
            "reason": f"invalid_homography:{homography_quality.get('reason', 'unknown')}",
            "raw_match_count": int(raw_match_count),
            "valid_match_count": int(len(sen_pts)),
            "inlier_count": int(inlier_count),
            "inlier_ratio": float(homography_quality.get("inlier_ratio", 0.0)),
            "homography_quality": homography_quality,
        }
    _save_minima_matches_vis(template_north, sat_crop, inlier_sen, inlier_ref, os.path.join(frame_dir, "minima_inlier_matches.jpg"))

    pointcloud_center_px_template = _mask_bbox_center_xy(mask_north, template_north.shape)
    _save_template_center_vis(
        template_north,
        mask_north,
        pointcloud_center_px_template,
        anchor_px_template,
        os.path.join(frame_dir, "template_center_marked.png"),
        camera_anchor_px=camera_anchor_px_template,
    )
    pointcloud_center_px_sat = _transform_points_h([pointcloud_center_px_template], homography)[0]
    pointcloud_center_utm_x = crop_origin_x + float(pointcloud_center_px_sat[0]) * ref_res
    pointcloud_center_utm_y = crop_origin_y - float(pointcloud_center_px_sat[1]) * ref_res
    anchor_px_sat = _transform_points_h([anchor_px_template], homography)[0]
    camera_px_sat = _transform_points_h([camera_anchor_px_template], homography)[0]
    direct_anchor_utm_x = crop_origin_x + float(anchor_px_sat[0]) * ref_res
    direct_anchor_utm_y = crop_origin_y - float(anchor_px_sat[1]) * ref_res
    direct_camera_utm_x = crop_origin_x + float(camera_px_sat[0]) * ref_res
    direct_camera_utm_y = crop_origin_y - float(camera_px_sat[1]) * ref_res
    yaw_delta = _homography_yaw_offset_deg(homography, anchor_px_template)
    anchor_offset_px = anchor_px_template - pointcloud_center_px_template
    anchor_offset_en = np.array([
        float(anchor_offset_px[0]) * float(ref_res),
        -float(anchor_offset_px[1]) * float(ref_res),
    ], dtype=np.float64)
    anchor_offset_en = _rotate_enu_clockwise(anchor_offset_en, yaw_delta)
    bbox_center_based_ground_utm = np.array([
        pointcloud_center_utm_x + anchor_offset_en[0],
        pointcloud_center_utm_y + anchor_offset_en[1],
    ], dtype=np.float64)
    # The template already contains both the oblique image-center ground point
    # and the camera center in the same local orthophoto coordinate frame.
    # Project both through the same homography so the camera back-projection
    # direction stays consistent with the matched template visualization.
    corrected_ground_utm_x = direct_anchor_utm_x
    corrected_ground_utm_y = direct_anchor_utm_y
    corrected_utm_x = direct_camera_utm_x
    corrected_utm_y = direct_camera_utm_y
    corrected_yaw = _wrap_angle_deg(pred_yaw + yaw_delta)
    corrected_ortho_yaw = _wrap_angle_deg(ortho_prior_yaw + yaw_delta)

    _draw_minima_anchor_overlay(
        sat_crop,
        template_north,
        mask_north,
        homography,
        anchor_px_sat,
        projected_center_px,
        os.path.join(frame_dir, "match_result_overlay.jpg"),
        overlay_alpha=overlay_alpha,
    )
    _save_anchor_match_process_panel(
        frame_dir,
        frame_name,
        raw_match_count,
        len(sen_pts),
        inlier_count,
    )

    result = {
        "ok": True,
        "frame": frame_name,
        "pred_utm": [float(pred_utm[0]), float(pred_utm[1])],
        "corrected_utm": [float(corrected_utm_x), float(corrected_utm_y)],
        "corrected_camera_utm": [float(corrected_utm_x), float(corrected_utm_y)],
        "corrected_image_center_ground_utm": [float(corrected_ground_utm_x), float(corrected_ground_utm_y)],
        "direct_homography_camera_utm": [float(direct_camera_utm_x), float(direct_camera_utm_y)],
        "matched_pointcloud_center_utm": [float(pointcloud_center_utm_x), float(pointcloud_center_utm_y)],
        "direct_homography_anchor_utm": [float(direct_anchor_utm_x), float(direct_anchor_utm_y)],
        "bbox_center_based_ground_utm": [float(bbox_center_based_ground_utm[0]), float(bbox_center_based_ground_utm[1])],
        "bbox_center_vs_direct_anchor_delta_m": [
            float(bbox_center_based_ground_utm[0] - direct_anchor_utm_x),
            float(bbox_center_based_ground_utm[1] - direct_anchor_utm_y),
        ],
        "camera_position_source": "direct_homography_camera_center_template_pixel",
        "anchor_offset_from_pointcloud_center_m": [float(anchor_offset_en[0]), float(anchor_offset_en[1])],
        "pred_yaw_deg": float(pred_yaw),
        "corrected_yaw_deg": float(corrected_yaw),
        "ortho_prior_yaw_deg": float(ortho_prior_yaw),
        "corrected_ortho_yaw_deg": float(corrected_ortho_yaw),
        "yaw_delta_deg": float(yaw_delta),
        "yaw_delta_definition": "orthomosaic_heading_delta_from_MINIMA_homography",
        "crop_center_utm": [float(crop_center_x), float(crop_center_y)],
        "crop_center_meta": crop_center,
        "crop_origin_utm": [float(crop_origin_x), float(crop_origin_y)],
        "ref_res_m": float(ref_res),
        "projected_center_px": [int(projected_center_px[0]), int(projected_center_px[1])],
        "center_inside_map": bool(center_inside),
        "anchor_template_px": [float(anchor_px_template[0]), float(anchor_px_template[1])],
        "camera_center_template_px": [float(camera_anchor_px_template[0]), float(camera_anchor_px_template[1])],
        "anchor_definition": "tail_image_center_ground_projection",
        "anchor_source": anchor_source,
        "anchor_ray_meta": anchor_ray_meta,
        "image_center_ground_offset_m": [float(image_center_ground_offset_en[0]), float(image_center_ground_offset_en[1])],
        "image_center_ground_offset_ortho_px": [float(image_center_ground_offset_ortho_px[0]), float(image_center_ground_offset_ortho_px[1])],
        "image_center_ground_offset_template_px": [float(image_center_ground_offset_template_px[0]), float(image_center_ground_offset_template_px[1])],
        "anchor_sat_px": [float(anchor_px_sat[0]), float(anchor_px_sat[1])],
        "camera_sat_px": [float(camera_px_sat[0]), float(camera_px_sat[1])],
        "pointcloud_center_template_px": [float(pointcloud_center_px_template[0]), float(pointcloud_center_px_template[1])],
        "pointcloud_center_sat_px": [float(pointcloud_center_px_sat[0]), float(pointcloud_center_px_sat[1])],
        "local_orthophoto_patch": {
            "enabled": True,
            "radius_m": float(anchor_patch_radius_m),
            "valid_ratio": float(valid_ratio),
            "parent_crop_xyxy": ortho_meta.get("parent_orthophoto_crop_xyxy"),
            "center_px_parent": ortho_meta.get("local_patch_center_px_parent"),
            "image_path": os.path.join(frame_dir, "local_orthophoto.png"),
            "mask_path": os.path.join(frame_dir, "local_orthophoto_mask.png"),
            "meta_path": os.path.join(frame_dir, "local_orthophoto_meta.json"),
        },
        "raw_match_count": int(raw_match_count),
        "valid_match_count": int(len(sen_pts)),
        "inlier_count": int(inlier_count),
        "inlier_ratio": float(homography_quality.get("inlier_ratio", 0.0)),
        "homography_inlier_rmse_px": homography_quality.get("rmse_px"),
        "homography_inlier_median_px": homography_quality.get("median_px"),
        "homography_scale": homography_quality.get("scale"),
        "homography_condition": homography_quality.get("condition"),
        "homography_perspective_norm": homography_quality.get("perspective_norm"),
        "homography_quality": homography_quality,
        "homography_template_to_satellite": homography.tolist(),
        "output_dir": frame_dir,
    }
    with open(os.path.join(frame_dir, "anchor_match_result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result

def apply_segment_minima_se2_correction(
        global_trajectory,
        chunk_records,
        run_dir,
        save_dir,
        ortho_path,
        true_pos,
        csv_path,
        region,
        opt,
        region_config,
        ref_map_full,
        search_margin_m=500.0,
        min_conf=0.0,
        min_inliers=12,
        ransac_reproj_threshold=5.0,
        overlay_alpha=0.55,
        matcher_method="MINIMA_Roma",
        lock_first_frame_prior=True,
        anchor_patch_radius_m=180.0,
        anchor_match_quality_gate_enabled=True,
        anchor_min_inlier_ratio=0.08,
        anchor_homography_scale_min=0.50,
        anchor_homography_scale_max=2.00,
        anchor_max_homography_condition=80.0,
        anchor_max_homography_perspective_norm=0.01,
        anchor_max_homography_rmse_px=10.0,
        anchor_min_local_valid_ratio=0.05,
        max_anchor_innovation_m=60.0,
        max_anchor_yaw_jump_deg=15.0,
        max_first_anchor_delta_m=120.0,
        anchor_geometry_gate_enabled=False,
        odom_constraint_sigma_m=1.0,
        anchor_constraint_sigma_m=6.0,
    ):
    from scalloc.matchers.minima import MINIMA_Init

    meta_path = os.path.splitext(ortho_path)[0] + "_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        ortho_meta = json.load(f)
    _, local_poses = load_scal3r_poses(os.path.join(run_dir, "extri.yml"))

    corrected = json.loads(json.dumps(global_trajectory))
    matcher = MINIMA_Init(matcher_method)
    anchors = []
    corrections = []
    active_transform = None
    accepted_anchor_observations = []

    def _apply_anchor_transform(xy, transform):
        if transform is None:
            return np.array(xy, dtype=np.float64)
        rel_xy = np.array(xy, dtype=np.float64) - transform["pred_xy"]
        return transform["corr_xy"] + _rotate_enu_clockwise(rel_xy, transform["yaw_delta"])

    def _interp_yaw_delta(start_delta, target_delta, weight):
        return float(start_delta + _wrap_angle_deg(target_delta - start_delta) * weight)

    def _propagate_previous_transform(chunk, anchor_name, owned_names, reason):
        if active_transform is None:
            return False
        for name in owned_names:
            pose = global_trajectory[name]
            old_xy = np.array(pose["Translation_UTM"][:2], dtype=np.float64)
            new_xy = _apply_anchor_transform(old_xy, active_transform)
            new_t = list(pose["Translation_UTM"])
            new_t[0] = float(new_xy[0])
            new_t[1] = float(new_xy[1])
            corrected[name]["Translation_UTM"] = new_t
            _apply_yaw_delta_to_global_orientation(corrected[name], active_transform["yaw_delta"])
            corrected[name]["Segment_SE2_Correction"] = {
                "chunk_idx": int(chunk["chunk_idx"]),
                "anchor_frame": anchor_name,
                "mode": "propagated_previous_transform",
                "reason": reason,
                "yaw_delta_deg": float(active_transform["yaw_delta"]),
                "applied_weight": 1.0,
            }
        return True

    for chunk in chunk_records:
        frame_names = [n for n in chunk.get("frames", []) if n in global_trajectory]
        owned_names = [n for n in chunk.get("owned_frames", []) if n in global_trajectory]
        if not frame_names or not owned_names:
            continue
        anchor_name = chunk.get("anchor_frame") if chunk.get("anchor_frame") in frame_names else frame_names[-1]
        print(f"[MINIMA-SE2] chunk {chunk['chunk_idx']:04d}: anchor={anchor_name}, owned_frames={len(owned_names)}")
        try:
            anchor = match_segment_anchor_with_minima(
                frame_name=anchor_name,
                frame_pose=global_trajectory[anchor_name],
                ortho_path=ortho_path,
                ortho_meta=ortho_meta,
                local_poses=local_poses,
                matcher=matcher,
                save_dir=save_dir,
                region=region,
                opt=opt,
                region_config=region_config,
                ref_map_full=ref_map_full,
                true_pos=true_pos,
                csv_path=csv_path,
                search_margin_m=search_margin_m,
                min_conf=min_conf,
                min_inliers=min_inliers,
                ransac_reproj_threshold=ransac_reproj_threshold,
                overlay_alpha=overlay_alpha,
                anchor_patch_radius_m=anchor_patch_radius_m,
            )
        except Exception as exc:
            anchor = {
                "ok": False,
                "frame": anchor_name,
                "reason": f"exception:{exc}",
            }

        anchors.append(anchor)
        if not anchor.get("ok"):
            print(f"[MINIMA-SE2] chunk {chunk['chunk_idx']:04d} skipped: {anchor.get('reason')}")
            continue

        pred_anchor_xy = np.array(anchor["pred_utm"], dtype=np.float64)
        corr_anchor_xy = np.array(anchor["corrected_utm"], dtype=np.float64)
        yaw_delta = float(anchor["yaw_delta_deg"])

        quality_gate = _validate_minima_se2_anchor_quality(
            anchor,
            min_inliers=min_inliers,
            min_valid_matches=min_inliers,
            min_inlier_ratio=anchor_min_inlier_ratio,
            homography_scale_min=anchor_homography_scale_min,
            homography_scale_max=anchor_homography_scale_max,
            max_homography_condition=anchor_max_homography_condition,
            max_homography_perspective_norm=anchor_max_homography_perspective_norm,
            max_homography_rmse_px=anchor_max_homography_rmse_px,
            min_local_valid_ratio=anchor_min_local_valid_ratio,
        )
        if not anchor_match_quality_gate_enabled:
            quality_gate["accepted"] = True
            quality_gate["disabled_reason"] = "anchor_match_quality_gate_enabled=False"
        anchor["match_quality_gate"] = quality_gate
        anchor["anchor_gate"] = {
            "enabled": False,
            "policy": "disabled; anchor acceptance uses match_quality_gate only",
            "deprecated_params": {
                "anchor_geometry_gate_enabled": bool(anchor_geometry_gate_enabled),
                "max_anchor_innovation_m": float(max_anchor_innovation_m),
                "max_anchor_yaw_jump_deg": float(max_anchor_yaw_jump_deg),
                "max_first_anchor_delta_m": float(max_first_anchor_delta_m),
            },
        }
        if not quality_gate["accepted"]:
            anchor["ok"] = False
            anchor["reason"] = "match_quality_gate_rejected:" + ";".join(quality_gate.get("reasons", []))
            print(
                f"[MINIMA-SE2] chunk {chunk['chunk_idx']:04d} rejected by match quality gate: "
                f"{anchor['reason']}"
            )
            continue

        target_transform = {
            "pred_xy": pred_anchor_xy,
            "corr_xy": corr_anchor_xy,
            "yaw_delta": yaw_delta,
        }
        active_transform = target_transform
        accepted_anchor_observations.append({
            "chunk_idx": int(chunk["chunk_idx"]),
            "frame": anchor_name,
            "pred_xy": pred_anchor_xy.tolist(),
            "corr_xy": corr_anchor_xy.tolist(),
            "yaw_delta": float(yaw_delta),
            "inlier_count": int(anchor["inlier_count"]),
        })
        corrections.append({
            "chunk_idx": int(chunk["chunk_idx"]),
            "anchor_frame": anchor_name,
            "owned_frames": owned_names,
            "yaw_delta_deg": float(yaw_delta),
            "translation_delta_m": (corr_anchor_xy - pred_anchor_xy).tolist(),
            "inlier_count": int(anchor["inlier_count"]),
            "mode": "anchor_observation_only",
        })
        print(
            f"[MINIMA-SE2] chunk {chunk['chunk_idx']:04d} applied: "
            f"dE={corr_anchor_xy[0] - pred_anchor_xy[0]:.2f}m, "
            f"dN={corr_anchor_xy[1] - pred_anchor_xy[1]:.2f}m, "
            f"dYaw={yaw_delta:.2f}deg, inliers={anchor['inlier_count']}"
        )

    pose_graph_summary = {
        "enabled": True,
        "mode": "first_frame_fixed_odometry_strong_anchor_soft",
        "odom_constraint_sigma_m": float(odom_constraint_sigma_m),
        "anchor_constraint_sigma_m": float(anchor_constraint_sigma_m),
        "anchor_observation_count": int(len(accepted_anchor_observations)),
    }
    pose_names = sorted([k for k in global_trajectory.keys() if not str(k).startswith("_")])
    if len(pose_names) >= 2:
        original_xy = np.array(
            [global_trajectory[name]["Translation_UTM"][:2] for name in pose_names],
            dtype=np.float64,
        )
        n = len(pose_names)
        name_to_idx = {name: i for i, name in enumerate(pose_names)}
        first_xy = original_xy[0].copy()
        rows = []
        rhs_e = []
        rhs_n = []

        def _add_equation(coeffs, target_xy, weight):
            rows.append(coeffs * float(weight))
            rhs_e.append(float(target_xy[0]) * float(weight))
            rhs_n.append(float(target_xy[1]) * float(weight))

        odom_w = 1.0 / max(float(odom_constraint_sigma_m), 1e-6)
        anchor_w_base = 1.0 / max(float(anchor_constraint_sigma_m), 1e-6)
        var_count = max(0, n - 1)
        for i in range(1, n):
            delta_xy = original_xy[i] - original_xy[i - 1]
            coeff = np.zeros(var_count, dtype=np.float64)
            coeff[i - 1] = 1.0
            target = delta_xy.copy()
            if i == 1:
                target = first_xy + delta_xy
            else:
                coeff[i - 2] -= 1.0
            _add_equation(coeff, target, odom_w)

        used_anchor_frames = []
        for obs in accepted_anchor_observations:
            idx = name_to_idx.get(obs["frame"])
            if idx is None or idx == 0:
                continue
            coeff = np.zeros(var_count, dtype=np.float64)
            coeff[idx - 1] = 1.0
            inlier_boost = math.sqrt(max(1.0, min(float(obs.get("inlier_count", 1)), 900.0)) / 300.0)
            _add_equation(coeff, np.asarray(obs["corr_xy"], dtype=np.float64), anchor_w_base * inlier_boost)
            used_anchor_frames.append(obs["frame"])

        if rows and var_count > 0:
            A = np.vstack(rows)
            be = np.asarray(rhs_e, dtype=np.float64)
            bn = np.asarray(rhs_n, dtype=np.float64)
            sol_e, *_ = np.linalg.lstsq(A, be, rcond=None)
            sol_n, *_ = np.linalg.lstsq(A, bn, rcond=None)
            optimized_xy = original_xy.copy()
            optimized_xy[0] = first_xy
            optimized_xy[1:, 0] = sol_e
            optimized_xy[1:, 1] = sol_n

            anchor_indices = [0]
            anchor_yaws = [0.0]
            for obs in accepted_anchor_observations:
                idx = name_to_idx.get(obs["frame"])
                if idx is not None:
                    anchor_indices.append(int(idx))
                    anchor_yaws.append(float(obs["yaw_delta"]))
            order = np.argsort(anchor_indices)
            anchor_indices = np.asarray(anchor_indices, dtype=np.float64)[order]
            anchor_yaws = np.asarray(anchor_yaws, dtype=np.float64)[order]

            max_pos_delta = 0.0
            for i, name in enumerate(pose_names):
                pose = corrected[name]
                old_xy = original_xy[i]
                new_xy = optimized_xy[i]
                max_pos_delta = max(max_pos_delta, float(np.linalg.norm(new_xy - old_xy)))
                new_t = list(pose["Translation_UTM"])
                new_t[0] = float(new_xy[0])
                new_t[1] = float(new_xy[1])
                pose["Translation_UTM"] = new_t
                if len(anchor_indices) > 1:
                    applied_yaw_delta = float(np.interp(float(i), anchor_indices, anchor_yaws))
                else:
                    applied_yaw_delta = 0.0
                _apply_yaw_delta_to_global_orientation(pose, applied_yaw_delta)
                pose["Segment_SE2_Correction"] = {
                    "mode": "pose_graph_odom_anchor_fusion",
                    "first_frame_fixed": bool(lock_first_frame_prior),
                    "odom_constraint_sigma_m": float(odom_constraint_sigma_m),
                    "anchor_constraint_sigma_m": float(anchor_constraint_sigma_m),
                    "applied_yaw_delta_deg": float(applied_yaw_delta),
                    "position_delta_east_m": float(new_xy[0] - old_xy[0]),
                    "position_delta_north_m": float(new_xy[1] - old_xy[1]),
                    "used_anchor_frames": used_anchor_frames,
                }
            pose_graph_summary.update({
                "optimized": True,
                "variable_pose_count": int(var_count),
                "equation_count": int(A.shape[0]),
                "used_anchor_frames": used_anchor_frames,
                "max_position_delta_m": float(max_pos_delta),
            })
        else:
            pose_graph_summary.update({
                "optimized": False,
                "reason": "insufficient_equations_or_variables",
            })
    else:
        pose_graph_summary.update({
            "optimized": False,
            "reason": "fewer_than_two_poses",
        })

    corrected["_segment_minima_se2"] = {
        "matcher": matcher_method,
        "chunk_count": int(len(chunk_records)),
        "anchor_count": int(len(anchors)),
        "applied_count": int(len(corrections)),
        "anchors": anchors,
        "corrections": corrections,
        "anchor_gate": {
            "enabled": False,
            "policy": "disabled; not used for accepting/rejecting anchors",
            "deprecated_params": {
                "anchor_geometry_gate_enabled": bool(anchor_geometry_gate_enabled),
                "max_anchor_innovation_m": float(max_anchor_innovation_m),
                "max_anchor_yaw_jump_deg": float(max_anchor_yaw_jump_deg),
                "max_first_anchor_delta_m": float(max_first_anchor_delta_m),
            },
        },
        "match_quality_gate": {
            "enabled": bool(anchor_match_quality_gate_enabled),
            "policy": "accept anchors using MINIMA/RANSAC match quality and transform validity only",
            "min_inliers": int(min_inliers),
            "min_valid_matches": int(min_inliers),
            "min_inlier_ratio": float(anchor_min_inlier_ratio),
            "homography_scale_range": [float(anchor_homography_scale_min), float(anchor_homography_scale_max)],
            "max_homography_condition": float(anchor_max_homography_condition),
            "max_homography_perspective_norm": float(anchor_max_homography_perspective_norm),
            "max_homography_rmse_px": float(anchor_max_homography_rmse_px),
            "min_local_valid_ratio": float(anchor_min_local_valid_ratio),
        },
        "ownership": "each correction segment owns exactly its configured frame interval",
        "lock_first_frame_prior": bool(lock_first_frame_prior),
        "pose_graph": pose_graph_summary,
        "chunk0_policy": "no direct chunk ramp; first frame is fixed by the pose graph prior",
        "segment_transition_policy": "anchors are soft observations; Scal3R adjacent-frame odometry is a strong constraint",
        "removed_global_rigid_fields": [],
        "note": "Anchor matches do not directly drag whole chunks. Final positions are solved by a small 2D pose graph with first-frame fixed, odometry strong, anchors soft.",
    }
    overview_path = save_segment_minima_match_overview(save_dir, anchors)
    if overview_path is not None:
        corrected["_segment_minima_se2"]["match_overview"] = overview_path
 
    out_path = os.path.join(save_dir, "global_poses_utm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=4, ensure_ascii=False)
    with open(os.path.join(save_dir, "segment_minima_se2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(corrected["_segment_minima_se2"], f, indent=2, ensure_ascii=False)
    return corrected, corrected["_segment_minima_se2"]

def apply_tail_minima_se2_correction(
        global_trajectory,
        chunk_records,
        run_dir,
        save_dir,
        ortho_path,
        true_pos,
        csv_path,
        region,
        opt,
        region_config,
        ref_map_full,
        search_margin_m=100.0,
        min_conf=0.0,
        min_inliers=12,
        ransac_reproj_threshold=5.0,
        overlay_alpha=0.55,
        matcher_method="MINIMA_Roma"
    ):
    from scalloc.matchers.minima import MINIMA_Init

    valid_chunks = [
        c for c in chunk_records
        if any(n in global_trajectory for n in c.get("frames", []))
    ]
    if not valid_chunks:
        raise RuntimeError("No valid chunk records for tail MINIMA SE(2) correction.")

    tail_chunk = valid_chunks[-1]
    tail_frame = [n for n in tail_chunk.get("frames", []) if n in global_trajectory][-1]
    print(f"[TAIL-MINIMA-SE2] tail chunk={tail_chunk['chunk_idx']:04d}, tail frame={tail_frame}")

    meta_path = os.path.splitext(ortho_path)[0] + "_meta.json"
    with open(meta_path, "r", encoding="utf-8") as f:
        ortho_meta = json.load(f)
    _, local_poses = load_scal3r_poses(os.path.join(run_dir, "extri.yml"))
    matcher = MINIMA_Init(matcher_method)

    anchor = match_segment_anchor_with_minima(
        frame_name=tail_frame,
        frame_pose=global_trajectory[tail_frame],
        ortho_path=ortho_path,
        ortho_meta=ortho_meta,
        local_poses=local_poses,
        matcher=matcher,
        save_dir=save_dir,
        region=region,
        opt=opt,
        region_config=region_config,
        ref_map_full=ref_map_full,
        true_pos=true_pos,
        csv_path=csv_path,
        search_margin_m=search_margin_m,
        min_conf=min_conf,
        min_inliers=min_inliers,
        ransac_reproj_threshold=ransac_reproj_threshold,
        overlay_alpha=overlay_alpha,
    )
    if not anchor.get("ok"):
        raise RuntimeError(f"Tail MINIMA anchor failed: {anchor.get('reason')}")

    frame_keys = sorted([k for k in global_trajectory.keys() if not str(k).startswith("_")])
    corrected = json.loads(json.dumps(global_trajectory))
    first_frame = frame_keys[0]
    first_xy = np.array(global_trajectory[first_frame]["Translation_UTM"][:2], dtype=np.float64)
    pred_tail_xy = np.array(anchor["pred_utm"], dtype=np.float64)
    corrected_tail_xy = np.array(anchor["corrected_utm"], dtype=np.float64)
    homography_yaw_delta = float(anchor["yaw_delta_deg"])

    pred_vec = pred_tail_xy - first_xy
    corrected_vec = corrected_tail_xy - first_xy
    if np.linalg.norm(pred_vec) > 1e-6 and np.linalg.norm(corrected_vec) > 1e-6:
        pred_bearing = math.degrees(math.atan2(pred_vec[0], pred_vec[1]))
        corrected_bearing = math.degrees(math.atan2(corrected_vec[0], corrected_vec[1]))
        se2_yaw_delta = _wrap_angle_deg(corrected_bearing - pred_bearing)
    else:
        se2_yaw_delta = homography_yaw_delta

    rotated_tail_xy = first_xy + _rotate_enu_clockwise(pred_vec, se2_yaw_delta)
    tail_residual_xy = corrected_tail_xy - rotated_tail_xy

    denom = max(1, len(frame_keys) - 1)
    for idx, name in enumerate(frame_keys):
        weight = float(idx) / float(denom)
        pose = global_trajectory[name]
        old_xy = np.array(pose["Translation_UTM"][:2], dtype=np.float64)
        rel_xy = old_xy - first_xy
        new_xy = first_xy + _rotate_enu_clockwise(rel_xy, se2_yaw_delta * weight) + tail_residual_xy * weight
        new_t = list(pose["Translation_UTM"])
        new_t[0] = float(new_xy[0])
        new_t[1] = float(new_xy[1])
        corrected[name]["Translation_UTM"] = new_t
        _apply_yaw_delta_to_global_orientation(corrected[name], se2_yaw_delta * weight)
        corrected[name]["Tail_SE2_Correction"] = {
            "tail_chunk_idx": int(tail_chunk["chunk_idx"]),
            "tail_frame": tail_frame,
            "first_frame": first_frame,
            "applied_weight": float(weight),
            "homography_yaw_delta_deg": float(homography_yaw_delta),
            "se2_yaw_delta_deg": float(se2_yaw_delta),
            "tail_delta_east_m": float(corrected_tail_xy[0] - pred_tail_xy[0]),
            "tail_delta_north_m": float(corrected_tail_xy[1] - pred_tail_xy[1]),
        }

    overview_path = save_segment_minima_match_overview(save_dir, [anchor], max_cols=1)
    summary = {
        "matcher": matcher_method,
        "tail_chunk_idx": int(tail_chunk["chunk_idx"]),
        "tail_frame": tail_frame,
        "first_frame": first_frame,
        "corrected_frame_count": int(len(frame_keys)),
        "tail_anchor": anchor,
        "matched_pointcloud_center_utm": anchor["matched_pointcloud_center_utm"],
        "pred_tail_utm": anchor["pred_utm"],
        "corrected_tail_utm": anchor["corrected_utm"],
        "homography_yaw_delta_deg": float(homography_yaw_delta),
        "se2_yaw_delta_deg": float(se2_yaw_delta),
        "translation_delta_m": (corrected_tail_xy - pred_tail_xy).tolist(),
        "tail_residual_after_first_locked_rotation_m": tail_residual_xy.tolist(),
        "first_frame_locked": True,
        "match_overview": overview_path,
        "removed_global_rigid_fields": [],
        "note": "Single tail-frame MINIMA anchor: match point-cloud orthomosaic center first, then back-project tail-frame pose and correct all recovered poses with one SE(2).",
    }
    corrected["_tail_minima_se2"] = summary

    out_path = os.path.join(save_dir, "global_poses_utm.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=4, ensure_ascii=False)
    with open(os.path.join(save_dir, "tail_minima_se2_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return corrected, summary

def recover_global_poses(
        run_dir,
        save_dir,
        pred_utm_x, 
        pred_utm_y, 
        pred_rel_height,
        yaw_offset_deg,
        true_yaw_prior,
        uav_res,
        ortho_meta_path=None,
        truePos=None,
        region=None,
        region_config=None,
        dsm_map_full=None,
        first_frame_known=False,
        gt_csv_path=None
    ):
    """
    纯物理预测版无作弊位姿恢复：
    定位阶段只允许使用 truePos 中的首帧先验；整段 GT CSV 只能在最终可视化评估中使用。
    """
    if gt_csv_path is not None:
        print(
            "[GT-LEAK-GUARD] recover_global_poses() 收到 gt_csv_path，已忽略。"
            " 位姿恢复阶段只允许使用首帧 truePos。"
        )
        gt_csv_path = None
    extri_path = os.path.join(run_dir, "extri.yml")
    meta_path = ortho_meta_path if ortho_meta_path is not None else os.path.join(run_dir, "orthomosaic_meta.json")
    
    print(f"正在读取 Scal3R 相机外参: {extri_path}")
    frame_names, local_poses = load_scal3r_poses(extri_path)
    
    with open(meta_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    # Scale: prefer the BEV scale estimated from the first-frame prior.
    meta_local_to_m = _read_float(meta, ["local_to_meter"])
    if meta_local_to_m is not None and meta_local_to_m > 0:
        scale_local_to_m = meta_local_to_m
        scale_source = meta.get("scale_source", "orthophoto_meta")
    else:
        scale_local_to_m = uav_res / meta["resolution"]
        scale_source = "legacy_camera_gsd_over_bev_resolution"
    ortho_gsd_m = float(meta["resolution"]) * scale_local_to_m
    print(
        f"[BEV scale] source={scale_source}, 1 Local Unit={scale_local_to_m:.6f} m, "
        f"orthophoto_gsd={ortho_gsd_m:.4f} m/px"
    )
    
    # 获取正射图的范围
    u_min = meta["u_min"]
    v_min = meta["v_min"]
    
    # 影像矩阵(以左上角为原点) 和 世界坐标原点(中心)的转换
    # 注意: generate_orthophoto 中的像素索引计算公式为:
    # u_idx = np.floor((u - u_min) / resolution)
    # v_idx = np.floor((v - v_min) / resolution)
    # 然而匹配出来的 (pred_utm_x, pred_utm_y) 是正射影像的*图中心*在世界坐标系中的对应位置.
    # 让我们直接使用第一帧的位置强制作为起点锚点进行分析
    
    u_center = (meta["u_min"] + meta["u_max"]) / 2.0
    v_center = (meta["v_min"] + meta["v_max"]) / 2.0

    # 逆向求解正射图的本地三维原点中心
    h_center = float(meta.get("h_center", 0.0))
    up_axis = meta.get("up_axis", "y")
    up_direction = meta.get("up_direction", -1)
    
    P_leveled_center = np.zeros(3)
    h_orig = h_center * up_direction
    if up_axis == 'y': P_leveled_center[:] = [u_center, h_orig, -v_center]
    elif up_axis == 'z': P_leveled_center[:] = [u_center, -v_center, h_orig]
    elif up_axis == 'x': P_leveled_center[:] = [h_orig, u_center, -v_center]

    R_align = np.array(meta.get("R_align", np.eye(3).tolist()), dtype=np.float64)
    t_align = np.array(meta.get("t_align", [0.0, 0.0, 0.0]), dtype=np.float64)
    P_orig_center = R_align.T @ (P_leveled_center - t_align)
    first_ground_h = _read_float(meta, ["first_frame_ground_h", "h_center"])
    if first_ground_h is None:
        first_ground_h = float(h_center)
    
    # Rotation: 取第一帧位姿及 IMU 先验，利用匹配修正偏航角 (不看 GT 轨迹走向)
    first_frame_name = sorted(frame_names)[0]
    R0_local = local_poses[first_frame_name]["R"]
    
    # 结合地图检索补偿偏航角：使用粗略罗盘先验 + 图像匹配测算出的精准偏航角修正
    # （在 GPS 丢失的重定位任务里，绝对朝向必须依靠匹配图与底图对齐来确定）
    camera_map_yaw = _wrap_angle_deg(true_yaw_prior + yaw_offset_deg)
    local_forward_bearing_deg, local_forward_dir_cam, local_forward_right_up = _local_camera_forward_bearing_deg(
        local_poses[first_frame_name],
        R_align,
        up_axis,
    )
    local_track_meta = _estimate_local_track_bearing_deg(
        frame_names,
        local_poses,
        R_align,
        t_align,
        up_axis,
    )
    local_track_bearing_deg = None if local_track_meta is None else local_track_meta.get("bearing_deg")
    local_track_vs_forward_deg = None
    if local_track_bearing_deg is not None and local_forward_bearing_deg is not None:
        local_track_vs_forward_deg = _wrap_angle_deg(local_track_bearing_deg - local_forward_bearing_deg)
    if local_forward_bearing_deg is None:
        orthophoto_position_yaw_offset_deg = 0.0
        yaw_source = "fallback_camera_yaw_no_local_forward"
    else:
        orthophoto_position_yaw_offset_deg = float(-local_forward_bearing_deg)
        yaw_source = "camera_yaw_minus_scal3r_local_forward_bearing"
    position_yaw = _wrap_angle_deg(camera_map_yaw + orthophoto_position_yaw_offset_deg)
    desired_track_yaw = position_yaw
    use_track_bearing = False
    use_orbit_tangent = False
    print(
        f"[Yaw split] camera_yaw={camera_map_yaw:.2f} deg, "
        f"desired_track_yaw={desired_track_yaw:.2f} deg, "
        f"local_track_bearing={local_track_bearing_deg if local_track_bearing_deg is not None else float('nan'):.2f} deg, "
        f"track_ratio={local_track_meta.get('chord_path_ratio', float('nan')) if local_track_meta is not None else float('nan'):.3f}, "
        f"turn_span={local_track_meta.get('turn_span_deg', float('nan')) if local_track_meta is not None else float('nan'):.1f} deg, "
        f"local_forward_bearing={local_forward_bearing_deg if local_forward_bearing_deg is not None else float('nan'):.2f} deg, "
        f"track_vs_forward={local_track_vs_forward_deg if local_track_vs_forward_deg is not None else float('nan'):.2f} deg, "
        f"auto_offset={orthophoto_position_yaw_offset_deg:.1f} deg, "
        f"use_orbit_tangent={use_orbit_tangent}, "
        f"position_yaw={position_yaw:.2f} deg ({yaw_source})"
    )
    
    # 【修复漂移】统一使用与正射匹配完全一致的 2D 变换，废弃依赖不准的 IMU Pitch/Roll
    # 点云已经被 auto_level_point_cloud 通过 R_align 矫正为水平，
    # 正射影像是基于矫正后的坐标生成的。必须使用这套矫正后的系统！
    
    # Translation: 全局重定位任务的核心 -> 绝对信赖大图检索(DINO)出的 UTM 坐标作为物理质心
    
    # [不依赖真实绝对高度进行求解]：我们通过查阅 DSM 高程图和相对高度(rel_height)来推算初始 Z 轴
    base_elevation = 0.0
    if dsm_map_full is not None and region_config is not None:
        try:
            dsm_res = region_config.get(f'{region}_DSM_resolution', 1.0)
            orig_initialX = region_config.get(f'{region}_DSM_initialX', 0)
            orig_initialY = region_config.get(f'{region}_DSM_initialY', 0)
            col_idx = int((pred_utm_x - orig_initialX) / dsm_res)
            row_idx = int((orig_initialY - pred_utm_y) / dsm_res)
            
            if 0 <= row_idx < dsm_map_full.shape[0] and 0 <= col_idx < dsm_map_full.shape[1]:
                dsm_val = float(dsm_map_full[row_idx, col_idx])
                if dsm_val > -1000:
                    base_elevation = dsm_val
        except Exception as e:
            print(f"[警告] 无法从 DSM 地图中提取基准海拔，使用 0.0 替代: {e}")
            
    first_anchor_height = base_elevation + pred_rel_height
    global_center_utm = np.array([pred_utm_x, pred_utm_y, first_anchor_height])
    print(f"应用全局检索锚点 (不作弊使用相对测量Z): UTM: X={global_center_utm[0]:.2f}, Y={global_center_utm[1]:.2f}, Z={global_center_utm[2]:.2f} (底 {base_elevation:.1f} + 相对 {pred_rel_height:.1f})")
    
    # ENU平面上的纯 2D 旋转矩阵。position_yaw 控制相机中心轨迹位置，
    # camera_map_yaw 只表示首帧相机朝向语义。
    # UTM坐标系：X=East, Y=North. 正射影像的 u = Right, -v = Up (即North)
    yaw_rad = np.deg2rad(position_yaw)
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    R_yaw_enu = _yaw_rotation_enu_clockwise_matrix(position_yaw)
    R_leveled_to_enu = _leveled_to_enu_matrix(up_axis, up_direction)
    R_cam_to_body = _camera_to_body_rotation_opencv()
    
    # 建立从 Leveled 坐标系 (u, v) 到 ENU 平面 (East, North) 的严密变换矩阵
    # 影像中的朝向：偏航为正表示顺时针旋转。
    # 影像初始北为 -v, 初东为 u.
    
    # 解决轨迹“首帧平移缩放飘移”：
    # 虽然 DINO 匹配出了整张图的中心，但由于图像内物体实际偏向一侧（比如正射图大量留白），
    # 直接以图中心作为质心可能会有几米到十几米的偏移。
    # 最严谨的做法是：取无人机在局部模型里第一帧的光心位置，
    # 看它在正射影像(Local)中的坐标，与红框图心算出一个局部偏差，
    # 然后再映射到全局 UTM，这样可以确切保证以无人机的首帧作为无偏起步。

    # 第一帧在局部物理系中的投影坐标
    first_frame_name = sorted(frame_names)[0]
    R0 = local_poses[first_frame_name]["R"]
    T0 = local_poses[first_frame_name]["T"]
    cam_center_0 = -R0.T @ T0
    cam_leveled_0 = R_align @ cam_center_0 + t_align
    
    u0, v0 = _camera_uv_from_local_pose(local_poses[first_frame_name], R_align, t_align, up_axis)
    
    # 相比 DINO 匹配出来的图中心 (u_center, v_center) 的像素物理偏差：
    du_0 = u0 - u_center
    dv_0 = v0 - v_center 
    
    du0_m = du_0 * scale_local_to_m
    dv0_m = dv_0 * scale_local_to_m
    
    east_offset_0  = du0_m * cos_y + (-dv0_m) * sin_y
    north_offset_0 = -du0_m * sin_y + (-dv0_m) * cos_y
    
    # 这就是无人机第一帧的绝对 UTM 坐标（不受整图留白质心偏移影响）
    if first_frame_known and truePos is not None:
        first_frame_utm_x, first_frame_utm_y = deg2utm(region, region_config, truePos['lon'], truePos['lat'])
        print(f"依据已知真实首帧信息 -> 第1帧绝对位置: X={first_frame_utm_x:.2f}, Y={first_frame_utm_y:.2f}")
        pred_utm_x = first_frame_utm_x - east_offset_0
        pred_utm_y = first_frame_utm_y - north_offset_0
    else:
        first_frame_utm_x = pred_utm_x + east_offset_0
        first_frame_utm_y = pred_utm_y + north_offset_0
        print(f"依据图内物理反算修正锚点 -> 第1帧绝对位置: X={first_frame_utm_x:.2f}, Y={first_frame_utm_y:.2f}")

    global_trajectory = {}
    
    # --- 计算正射影像的参考边界框。为避免 GT 泄露，这里只使用首帧先验。
    # Full-sequence GT is intentionally kept out of pose recovery and only used
    # later by visualize_comparison_on_map() for final evaluation.
    gt_sequence_est = None

    if gt_sequence_est is not None:
        gt_ortho_center_utm_x = gt_sequence_est["utm_x"]
        gt_ortho_center_utm_y = gt_sequence_est["utm_y"]
        gt_yaw_used = gt_sequence_est["yaw_deg"]
        print(
            f"[PointCloud GT] using sequence GT: center=({gt_ortho_center_utm_x:.2f}, "
            f"{gt_ortho_center_utm_y:.2f}), yaw={gt_yaw_used:.2f} deg, "
            f"frames={gt_sequence_est['frame_count']}, center_rmse={gt_sequence_est['rmse_m']:.2f} m, "
            f"max_residual={gt_sequence_est['max_residual_m']:.2f} m"
        )
    else:
        gt_utm_x, gt_utm_y = deg2utm(region, region_config, truePos['lon'], truePos['lat'])
        gt_yaw_used = true_yaw_prior
        gt_yaw_rad = np.deg2rad(gt_yaw_used)
        cos_gt, sin_gt = np.cos(gt_yaw_rad), np.sin(gt_yaw_rad)

        # 倒推 GT 状态下的正射图中心
        gt_east_offset_0  = du0_m * cos_gt + (-dv0_m) * sin_gt
        gt_north_offset_0 = -du0_m * sin_gt + (-dv0_m) * cos_gt
        gt_ortho_center_utm_x = gt_utm_x - gt_east_offset_0
        gt_ortho_center_utm_y = gt_utm_y - gt_north_offset_0
        print(
            f"[PointCloud GT] sequence GT unavailable; fallback to first frame: "
            f"center=({gt_ortho_center_utm_x:.2f}, {gt_ortho_center_utm_y:.2f}), "
            f"yaw={gt_yaw_used:.2f} deg"
        )

    gt_yaw_rad = np.deg2rad(gt_yaw_used)
    cos_gt, sin_gt = np.cos(gt_yaw_rad), np.sin(gt_yaw_rad)

    pred_poly_utm = []
    gt_poly_utm = []
    corners = [
        (meta["u_min"], meta["v_min"]),
        (meta["u_max"], meta["v_min"]),
        (meta["u_max"], meta["v_max"]),
        (meta["u_min"], meta["v_max"])
    ]
    for (cu, cv) in corners:
        c_du_m = (cu - u_center) * scale_local_to_m
        c_dv_m = (cv - v_center) * scale_local_to_m
        
        # Pred 包围框 (用预测的 Yaw 和 Pred 图中心)
        p_east = c_du_m * cos_y + (-c_dv_m) * sin_y
        p_north = -c_du_m * sin_y + (-c_dv_m) * cos_y
        pred_poly_utm.append((pred_utm_x + p_east, pred_utm_y + p_north))
        
        # GT 包围框 (用先验 Yaw 和 GT 图中心)
        g_east = c_du_m * cos_gt + (-c_dv_m) * sin_gt
        g_north = -c_du_m * sin_gt + (-c_dv_m) * cos_gt
        gt_poly_utm.append((gt_ortho_center_utm_x + g_east, gt_ortho_center_utm_y + g_north))
    
    global_trajectory["_ortho_polygon_pred"] = pred_poly_utm
    global_trajectory["_ortho_polygon_gt"] = gt_poly_utm
    pred_center_eval = np.array(pred_poly_utm, dtype=np.float64).mean(axis=0)
    gt_center_eval = np.array(gt_poly_utm, dtype=np.float64).mean(axis=0)
    pc_center_delta = pred_center_eval - gt_center_eval
    global_trajectory["_pointcloud_eval"] = {
        "pred_center_utm": pred_center_eval.tolist(),
        "gt_center_utm": gt_center_eval.tolist(),
        "center_error_m": float(np.linalg.norm(pc_center_delta)),
        "delta_east_m": float(pc_center_delta[0]),
        "delta_north_m": float(pc_center_delta[1]),
        "gt_source": "sequence_csv" if gt_sequence_est is not None else "first_frame_fallback",
        "gt_frame_count": int(gt_sequence_est["frame_count"]) if gt_sequence_est is not None else 1,
        "gt_center_rmse_m": float(gt_sequence_est["rmse_m"]) if gt_sequence_est is not None else 0.0,
    }
    print(
        f"[PointCloud Eval] pred_center=({pred_center_eval[0]:.2f}, {pred_center_eval[1]:.2f}), "
        f"gt_center=({gt_center_eval[0]:.2f}, {gt_center_eval[1]:.2f}), "
        f"error={global_trajectory['_pointcloud_eval']['center_error_m']:.2f} m, "
        f"dE={pc_center_delta[0]:.2f} m, dN={pc_center_delta[1]:.2f} m"
    )

    print("\n================ 基于严格二维正射投影匹配矩阵的绝对位姿 ================")
    global_trajectory["_yaw_alignment"] = {
        "camera_map_yaw_deg": float(camera_map_yaw),
        "desired_track_yaw_deg": float(desired_track_yaw),
        "position_yaw_deg": float(position_yaw),
        "orthophoto_position_yaw_offset_deg": float(orthophoto_position_yaw_offset_deg),
        "yaw_source": yaw_source,
        "local_track": local_track_meta,
        "used_track_bearing": bool(use_track_bearing),
        "used_orbit_tangent": bool(use_orbit_tangent),
        "local_track_vs_forward_deg": None if local_track_vs_forward_deg is None else float(local_track_vs_forward_deg),
        "local_forward_bearing_deg": None if local_forward_bearing_deg is None else float(local_forward_bearing_deg),
        "local_forward_dir_cam": local_forward_dir_cam,
        "local_forward_right_up": local_forward_right_up,
        "note": "Straight trajectories use local track bearing; orbit-like trajectories use local camera forward bearing.",
    }
    for name in frame_names:
        local_R = local_poses[name]["R"]
        local_T = local_poses[name]["T"]
        
        # 1. 取出原始 3D 光心坐标
        cam_center = -local_R.T @ local_T
        
        # 2. 映射到 Leveled (正射) 坐标系
        cam_leveled = R_align @ cam_center + t_align
        _, _, cam_h_arr = _orthophoto_axes(cam_leveled.reshape(1, 3), up_axis, up_direction)
        pred_frame_rel_height = abs(float(cam_h_arr[0]) - float(first_ground_h)) * float(scale_local_to_m)
        
        u, v = _camera_uv_from_local_pose(local_poses[name], R_align, t_align, up_axis)
        
        # 3. [关键！] 直接计算当前帧相对于第一帧的物理位置差
        du_rel = (u - u0) * scale_local_to_m
        dv_rel = (v - v0) * scale_local_to_m
        
        up_m_rel = -dv_rel
        right_m_rel = du_rel
        
        # 4. 在 ENU (East, North) 中顺时针旋转 Yaw 角度：
        east_rel  = right_m_rel * cos_y + up_m_rel * sin_y
        north_rel = -right_m_rel * sin_y + up_m_rel * cos_y
        
        enu_offset_rel = np.array([east_rel, north_rel, 0.0])
        
        # 5. 加上第一帧的锚点中心
        global_T = np.array([first_frame_utm_x, first_frame_utm_y, base_elevation + pred_frame_rel_height]) + enu_offset_rel
        R_cam_to_leveled = R_align @ local_R.T
        R_cam_to_enu = R_yaw_enu @ R_leveled_to_enu @ R_cam_to_leveled
        R_body_to_enu = R_cam_to_enu @ R_cam_to_body
        ypr = _pred_pose_ypr_in_gt_csv_convention(R_body_to_enu)
        
        global_trajectory[name] = {
            "Translation_UTM": global_T.tolist(),
            "Height": float(pred_frame_rel_height),
            "Height_Source": "scal3r_local_height_delta_from_first_frame_ground",
            "Pitch_deg": float(ypr["Pitch_deg"]),
            "Roll_deg": float(ypr["Roll_deg"]),
            "Yaw_deg": float(ypr["Yaw_deg"]),
            "Ortho_Yaw_deg": float(_wrap_angle_deg(position_yaw)),
            "Map_Yaw_deg": float(_wrap_angle_deg(position_yaw)),
            "Camera_Map_Yaw_deg": float(camera_map_yaw),
            "Desired_Track_Yaw_deg": float(desired_track_yaw),
            "Orthophoto_Position_Yaw_Offset_deg": float(orthophoto_position_yaw_offset_deg),
            "Local_Track_Bearing_deg": None if local_track_bearing_deg is None else float(local_track_bearing_deg),
            "Local_Track_vs_Forward_deg": None if local_track_vs_forward_deg is None else float(local_track_vs_forward_deg),
            "Used_Track_Bearing_For_Position_Yaw": bool(use_track_bearing),
            "Used_Orbit_Tangent_For_Position_Yaw": bool(use_orbit_tangent),
            "Local_Forward_Bearing_deg": None if local_forward_bearing_deg is None else float(local_forward_bearing_deg),
            "Position_Yaw_Source": yaw_source,
            "Local_Forward_Dir_Cam": local_forward_dir_cam,
            "Local_Forward_Right_Up": local_forward_right_up,
            "Rotation_Matrix": R_body_to_enu.tolist()
        }
        
    out_path = os.path.join(save_dir, "global_poses_utm.json")
    with open(out_path, "w", encoding='utf-8') as f:
        json.dump(global_trajectory, f, indent=4)
        
    return global_trajectory

def _match_pred_gt_for_metrics(global_trajectory, csv_path, region, region_config):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'num' in df.columns:
        df = df.sort_values(by='num').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df_by_stem = {}
    if 'filename' in df.columns:
        for _, row in df.iterrows():
            stem = _csv_frame_stem(row['filename'])
            if stem:
                df_by_stem[stem] = row

    pred_keys = sorted([k for k in global_trajectory.keys() if not str(k).startswith("_")])
    matched_pairs = []
    fallback_match_count = 0

    for display_idx, name_key in enumerate(pred_keys):
        row = df_by_stem.get(str(name_key))
        if row is None:
            row = df_by_stem.get(_csv_frame_stem(name_key))
        if row is None:
            if display_idx >= len(df):
                continue
            row = df.iloc[display_idx]
            fallback_match_count += 1

        pred_entry = global_trajectory.get(name_key, {})
        pred_utm_t = pred_entry.get("Translation_UTM")
        if pred_utm_t is None or len(pred_utm_t) < 2:
            continue

        gt_utm_x, gt_utm_y = deg2utm(region, region_config, row['lon'], row['lat'])
        pred_x = float(pred_utm_t[0])
        pred_y = float(pred_utm_t[1])
        delta_e = pred_x - float(gt_utm_x)
        delta_n = pred_y - float(gt_utm_y)
        pos_err_m = float(np.hypot(delta_e, delta_n))

        gt_yaw = None
        yaw_err_deg = None
        pred_yaw = None
        if "Yaw_deg" in pred_entry:
            pred_yaw = float(pred_entry["Yaw_deg"])
        if 'yaw' in row.index and not pd.isna(row['yaw']) and pred_yaw is not None:
            gt_yaw = float(row['yaw'])
            yaw_err_deg = float(_wrap_angle_deg(pred_yaw - gt_yaw))

        label_idx = int(row['num']) if 'num' in row.index and not pd.isna(row['num']) else display_idx
        matched_pairs.append({
            "idx": int(display_idx),
            "label_idx": int(label_idx),
            "pred_key": str(name_key),
            "gt_row_index": int(row.name) if row.name is not None else int(display_idx),
            "gt_utm": [float(gt_utm_x), float(gt_utm_y)],
            "pred_utm": [pred_x, pred_y],
            "delta_east_m": float(delta_e),
            "delta_north_m": float(delta_n),
            "position_error_m": pos_err_m,
            "gt_yaw_deg": gt_yaw,
            "pred_yaw_deg": pred_yaw,
            "yaw_error_deg": yaw_err_deg,
        })

    return matched_pairs, {
        "pred_frame_count": int(len(pred_keys)),
        "gt_row_count": int(len(df)),
        "fallback_match_count": int(fallback_match_count),
    }


def _safe_stat(values, stat):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    if stat == "mean":
        return float(np.mean(arr))
    if stat == "median":
        return float(np.median(arr))
    if stat == "max":
        return float(np.max(arr))
    if stat == "p95":
        return float(np.percentile(arr, 95))
    if stat == "variance":
        return 0.0 if arr.size == 1 else float(np.var(arr, ddof=1))
    raise ValueError(f"Unknown stat: {stat}")


def compute_and_save_trajectory_metrics(
        global_trajectory,
        csv_path,
        region,
        region_config,
        out_dir,
        runtime_sec=None,
        output_name="trajectory_metrics.pkl",
        label="trajectory",
    ):
    matched_pairs, match_info = _match_pred_gt_for_metrics(global_trajectory, csv_path, region, region_config)

    pos_err = np.array([p["position_error_m"] for p in matched_pairs], dtype=np.float64)
    delta_en = np.array([[p["delta_east_m"], p["delta_north_m"]] for p in matched_pairs], dtype=np.float64)
    yaw_err = np.array(
        [p["yaw_error_deg"] for p in matched_pairs if p["yaw_error_deg"] is not None],
        dtype=np.float64,
    )
    yaw_abs_err = np.abs(yaw_err)

    if delta_en.shape[0] >= 2:
        pos_delta_cov = np.cov(delta_en.T, ddof=1).astype(float).tolist()
    else:
        pos_delta_cov = [[0.0, 0.0], [0.0, 0.0]] if delta_en.shape[0] == 1 else None

    threshold_recall = {}
    for threshold in [5.0, 10.0, 20.0, 30.0]:
        key = f"{int(threshold)}m"
        threshold_recall[key] = None if pos_err.size == 0 else float(np.mean(pos_err <= threshold))

    metrics = {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "label": label,
        "csv_path": csv_path,
        "save_dir": out_dir,
        "runtime_sec": None if runtime_sec is None else float(runtime_sec),
        "sample_count": int(len(matched_pairs)),
        "match_info": match_info,
        "position": {
            "error_mean_m": _safe_stat(pos_err, "mean"),
            "error_variance_m2": _safe_stat(pos_err, "variance"),
            "error_covariance_m2": _safe_stat(pos_err, "variance"),
            "delta_east_north_covariance_m2": pos_delta_cov,
            "error_median_m": _safe_stat(pos_err, "median"),
            "error_max_m": _safe_stat(pos_err, "max"),
            "error_p95_m": _safe_stat(pos_err, "p95"),
            "threshold_recall": threshold_recall,
        },
        "yaw": {
            "sample_count": int(yaw_err.size),
            "signed_error_mean_deg": _safe_stat(yaw_err, "mean"),
            "signed_error_variance_deg2": _safe_stat(yaw_err, "variance"),
            "signed_error_covariance_deg2": _safe_stat(yaw_err, "variance"),
            "abs_error_mean_deg": _safe_stat(yaw_abs_err, "mean"),
            "abs_error_median_deg": _safe_stat(yaw_abs_err, "median"),
            "abs_error_max_deg": _safe_stat(yaw_abs_err, "max"),
            "abs_error_p95_deg": _safe_stat(yaw_abs_err, "p95"),
        },
        "per_frame": matched_pairs,
    }

    out_path = os.path.join(out_dir, output_name)
    with open(out_path, "wb") as f:
        pickle.dump(metrics, f)

    pos = metrics["position"]
    yaw = metrics["yaw"]
    print(f"\n[Metrics] Trajectory evaluation: {label}")
    print(f"[Metrics] samples={metrics['sample_count']} / pred_frames={match_info['pred_frame_count']} / fallback_by_index={match_info['fallback_match_count']}")
    if pos["error_mean_m"] is not None:
        print(
            "[Metrics] position error: "
            f"mean={pos['error_mean_m']:.3f}m, median={pos['error_median_m']:.3f}m, "
            f"max={pos['error_max_m']:.3f}m, p95={pos['error_p95_m']:.3f}m, "
            f"var={pos['error_variance_m2']:.3f}m^2"
        )
    else:
        print("[Metrics] position error: no matched samples")
    print(f"[Metrics] position delta covariance [[E,N],[E,N]] m^2 = {pos['delta_east_north_covariance_m2']}")
    print(
        "[Metrics] position recall: "
        + ", ".join(
            f"@{k}={v * 100.0:.2f}%" if v is not None else f"@{k}=N/A"
            for k, v in pos["threshold_recall"].items()
        )
    )
    if yaw["sample_count"] > 0:
        print(
            "[Metrics] yaw error: "
            f"abs_mean={yaw['abs_error_mean_deg']:.3f}deg, "
            f"abs_median={yaw['abs_error_median_deg']:.3f}deg, "
            f"abs_max={yaw['abs_error_max_deg']:.3f}deg, "
            f"abs_p95={yaw['abs_error_p95_deg']:.3f}deg, "
            f"signed_mean={yaw['signed_error_mean_deg']:.3f}deg, "
            f"signed_var={yaw['signed_error_variance_deg2']:.3f}deg^2"
        )
    else:
        print("[Metrics] yaw error: no yaw samples")
    if runtime_sec is not None:
        print(f"[Metrics] runtime={runtime_sec:.2f}s")
    print(f"[Metrics] saved pkl: {out_path}")
    return metrics

def _get_row_value(row, key):
    if row is None or key not in row.index or pd.isna(row[key]):
        return None
    value = row[key]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    return value

def _get_pose_angle_deg(pose, key):
    if pose is None:
        return None
    value = pose.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def _angle_error_deg(pred, gt):
    if pred is None or gt is None:
        return None
    return float(_wrap_angle_deg(float(pred) - float(gt)))

def export_pose_attitude_comparison_csv(
        uncorrected_trajectory,
        corrected_trajectory,
        csv_path,
        region,
        region_config,
        out_dir,
        output_name="pose_attitude_gt_uncorrected_corrected.csv",
    ):
    """
    Save per-frame GT / uncorrected / corrected attitude only.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"GT CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "num" in df.columns:
        df = df.sort_values(by="num").reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df_by_stem = {}
    if "filename" in df.columns:
        for _, row in df.iterrows():
            stem = _csv_frame_stem(row["filename"])
            if stem:
                df_by_stem[stem] = row

    corrected_keys = sorted([k for k in corrected_trajectory.keys() if not str(k).startswith("_")])
    uncorrected_keys = sorted([k for k in uncorrected_trajectory.keys() if not str(k).startswith("_")])
    key_order = corrected_keys if corrected_keys else uncorrected_keys

    rows = []
    fallback_match_count = 0
    for display_idx, name_key in enumerate(key_order):
        row = df_by_stem.get(str(name_key))
        if row is None:
            row = df_by_stem.get(_csv_frame_stem(name_key))
        match_source = "filename"
        if row is None:
            if display_idx >= len(df):
                continue
            row = df.iloc[display_idx]
            fallback_match_count += 1
            match_source = "index"

        un_pose = uncorrected_trajectory.get(name_key) or uncorrected_trajectory.get(_csv_frame_stem(name_key))
        co_pose = corrected_trajectory.get(name_key) or corrected_trajectory.get(_csv_frame_stem(name_key))

        gt_yaw = _get_row_value(row, "yaw")
        gt_pitch = _get_row_value(row, "pitch")
        gt_roll = _get_row_value(row, "roll")
        un_yaw = _get_pose_angle_deg(un_pose, "Yaw_deg")
        un_pitch = _get_pose_angle_deg(un_pose, "Pitch_deg")
        un_roll = _get_pose_angle_deg(un_pose, "Roll_deg")
        co_yaw = _get_pose_angle_deg(co_pose, "Yaw_deg")
        co_pitch = _get_pose_angle_deg(co_pose, "Pitch_deg")
        co_roll = _get_pose_angle_deg(co_pose, "Roll_deg")

        label_idx = _get_row_value(row, "num")
        rows.append({
            "idx": int(display_idx),
            "label_idx": int(label_idx) if label_idx is not None else int(display_idx),
            "frame": str(name_key),
            "match_source": match_source,
            "gt_filename": _get_row_value(row, "filename"),
            "gt_yaw_deg": gt_yaw,
            "gt_pitch_deg": gt_pitch,
            "gt_roll_deg": gt_roll,
            "uncorrected_yaw_deg": un_yaw,
            "uncorrected_pitch_deg": un_pitch,
            "uncorrected_roll_deg": un_roll,
            "corrected_yaw_deg": co_yaw,
            "corrected_pitch_deg": co_pitch,
            "corrected_roll_deg": co_roll,
        })

    out_path = os.path.join(out_dir, output_name)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
    print(
        f"[Pose CSV] saved attitude comparison {len(rows)} rows -> {out_path} "
        f"(fallback_by_index={fallback_match_count})"
    )
    return out_path


def visualize_comparison_on_map(
        global_trajectory,
        csv_path,
        region,
        opt,
        region_config,
        ref_map_full,
        out_dir,
        crop_size_m=1200,
        stream_window=8,
        stream_stride=5,
        color_pred_by_segment=False,
        draw_trajectory_lines=True,
        draw_trajectory_arrows=True,
        uncorrected_trajectory=None,
        draw_uncorrected_trajectory_points=False,
    ):
    """
    可视化预测轨迹(Pred)与真实轨迹(GT)的对比图
    """
    if not os.path.exists(csv_path):
        print(f"[警告] 找不到 GT CSV 文件: {csv_path}，无法绘制对比图。")
        return

    df = pd.read_csv(csv_path)
    if 'num' in df.columns:
        df = df.sort_values(by='num').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df_by_stem = {}
    if 'filename' in df.columns:
        for _, row in df.iterrows():
            stem = _csv_frame_stem(row['filename'])
            if stem:
                df_by_stem[stem] = row

    ref_res = region_config[f'{region}_{opt.Ref_type}_REF_resolution']
    orig_initialX = region_config[f'{region}_{opt.Ref_type}_REF_initialX']
    orig_initialY = region_config[f'{region}_{opt.Ref_type}_REF_initialY']

    # === 修改为基于整个 GT 和 Pred 轨迹的包围盒来裁切 ===
    all_utm_x = []
    all_utm_y = []
    
    # 获取 GT 轨迹坐标
    for _, row in df.iterrows():
        gt_x, gt_y = deg2utm(region, region_config, row['lon'], row['lat'])
        all_utm_x.append(gt_x)
        all_utm_y.append(gt_y)
        
    # 获取预测轨迹坐标
    for key, val in global_trajectory.items():
        if not str(key).startswith("_") and "Translation_UTM" in val:
            all_utm_x.append(val["Translation_UTM"][0])
            all_utm_y.append(val["Translation_UTM"][1])

    # 如果有无校正的轨迹也加入
    if uncorrected_trajectory:
        for key, val in uncorrected_trajectory.items():
            if not str(key).startswith("_") and "Translation_UTM" in val:
                all_utm_x.append(val["Translation_UTM"][0])
                all_utm_y.append(val["Translation_UTM"][1])

    if len(all_utm_x) > 0:
        min_utm_x, max_utm_x = min(all_utm_x), max(all_utm_x)
        min_utm_y, max_utm_y = min(all_utm_y), max(all_utm_y)
    else:
        # Fallback
        first_frame = df.iloc[0]
        gt_x, gt_y = deg2utm(region, region_config, first_frame['lon'], first_frame['lat'])
        min_utm_x, max_utm_x = gt_x, gt_x
        min_utm_y, max_utm_y = gt_y, gt_y

    # 保留一定的边缘 padding (米)
    padding_m = crop_size_m / 4.0 if crop_size_m > 0 else 300.0
    
    tl_crop_utm_x = min_utm_x - padding_m
    br_crop_utm_x = max_utm_x + padding_m
    # 注意 UTM_Y向北增加，图像原点在左上角
    tl_crop_utm_y = max_utm_y + padding_m
    br_crop_utm_y = min_utm_y - padding_m

    ref_col_start = int((tl_crop_utm_x - orig_initialX) / ref_res)
    ref_row_start = int((orig_initialY - tl_crop_utm_y) / ref_res)
    ref_col_end = int((br_crop_utm_x - orig_initialX) / ref_res)
    ref_row_end = int((orig_initialY - br_crop_utm_y) / ref_res)

    H_full, W_full = ref_map_full.shape[:2]
    ref_col_start = max(0, ref_col_start)
    ref_col_end = min(W_full, max(ref_col_start + 1, ref_col_end))
    ref_row_start = max(0, ref_row_start)
    ref_row_end = min(H_full, max(ref_row_start + 1, ref_row_end))

    ref_crop = ref_map_full[ref_row_start:ref_row_end, ref_col_start:ref_col_end].copy()
    ref_crop_clean = ref_crop.copy()

    new_ref_initialX = orig_initialX + ref_col_start * ref_res
    new_ref_initialY = orig_initialY - ref_row_start * ref_res

    gt_points = []
    pred_points = []
    matched_pairs = []

    # 提取有效的预测轨迹名称（过滤掉下划线开头的隐藏字段）
    pred_keys = sorted([k for k in global_trajectory.keys() if not str(k).startswith("_")])
    has_segment_se2 = (
        "_segment_minima_se2" in global_trajectory
        or "_tail_minima_se2" in global_trajectory
        or "_segment_pnp_3d" in global_trajectory
        or "_segment_minima_pnp_3d" in global_trajectory
        or "_segment_minima_render_2d" in global_trajectory
        or "_segment_minima_swg" in global_trajectory
        or "_segment_satellite_pnp" in global_trajectory
    )

    # 计算 GT 与 Pred 的像素坐标，并按同一帧对齐记录误差(米)
    fallback_match_count = 0
    for display_idx, name_key in enumerate(pred_keys):
        row = df_by_stem.get(str(name_key))
        if row is None:
            row = df_by_stem.get(_csv_frame_stem(name_key))
        if row is None:
            if display_idx >= len(df):
                continue
            row = df.iloc[display_idx]
            fallback_match_count += 1
            
        # 1. 真实坐标 (GT)
        gt_utm_x, gt_utm_y = deg2utm(region, region_config, row['lon'], row['lat'])
        gt_px = int((gt_utm_x - new_ref_initialX) / ref_res)
        gt_py = int((new_ref_initialY - gt_utm_y) / ref_res)
        gt_points.append((gt_px, gt_py))

        # 2. 预测坐标 (Pred) 取出对应的 filename/key
        pred_utm_t = global_trajectory[name_key]["Translation_UTM"]
        p_px = int((pred_utm_t[0] - new_ref_initialX) / ref_res)
        p_py = int((new_ref_initialY - pred_utm_t[1]) / ref_res)
        pred_points.append((p_px, p_py))

        # 逐帧平面误差（米）：UTM East/North 二维距离
        err_m = float(np.hypot(pred_utm_t[0] - gt_utm_x, pred_utm_t[1] - gt_utm_y))
        gt_yaw = float(row['yaw']) if 'yaw' in row.index and not pd.isna(row['yaw']) else None
        label_idx = int(row['num']) if 'num' in row.index and not pd.isna(row['num']) else display_idx
        matched_pairs.append({
            "idx": display_idx,
            "display_idx": display_idx,
            "label_idx": label_idx,
            "pred_key": name_key,
            "gt_row_index": int(row.name) if row.name is not None else display_idx,
            "gt_pt": (gt_px, gt_py),
            "pred_pt": (p_px, p_py),
            "err_m": err_m,
            "gt_yaw": gt_yaw
        })

    print(
        f"[Trajectory Viz] matched {len(matched_pairs)}/{len(pred_keys)} predicted frames "
        f"with GT rows; fallback_by_index={fallback_match_count}"
    )

    # ---- 开始绘制 ----
    
    # 画预测和真实的 Ortho Bounding Box。分段 SE(2) 校正后不存在单一整图刚体变换，
    # 因此不再画旧的正射图整体边界，避免把旧红框误认为校正后结果。
    anchors_for_viz = []
    if "_segment_minima_se2" in global_trajectory:
        anchors_for_viz = global_trajectory["_segment_minima_se2"].get("anchors", [])
    elif "_segment_minima_pnp_3d" in global_trajectory:
        anchors_for_viz = global_trajectory["_segment_minima_pnp_3d"].get("se2_anchors", [])
    elif "_segment_minima_render_2d" in global_trajectory:
        anchors_for_viz = global_trajectory["_segment_minima_render_2d"].get("se2_anchors", [])
    elif "_segment_minima_swg" in global_trajectory:
        anchors_for_viz = global_trajectory["_segment_minima_swg"].get("se2_anchors", [])
    elif "_tail_minima_se2" in global_trajectory:
        anchors_for_viz = [global_trajectory["_tail_minima_se2"].get("tail_anchor", {})]

    correction_anchor_records = list(anchors_for_viz)
    if "_segment_pnp_3d" in global_trajectory:
        correction_anchor_records.extend(global_trajectory["_segment_pnp_3d"].get("anchors", []))
    if "_segment_minima_pnp_3d" in global_trajectory:
        correction_anchor_records.extend(global_trajectory["_segment_minima_pnp_3d"].get("anchors", []))
    if "_segment_minima_render_2d" in global_trajectory:
        correction_anchor_records.extend(global_trajectory["_segment_minima_render_2d"].get("anchors", []))
    if "_segment_minima_swg" in global_trajectory:
        correction_anchor_records.extend(global_trajectory["_segment_minima_swg"].get("anchors", []))
    if "_segment_satellite_pnp" in global_trajectory:
        correction_anchor_records.extend(global_trajectory["_segment_satellite_pnp"].get("anchors", []))

    correction_frame_labels = {}
    for anchor_idx, anchor in enumerate(correction_anchor_records):
        if not anchor.get("ok"):
            continue
        anchor_frame = anchor.get("frame")
        if not anchor_frame:
            continue
        short_name = _csv_frame_stem(anchor_frame)
        short_suffix = short_name[-5:] if len(short_name) >= 5 else short_name
        label = f"C{anchor_idx} {short_suffix}"
        correction_frame_labels[str(anchor_frame)] = label
        correction_frame_labels[short_name] = label

    for anchor in anchors_for_viz:
        _overlay_minima_pointcloud_on_crop(
            ref_crop,
            anchor,
            new_ref_initialX,
            new_ref_initialY,
            ref_res,
            alpha=0.70,
        )

    draw_pointcloud_prior_bounds = False
    if draw_pointcloud_prior_bounds and "_ortho_polygon_gt" in global_trajectory:
        gt_poly = global_trajectory["_ortho_polygon_gt"]
        gt_poly_px = []
        for (px, py) in gt_poly:
            cx = int((px - new_ref_initialX) / ref_res)
            cy = int((new_ref_initialY - py) / ref_res)
            gt_poly_px.append((cx, cy))
        gt_pts = np.array(gt_poly_px, np.int32).reshape((-1, 1, 2))
        cv2.polylines(ref_crop, [gt_pts], isClosed=True, color=(0, 200, 0), thickness=3, lineType=cv2.LINE_AA) # 绿色真值边界框
    
    if draw_pointcloud_prior_bounds and "_ortho_polygon_pred" in global_trajectory:
        pred_poly = global_trajectory["_ortho_polygon_pred"]
        pred_poly_px = []
        for (px, py) in pred_poly:
            cx = int((px - new_ref_initialX) / ref_res)
            cy = int((new_ref_initialY - py) / ref_res)
            pred_poly_px.append((cx, cy))
        pred_pts = np.array(pred_poly_px, np.int32).reshape((-1, 1, 2))
        cv2.polylines(ref_crop, [pred_pts], isClosed=True, color=(0, 0, 200), thickness=3, lineType=cv2.LINE_AA) # 红色预测边界框
        
        # 标出点云预测位置文字
        cv2.putText(ref_crop, "Point Cloud Pred Bounds", (pred_poly_px[0][0], max(0, pred_poly_px[0][1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 200), 3, cv2.LINE_AA)

    draw_pointcloud_center_markers = False
    if draw_pointcloud_center_markers and "_ortho_polygon_pred" in global_trajectory and "_ortho_polygon_gt" in global_trajectory:
        pred_poly = np.array(global_trajectory["_ortho_polygon_pred"], dtype=np.float64)
        gt_poly = np.array(global_trajectory["_ortho_polygon_gt"], dtype=np.float64)
        pred_center_utm = pred_poly.mean(axis=0)
        gt_center_utm = gt_poly.mean(axis=0)
        pc_err_m = float(np.linalg.norm(pred_center_utm - gt_center_utm))
        pc_eval = global_trajectory.get("_pointcloud_eval", {})
        pc_gt_source = pc_eval.get("gt_source", "unknown")
        gt_label = "PC GT center (CSV)" if pc_gt_source == "sequence_csv" else "PC prior center"

        pred_center_px = (
            int((pred_center_utm[0] - new_ref_initialX) / ref_res),
            int((new_ref_initialY - pred_center_utm[1]) / ref_res)
        )
        gt_center_px = (
            int((gt_center_utm[0] - new_ref_initialX) / ref_res),
            int((new_ref_initialY - gt_center_utm[1]) / ref_res)
        )

        cv2.line(ref_crop, gt_center_px, pred_center_px, (0, 255, 255), 6, lineType=cv2.LINE_AA)
        cv2.drawMarker(ref_crop, gt_center_px, color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=45, thickness=5)
        cv2.drawMarker(ref_crop, pred_center_px, color=(0, 0, 255), markerType=cv2.MARKER_DIAMOND, markerSize=45, thickness=5)
        _draw_labeled_marker(ref_crop, gt_center_px, (0, 255, 0), gt_label, marker_type=cv2.MARKER_CROSS, marker_size=45)
        _draw_labeled_marker(ref_crop, pred_center_px, (0, 0, 255), "PC recovered center", marker_type=cv2.MARKER_DIAMOND, marker_size=45)

        matched_pc_utm_for_error = None
        for anchor in anchors_for_viz:
            if anchor.get("ok") and anchor.get("matched_pointcloud_center_utm") is not None:
                matched_pc_utm_for_error = np.array(anchor["matched_pointcloud_center_utm"], dtype=np.float64)
                break
        if matched_pc_utm_for_error is not None:
            matched_pc_px_for_error = _utm_to_crop_px(matched_pc_utm_for_error, new_ref_initialX, new_ref_initialY, ref_res)
            minima_pc_err_m = float(np.linalg.norm(matched_pc_utm_for_error - gt_center_utm))
            cv2.line(ref_crop, gt_center_px, matched_pc_px_for_error, (255, 255, 0), 5, lineType=cv2.LINE_AA)
            _draw_labeled_marker(
                ref_crop,
                matched_pc_px_for_error,
                (255, 255, 0),
                "MINIMA matched PC center",
                marker_type=cv2.MARKER_STAR,
                marker_size=40,
            )
            pc_label = f"MINIMA PC vs GT: {minima_pc_err_m:.2f} m"
        else:
            pc_label = f"Recovered PC vs GT: {pc_err_m:.2f} m"

        label_x = min(max(pred_center_px[0] + 20, 20), max(20, ref_crop.shape[1] - 760))
        label_y = min(max(pred_center_px[1] - 35, 60), max(60, ref_crop.shape[0] - 80))
        cv2.putText(ref_crop, pc_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(ref_crop, pc_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)

    # 画线
    for anchor_idx, anchor in enumerate(anchors_for_viz):
        if not anchor.get("ok"):
            continue
        matched_pc_utm = anchor.get("matched_pointcloud_center_utm")
        direct_anchor_utm = anchor.get("direct_homography_anchor_utm")
        if matched_pc_utm is not None:
            matched_pc_px = _utm_to_crop_px(matched_pc_utm, new_ref_initialX, new_ref_initialY, ref_res)
            _draw_labeled_marker(
                ref_crop,
                matched_pc_px,
                (255, 255, 0),
                "MINIMA PC center",
                marker_type=cv2.MARKER_STAR,
                marker_size=36,
            )
        if False and direct_anchor_utm is not None:
            direct_anchor_px = _utm_to_crop_px(direct_anchor_utm, new_ref_initialX, new_ref_initialY, ref_res)
            _draw_labeled_marker(
                ref_crop,
                direct_anchor_px,
                (255, 0, 128),
                f"Direct tail anchor C{anchor_idx}",
                marker_type=cv2.MARKER_TILTED_CROSS,
                marker_size=30,
            )

    if has_segment_se2:
        if "_segment_minima_se2" in global_trajectory:
            anchors = global_trajectory["_segment_minima_se2"].get("anchors", [])
        elif "_segment_minima_pnp_3d" in global_trajectory:
            anchors = global_trajectory["_segment_minima_pnp_3d"].get("anchors", [])
        elif "_segment_minima_render_2d" in global_trajectory:
            anchors = global_trajectory["_segment_minima_render_2d"].get("anchors", [])
        elif "_segment_minima_swg" in global_trajectory:
            anchors = global_trajectory["_segment_minima_swg"].get("anchors", [])
        elif "_segment_satellite_pnp" in global_trajectory:
            anchors = global_trajectory["_segment_satellite_pnp"].get("anchors", [])
        elif "_segment_pnp_3d" in global_trajectory:
            anchors = global_trajectory["_segment_pnp_3d"].get("anchors", [])
        else:
            anchors = [global_trajectory["_tail_minima_se2"].get("tail_anchor", {})]
        for anchor_idx, anchor in enumerate(anchors):
            if not anchor.get("ok"):
                continue
            pred_utm = anchor.get("pred_utm")
            corr_utm = anchor.get("corrected_utm")
            if pred_utm is None or corr_utm is None:
                continue
            pred_px = (
                int((float(pred_utm[0]) - new_ref_initialX) / ref_res),
                int((new_ref_initialY - float(pred_utm[1])) / ref_res),
            )
            corr_px = (
                int((float(corr_utm[0]) - new_ref_initialX) / ref_res),
                int((new_ref_initialY - float(corr_utm[1])) / ref_res),
            )
            color = (255, 0, 255)
            cv2.line(ref_crop, pred_px, corr_px, color, 4, cv2.LINE_AA)
            cv2.drawMarker(ref_crop, pred_px, (0, 128, 255), cv2.MARKER_TILTED_CROSS, 28, 4, cv2.LINE_AA)
            cv2.drawMarker(ref_crop, corr_px, color, cv2.MARKER_DIAMOND, 30, 4, cv2.LINE_AA)
            label = (
                f"C{anchor_idx} {anchor.get('frame', '')} "
                f"inl={anchor.get('inlier_count', 0)} "
                f"dY={anchor.get('yaw_delta_deg', 0.0):.1f}"
            )
            label_x = min(max(corr_px[0] + 10, 10), max(10, ref_crop.shape[1] - 520))
            label_y = min(max(corr_px[1] - 10, 35), max(35, ref_crop.shape[0] - 20))
            cv2.putText(ref_crop, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 4, cv2.LINE_AA)
            cv2.putText(ref_crop, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    if draw_trajectory_lines and len(gt_points) > 1:
        pts_gt = np.array(gt_points, np.int32).reshape((-1, 1, 2))
        cv2.polylines(ref_crop, [pts_gt], isClosed=False, color=(0, 255, 0), thickness=6) # 绿线GT
    
    if draw_trajectory_lines and len(pred_points) > 1:
        if color_pred_by_segment:
            # Debug view: color prediction by correction segment.
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap("tab10")
            for i in range(len(pred_points) - 1):
                if i < stream_window - 1:
                    chunk_id = 0
                else:
                    chunk_id = 1 + (i - (stream_window - 1)) // stream_stride
                color_rgba = cmap(chunk_id % 10)
                color_bgr = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))
                cv2.line(ref_crop, pred_points[i], pred_points[i+1], color_bgr, 6, cv2.LINE_AA)
                cv2.circle(ref_crop, pred_points[i], 4, color_bgr, -1, cv2.LINE_AA)
            if len(pred_points) > 0:
                color_rgba = cmap(((len(pred_points) - 1 - (stream_window - 1)) // stream_stride + 1) % 10 if len(pred_points) > stream_window else 0)
                color_bgr = (int(color_rgba[2]*255), int(color_rgba[1]*255), int(color_rgba[0]*255))
                cv2.circle(ref_crop, pred_points[-1], 4, color_bgr, -1, cv2.LINE_AA)
        else:
            pts_pred = np.array(pred_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(ref_crop, [pts_pred], isClosed=False, color=(0, 0, 255), thickness=6)
            for pt in pred_points:
                cv2.circle(ref_crop, pt, 4, (0, 0, 255), -1, cv2.LINE_AA)
            
    # 画偏航角箭头 (Yaw Arrow)
    uncorrected_points = []
    if draw_uncorrected_trajectory_points and uncorrected_trajectory is not None:
        uncorrected_keys = sorted([k for k in uncorrected_trajectory.keys() if not str(k).startswith("_")])
        for name_key in uncorrected_keys:
            pred_entry = uncorrected_trajectory.get(name_key, {})
            pred_utm_t = pred_entry.get("Translation_UTM")
            if pred_utm_t is None or len(pred_utm_t) < 2:
                continue
            u_px = int((float(pred_utm_t[0]) - new_ref_initialX) / ref_res)
            u_py = int((new_ref_initialY - float(pred_utm_t[1])) / ref_res)
            uncorrected_points.append((u_px, u_py))

        if draw_trajectory_lines and len(uncorrected_points) > 1:
            pts_uncorrected = np.array(uncorrected_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(ref_crop, [pts_uncorrected], isClosed=False, color=(255, 0, 0), thickness=4, lineType=cv2.LINE_AA)
        for pt in uncorrected_points:
            cv2.circle(ref_crop, pt, 9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(ref_crop, pt, 6, (255, 0, 0), 2, cv2.LINE_AA)

    # Mark which predicted trajectory frame was used as each correction anchor.
    for pair in matched_pairs:
        pred_key = str(pair["pred_key"])
        anchor_label = correction_frame_labels.get(pred_key)
        if anchor_label is None:
            anchor_label = correction_frame_labels.get(_csv_frame_stem(pred_key))
        if anchor_label is None:
            continue

        p_px, p_py = pair["pred_pt"]
        p_px, p_py = int(p_px), int(p_py)
        cv2.circle(ref_crop, (p_px, p_py), 15, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.drawMarker(
            ref_crop,
            (p_px, p_py),
            (255, 128, 0),
            cv2.MARKER_STAR,
            44,
            5,
            cv2.LINE_AA,
        )
        label_x = min(max(p_px + 14, 10), max(10, ref_crop.shape[1] - 190))
        label_y = min(max(p_py - 14, 35), max(35, ref_crop.shape[0] - 15))
        cv2.putText(ref_crop, anchor_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(ref_crop, anchor_label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2, cv2.LINE_AA)

    def draw_yaw_arrow_local(img, cx, cy, yaw_deg, color, length=40):
        import math
        rad = math.radians(yaw_deg)
        # 图像坐标系：向上为北(-Y)， 向右为东(+X)， 偏航角顺时针
        dx = int(length * math.sin(rad))
        dy = int(-length * math.cos(rad))
        cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), color, 3, tipLength=0.3)

    # 取几帧（比如间隔 画几个预测的箭头和 GT 的箭头对比）
    if draw_trajectory_arrows:
        for i, pair in enumerate(matched_pairs):
            gt_px, gt_py = pair["gt_pt"]
            p_px, p_py = pair["pred_pt"]
            
            # 拿到偏航角
            gt_yaw = pair.get("gt_yaw")
            pred_yaw = global_trajectory[pair["pred_key"]]["Yaw_deg"]
            
            if gt_yaw is not None:
                draw_yaw_arrow_local(ref_crop, gt_px, gt_py, gt_yaw, (0, 255, 0), length=30)
            draw_yaw_arrow_local(ref_crop, p_px, p_py, pred_yaw, (0, 0, 255), length=30)

    # 单独打点 + 逐帧误差文字
    for pair in matched_pairs:
        i = pair["display_idx"]
        pt_gt = pair["gt_pt"]
        pt_pred = pair["pred_pt"]
        err_m = pair["err_m"]

        # 连接 GT和Pred，展示位移误差 (黄线段)
        cv2.line(ref_crop, pt_gt, pt_pred, (0, 255, 255), 2, lineType=cv2.LINE_AA)
        
        # 标 GT 点
        if i == 0:
            cv2.drawMarker(ref_crop, pt_gt, color=(0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=25, thickness=4)
        else:
            cv2.circle(ref_crop, pt_gt, radius=5, color=(0, 200, 0), thickness=-1)
            
        # 标 Pred 点
        if i == 0:
            cv2.drawMarker(ref_crop, pt_pred, color=(0, 0, 255), markerType=cv2.MARKER_STAR, markerSize=25, thickness=4)
        else:
            cv2.circle(ref_crop, pt_pred, radius=5, color=(0, 0, 200), thickness=-1)

        if draw_trajectory_arrows:
            # 这里读取当前帧的 Pred 偏航角，并绘制红色箭头
            name_key = pair["pred_key"]
            if name_key in global_trajectory and "Yaw_deg" in global_trajectory[name_key]:
                pred_yaw_deg = global_trajectory[name_key]["Yaw_deg"]
                draw_yaw_arrow_local(ref_crop, pt_pred[0], pt_pred[1], pred_yaw_deg, color=(0, 0, 255), length=35)
                
            # 绘制真实偏航角（如果有 df 中的记录）
            true_yaw_deg = pair.get("gt_yaw")
            if true_yaw_deg is not None:
                draw_yaw_arrow_local(ref_crop, pt_gt[0], pt_gt[1], true_yaw_deg, color=(0, 255, 0), length=35)

        # 标注每一帧当前误差（米），放在预测点附近
        label_x = pt_pred[0] + 8
        label_y = pt_pred[1] - 8
        label = f"{pair['label_idx']}: {err_m:.2f}m"
        cv2.putText(ref_crop, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(ref_crop, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    # 图例
    gt_legend = "GT Trajectory (Green)" if draw_trajectory_lines else "GT Points (Green)"
    pred_legend = "Pred Trajectory (Red)" if draw_trajectory_lines else "Pred Points (Red)"
    if not draw_trajectory_arrows:
        pred_legend += ", arrows off"
    cv2.putText(ref_crop, gt_legend, (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(ref_crop, pred_legend, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
    cv2.putText(ref_crop, "Error Offset (Yellow)", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)
    cv2.putText(ref_crop, "Per-frame Error Label: idx: xx.xxm", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(ref_crop, "PC GT(CSV)=green cross, recovered=red diamond, MINIMA=cyan star", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    if draw_uncorrected_trajectory_points and uncorrected_trajectory is not None:
        cv2.putText(ref_crop, "Uncorrected Pred (Blue circles)", (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    if has_segment_se2:
        minima_legend_y = 400 if draw_uncorrected_trajectory_points and uncorrected_trajectory is not None else 350
        cv2.putText(ref_crop, "MINIMA warp: translucent pointcloud overlay; orange=before, magenta=after", (50, minima_legend_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 3)

    out_path = os.path.join(out_dir, "trajectory_compare_utm.jpg")
    clean_out_path = os.path.join(out_dir, "trajectory_compare_utm_clean_satellite.jpg")
    cv2.imwrite(clean_out_path, ref_crop_clean)
    cv2.imwrite(out_path, ref_crop)
    print(f"\n可视化完成！对比图已保存至: {out_path}")


if __name__ == "__main__":
    pipeline_t0 = time.perf_counter()
    seed_everything(GLOBAL_RANDOM_SEED)
    # ========================================================================== 
    # 完整 Pipeline 配置 
    # ========================================================================== 
    # --- 输入数据路径 --- 
    # 每次只改这一个数据集名称即可，下面两个路径会自动拼出来
    # DATASET_NAME = "DJI_20260406225117_0005" # 原始 
    # DATASET_NAME = "DJI_20260520105258_0002" # 直线 
    # DATASET_NAME = "DJI_20260520103322_0008" # 学院 
    # DATASET_NAME = "DJI_20260616143832_0044" # 城市
    DATASET_NAME = "DJI_farm" # 农田
    # DATASET_NAME = "DJI_city" # 城市
     
    
    _DATASET_DIR = os.path.join(r"data\Thermal-UAV-v2\test\changsha", DATASET_NAME)
    INPUT_IMAGE_DIR = os.path.join(_DATASET_DIR, "Thermal")
    CSV_PATH = os.path.join(_DATASET_DIR, "Thermal_info.csv")

    # Scal3R 模型配置
    SCAL3R_CONFIG = "configs/models/scal3r.yaml"

    # --- 全局参数 ---
    REGION_NAME  = "changsha"

    # 无人机首帧先验信息：只允许读取 CSV 第一帧 GT/飞控记录。
    ### 005
    # 从 CSV 第一帧读取无人机先验信息（位置/姿态），相机参数固定写死；不得读取后续帧参与定位。
    _first_row = pd.read_csv(CSV_PATH).iloc[0]
    truePos = {
        "name": str(_first_row["filename"]),
        "lat": float(_first_row["lat"]),
        "lon": float(_first_row["lon"]),
        "rel_height": float(_first_row["rel_height"]),
        "abs_height": float(_first_row["abs_height"]),
        "alt": float(_first_row["rel_height"]),
        "pitch": float(_first_row["pitch"]),
        "roll": float(_first_row["roll"]),
        "yaw": float(_first_row["yaw"]),
        "cam_size": 9.83,
        "focal_len": 12.0,
        "width": 1280.0,
        "height": 1024.0,
    }
    # ### 004
    # INPUT_IMAGE_DIR = r"Data\Thermal-UAV\test\changsha\DJI_20260406235250_0004\Thermal"
    # CSV_PATH = r"Data\Thermal-UAV\test\changsha\DJI_20260406235250_0004\Thermal_info.csv"
    # truePos = {
   
    #     "name": "./Data/Thermal-UAV/test/changsha/DJI_20260406235250_0004/Thermal/DJI_20260406235250_0004_T_f0000.JPG",
    #     "lat": 28.254808,
    #     "lon": 113.058161,
    #     "rel_height": 304.321,
    #     "abs_height": 341.262,
    #     "alt": 304.321,
    #     "pitch": -35.2,
    #     "yaw": -116.2,
    #     "roll": 0.0,
    #     "cam_size": 9.83,
    #     "focal_len": 12.0,
    #     "width": 1280.0,
    #     "height": 1024.0,
    # }

    time_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.path.join("Result", f"Global_Pose_Recovery_{time_suffix}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "random_seed.json"), "w", encoding="utf-8") as f:
        json.dump({
            "GLOBAL_RANDOM_SEED": int(GLOBAL_RANDOM_SEED),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "STRICT_TORCH_DETERMINISM": bool(STRICT_TORCH_DETERMINISM),
            "DISABLE_TF32_FOR_REPRODUCIBILITY": bool(DISABLE_TF32_FOR_REPRODUCIBILITY),
            "OPENCV_SINGLE_THREAD_FOR_REPRODUCIBILITY": bool(OPENCV_SINGLE_THREAD_FOR_REPRODUCIBILITY),
        }, f, indent=2, ensure_ascii=False)

    # Scal3R 输出目录（点云、外参文件均在此处）
    # 使用统一的时间戳并全部放入 Result 目录下
    SCAL3R_TAG    = "scal3r_output"
    SCAL3R_OUTPUT = os.path.join(SAVE_DIR, SCAL3R_TAG)
    os.makedirs(SCAL3R_OUTPUT, exist_ok=True)
    PLY_PATH      = os.path.join(SCAL3R_OUTPUT, "points", "whole.ply")
    ORTHO_PATH    = os.path.join(SAVE_DIR, "orthomosaic.png")

    print("=========================== 完整 Pipeline 启动 ===========================")

    # ------------------------------------------------------------------
    # 公共配置加载（只读一次大图缓存）
    # ------------------------------------------------------------------
    class Opt:
        save_dir     = SAVE_DIR
        device       = 'cuda'
        pose_priori  = 'yp'
        Ref_type     = 'HIGH'
        resize_ratio = 1.0
        crop_size_m  = 1200
        crop_gain    = 1

    opt = Opt()

    region_path = f"./data/Regions_params/{REGION_NAME}.yaml"
    with open(region_path, 'r', encoding='utf-8') as f:
        region_config = yaml.safe_load(f)

    print("正在读取底层大图缓存（卫星底图 + DSM 高程图），仅加载一次...")
    ref_map_full, dsm_map_full, _, _ = load_config_parameters_new(region_config, opt, REGION_NAME)

    # ==================================================================
    # STEP 0: Scal3R 三维重建
    # ==================================================================
    print("\n▶ [STEP 0] 调用 Scal3R 进行三维重建与相对位姿估计...")
    from os.path import abspath, expanduser
    from scal3r.engine import load_config as scal3r_load_config
    from scal3r.engine.path import resolve_release_path
    from scal3r.pipelines.inference import InferenceRequest, run_inference

    # Scal3R only runs once on the full sequence.  The MINIMA/RoMa correction
    # cadence is controlled separately by this interval.
    CORRECTION_INTERVAL_FRAMES = 20
    ANCHOR_LOCAL_PATCH_RADIUS_M = 240
    MAX_ANCHOR_INNOVATION_M = 60.0
    MAX_ANCHOR_YAW_JUMP_DEG = 15.0
    MAX_FIRST_ANCHOR_DELTA_M = 120.0
    ANCHOR_GEOMETRY_GATE_ENABLED = False # default false
    ANCHOR_MATCH_QUALITY_GATE_ENABLED = True
    ANCHOR_MIN_INLIER_RATIO = 0.08
    ANCHOR_HOMOGRAPHY_SCALE_MIN = 0.50
    ANCHOR_HOMOGRAPHY_SCALE_MAX = 2.00
    ANCHOR_MAX_HOMOGRAPHY_CONDITION = 80.0
    ANCHOR_MAX_HOMOGRAPHY_PERSPECTIVE_NORM = 0.01
    ANCHOR_MAX_HOMOGRAPHY_RMSE_PX = 10.0
    ANCHOR_MIN_LOCAL_VALID_RATIO = 0.05
    SCAL3R_STREAMING_STATE = 0
    SCAL3R_OVERLAP_SIZE = 3
    COLOR_PRED_BY_CORRECTION_SEGMENT = False
    DRAW_GT_PRED_TRAJECTORY_LINES = False
    DRAW_GT_PRED_TRAJECTORY_ARROWS = False
    DRAW_UNCORRECTED_TRAJECTORY_POINTS = True
    POSE_GRAPH_ODOM_SIGMA_M = 1.0
    POSE_GRAPH_ANCHOR_SIGMA_M = 6.0
    STREAM_WINDOW = CORRECTION_INTERVAL_FRAMES
    STREAM_STRIDE = CORRECTION_INTERVAL_FRAMES
    if CORRECTION_INTERVAL_FRAMES < 1:
        raise ValueError("CORRECTION_INTERVAL_FRAMES 必须 >= 1。")

    print("\n[INFO] 将整段序列一次性送入 Scal3R；后续按固定帧间隔做 MINIMA/RoMa 校正...")
    all_frames = sorted(list(set(
        glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.JPG")) +
        glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.png")) +
        glob.glob(os.path.join(INPUT_IMAGE_DIR, "*.jpg"))
    )))
    total_frames = len(all_frames)
    print(f"总计输入 {total_frames} 帧图像。")
    if total_frames == 0:
        raise FileNotFoundError(
            f"未在 INPUT_IMAGE_DIR 找到输入图像: {INPUT_IMAGE_DIR}\n"
            "请检查 DATASET_NAME、INPUT_IMAGE_DIR，或图像后缀是否为 JPG/png/jpg。"
        )

    # SCAL3R_BLOCK_SIZE = max(10, total_frames)
    SCAL3R_BLOCK_SIZE = 30
    print(
        f"[INFO] Scal3R internal streaming_state={SCAL3R_STREAMING_STATE}, "
        f"block_size={SCAL3R_BLOCK_SIZE}, overlap_size={SCAL3R_OVERLAP_SIZE}"
    )
    scal3r_cfg = scal3r_load_config(SCAL3R_CONFIG)
    scal3r_req = InferenceRequest(
        config_path = resolve_release_path(SCAL3R_CONFIG),
        input_dir = abspath(expanduser(INPUT_IMAGE_DIR)),
        output_dir = abspath(SCAL3R_OUTPUT),
        runtime_dir = abspath(os.path.join(SCAL3R_OUTPUT, "runtime")),
        save_xyz = 1, save_dpt = 0, use_loop = 0,
        block_size = SCAL3R_BLOCK_SIZE, overlap_size = SCAL3R_OVERLAP_SIZE,
        preprocess_workers = 2, pgo_workers = 2,
        max_align_points_per_frame = 512,
        streaming_state = SCAL3R_STREAMING_STATE,
        offload_outputs = 1
    )
    run_inference(scal3r_cfg, scal3r_req)

    unified_extri_path = os.path.join(SCAL3R_OUTPUT, "extri.yml")
    if not os.path.exists(unified_extri_path):
        raise FileNotFoundError(f"Scal3R 未生成外参文件: {unified_extri_path}")

    # Normalize Scal3R numeric camera keys (0, 1, ...) to real frame stems so
    # later CSV/trajectory/anchor code can address frames consistently.
    frame_stems = [os.path.splitext(os.path.basename(f))[0] for f in all_frames]
    _, raw_poses_temp = load_scal3r_poses(unified_extri_path)
    normalized_poses = {}
    for cam_name, pose in raw_poses_temp.items():
        safe_name = None
        try:
            idx = int(cam_name)
            if 0 <= idx < len(frame_stems):
                safe_name = frame_stems[idx]
        except (TypeError, ValueError):
            cam_base = os.path.splitext(os.path.basename(str(cam_name)))[0]
            if cam_base in frame_stems:
                safe_name = cam_base
        if safe_name is not None:
            normalized_poses[safe_name] = pose

    if not normalized_poses:
        raise RuntimeError(f"无法把 Scal3R 外参帧名映射到输入图像帧名: {unified_extri_path}")
    if len(normalized_poses) != len(raw_poses_temp):
        print(f"[WARN] Scal3R 外参映射数量 {len(normalized_poses)}/{len(raw_poses_temp)}，请检查是否有帧未成功重建。")

    ordered_pose_names = [name for name in frame_stems if name in normalized_poses]
    extra_pose_names = [name for name in normalized_poses.keys() if name not in ordered_pose_names]
    fs = cv2.FileStorage(unified_extri_path, cv2.FILE_STORAGE_WRITE)
    fs.write("names", ordered_pose_names + extra_pose_names)
    for name in ordered_pose_names + extra_pose_names:
        pose = normalized_poses[name]
        fs.write(f"Rot_{name}", pose["R"])
        fs.write(f"T_{name}", pose["T"])
    fs.release()

    streaming_chunk_records = []
    for chunk_idx, start_idx in enumerate(range(0, len(ordered_pose_names), CORRECTION_INTERVAL_FRAMES)):
        end_idx = min(start_idx + CORRECTION_INTERVAL_FRAMES, len(ordered_pose_names))
        segment_frame_names = ordered_pose_names[start_idx:end_idx]
        if not segment_frame_names:
            continue
        streaming_chunk_records.append({
            "chunk_idx": int(chunk_idx),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "frames": segment_frame_names,
            "owned_frames": segment_frame_names,
            "anchor_frame": segment_frame_names[-1],
            "chunk_dir": SCAL3R_OUTPUT,
            "correction_interval_frames": int(CORRECTION_INTERVAL_FRAMES),
        })
    print(
        f"[INFO] Scal3R 已一次性完成；按每 {CORRECTION_INTERVAL_FRAMES} 帧生成 "
        f"{len(streaming_chunk_records)} 个 MINIMA/RoMa 校正段。"
    )

    if not os.path.exists(PLY_PATH):
        raise FileNotFoundError(
            f"Scal3R 重建完成，但未找到点云文件: {PLY_PATH}\n"
            "请确认 save_xyz=1 已生效，或手动指定 PLY_PATH。"
        )
    print(f"Scal3R 重建完成，点云已保存至: {PLY_PATH}")

    # ==================================================================
    # STEP 1: 生成正射影像
    # ==================================================================
    print("\n▶ [STEP 1] 从重建点云生成正射影像...")
    sat_gsd_m = getattr(opt, "ref_resolution", None)
    if sat_gsd_m is None:
        sat_gsd_m = region_config.get(f'{REGION_NAME}_{opt.Ref_type}_REF_resolution', None)
    print(f"[BEV scale] Target satellite GSD for orthophoto: {sat_gsd_m} m/px")
    generate_orthophoto(
        PLY_PATH, ORTHO_PATH,
        max_resolution=3000,
        up_axis='y', up_direction=-1,
        use_morphology=True, auto_level=True,
        target_gsd_m=sat_gsd_m,
        first_frame_prior=truePos,
        extri_path=os.path.join(SCAL3R_OUTPUT, "extri.yml"),
        interpolate_for_matching=True
    )

    # ==================================================================
    # STEP 2: DINO 模板匹配全局定位
    # ==================================================================
    # Coarse global bootstrap: only the first-frame pose prior is known here.
    # Orthomosaic-to-satellite matching happens later after the tail-frame pose
    # has been predicted from the relative trajectory.
    SKIP_BEV_RETRIEVAL = True
    
    if SKIP_BEV_RETRIEVAL and truePos is not None:
        print("\n▶ [STEP 2] 首帧真实位姿已知，跳过正射投影模板匹配（BEV检索）...")
        match_utm_x, match_utm_y = 0.0, 0.0
        matched_yaw_offset = 0.0
    else:
        print("\n▶ [STEP 2] 运行正射影像模板匹配进行全局地理定位...")
        match_utm_x, match_utm_y, matched_yaw_offset = run_ortho_matching_and_get_coords(
            ortho_path   = ORTHO_PATH,
            save_dir     = SAVE_DIR,
            region       = REGION_NAME,
            opt          = opt,
            region_config= region_config,
            ref_map_full = ref_map_full,
            dsm_map_full = dsm_map_full,
            truePos      = truePos
        )

    # ==================================================================
    # STEP 3: 全局位姿恢复（将相对轨迹对齐到世界坐标系）
    # ==================================================================
    print("\n▶ [STEP 3] 利用匹配锚点恢复完整全局轨迹（非作弊纯物理推算）...")
    uav_res = (truePos['rel_height'] * truePos['cam_size']) / (truePos['focal_len'] * truePos['width'])
    global_traj = recover_global_poses(
        run_dir          = SCAL3R_OUTPUT,
        save_dir         = SAVE_DIR,
        pred_utm_x       = match_utm_x,
        pred_utm_y       = match_utm_y,
        pred_rel_height  = truePos['rel_height'],
        yaw_offset_deg   = matched_yaw_offset,
        true_yaw_prior   = truePos['yaw'],
        uav_res          = uav_res,
        ortho_meta_path  = os.path.splitext(ORTHO_PATH)[0] + "_meta.json",
        truePos          = truePos,
        region           = REGION_NAME,
        region_config    = region_config,
        dsm_map_full     = dsm_map_full,
        first_frame_known= SKIP_BEV_RETRIEVAL,
        gt_csv_path      = None
    )
    global_traj_uncorrected = json.loads(json.dumps(global_traj))
    uncorrected_runtime_sec = time.perf_counter() - pipeline_t0

    # This standalone pipeline intentionally uses only MINIMA-RoMa SE(2) correction.
    CORRECTION_MODE = "MINIMA_ROMA_SE2_ONLY"
    print(
        f"\n▶ [STEP 4] 使用每 {CORRECTION_INTERVAL_FRAMES} 帧一个 MINIMA_Roma 锚点进行序贯 SE(2) 校正..."
    )
    global_traj, segment_summary = apply_segment_minima_se2_correction(
        global_trajectory       = global_traj,
        chunk_records           = streaming_chunk_records,
        run_dir                 = SCAL3R_OUTPUT,
        save_dir                = SAVE_DIR,
        ortho_path              = ORTHO_PATH,
        true_pos                = truePos,
        csv_path                = None,
        region                  = REGION_NAME,
        opt                     = opt,
        region_config           = region_config,
        ref_map_full            = ref_map_full,
        search_margin_m         = 0,
        min_conf                = 0.0,
        min_inliers             = 12,
        ransac_reproj_threshold = 5.0,
        overlay_alpha           = 0.55,
        matcher_method          = "MINIMA_Roma",
        anchor_patch_radius_m   = ANCHOR_LOCAL_PATCH_RADIUS_M,
        anchor_match_quality_gate_enabled = ANCHOR_MATCH_QUALITY_GATE_ENABLED,
        anchor_min_inlier_ratio = ANCHOR_MIN_INLIER_RATIO,
        anchor_homography_scale_min = ANCHOR_HOMOGRAPHY_SCALE_MIN,
        anchor_homography_scale_max = ANCHOR_HOMOGRAPHY_SCALE_MAX,
        anchor_max_homography_condition = ANCHOR_MAX_HOMOGRAPHY_CONDITION,
        anchor_max_homography_perspective_norm = ANCHOR_MAX_HOMOGRAPHY_PERSPECTIVE_NORM,
        anchor_max_homography_rmse_px = ANCHOR_MAX_HOMOGRAPHY_RMSE_PX,
        anchor_min_local_valid_ratio = ANCHOR_MIN_LOCAL_VALID_RATIO,
        max_anchor_innovation_m = MAX_ANCHOR_INNOVATION_M,
        max_anchor_yaw_jump_deg = MAX_ANCHOR_YAW_JUMP_DEG,
        max_first_anchor_delta_m = MAX_FIRST_ANCHOR_DELTA_M,
        anchor_geometry_gate_enabled = ANCHOR_GEOMETRY_GATE_ENABLED,
        odom_constraint_sigma_m = POSE_GRAPH_ODOM_SIGMA_M,
        anchor_constraint_sigma_m = POSE_GRAPH_ANCHOR_SIGMA_M,
    )
    print(
        f"-> [SEGMENT-MINIMA-SE2] anchors={segment_summary['anchor_count']}, "
        f"applied={segment_summary['applied_count']}, "
        f"summary={os.path.join(SAVE_DIR, 'segment_minima_se2_summary.json')}"
    )
    print("\n▶ [STEP 5] 使用分段 SE(2) 校正后的轨迹进行GT误差可视化...")
    visualize_comparison_on_map(
        global_trajectory = global_traj,
        csv_path          = CSV_PATH,
        region            = REGION_NAME,
        opt               = opt,
        region_config     = region_config,
        ref_map_full      = ref_map_full,
        out_dir           = SAVE_DIR,
        crop_size_m       = 2500,
        stream_window     = STREAM_WINDOW,
        stream_stride     = STREAM_STRIDE,
        color_pred_by_segment = COLOR_PRED_BY_CORRECTION_SEGMENT,
        draw_trajectory_lines = DRAW_GT_PRED_TRAJECTORY_LINES,
        draw_trajectory_arrows = DRAW_GT_PRED_TRAJECTORY_ARROWS,
        uncorrected_trajectory = global_traj_uncorrected,
        draw_uncorrected_trajectory_points = DRAW_UNCORRECTED_TRAJECTORY_POINTS,
    )
    print(f"\n✅ 完整 Pipeline 执行完毕！所有结果已保存至: {SAVE_DIR}")
    print(f"   - Scal3R 重建输出  : {SCAL3R_OUTPUT}")
    print(f"   - 正射影像          : {ORTHO_PATH}")
    print(f"   - 全局轨迹 JSON     : {os.path.join(SAVE_DIR, 'global_poses_utm.json')}")
    print(f"   - 分段校正摘要      : {os.path.join(SAVE_DIR, 'segment_minima_se2_summary.json')}")
    print(f"   - 轨迹对比可视化    : {os.path.join(SAVE_DIR, 'trajectory_compare_utm.jpg')}")
    metrics_path = os.path.join(SAVE_DIR, "trajectory_metrics.pkl")
    uncorrected_metrics_path = os.path.join(SAVE_DIR, "trajectory_metrics_uncorrected.pkl")
    compute_and_save_trajectory_metrics(
        global_trajectory = global_traj_uncorrected,
        csv_path          = CSV_PATH,
        region            = REGION_NAME,
        region_config     = region_config,
        out_dir           = SAVE_DIR,
        runtime_sec       = uncorrected_runtime_sec,
        output_name       = "trajectory_metrics_uncorrected.pkl",
        label             = "uncorrected_no_anchor",
    )
    compute_and_save_trajectory_metrics(
        global_trajectory = global_traj,
        csv_path          = CSV_PATH,
        region            = REGION_NAME,
        region_config     = region_config,
        out_dir           = SAVE_DIR,
        runtime_sec       = time.perf_counter() - pipeline_t0,
        output_name       = "trajectory_metrics.pkl",
        label             = "corrected_with_anchor",
    )
    pose_attitude_csv_path = export_pose_attitude_comparison_csv(
        uncorrected_trajectory = global_traj_uncorrected,
        corrected_trajectory   = global_traj,
        csv_path               = CSV_PATH,
        region                 = REGION_NAME,
        region_config          = region_config,
        out_dir                = SAVE_DIR,
    )
    print(f"   - Trajectory metrics   : {metrics_path}")
    print(f"   - Uncorrected metrics  : {uncorrected_metrics_path}")
    print(f"   - Pose attitude CSV    : {pose_attitude_csv_path}")
    raise SystemExit(0)
