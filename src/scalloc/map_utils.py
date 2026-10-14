from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from pyproj import Transformer
import shutil
import glob
import pickle
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import math
import json
import os
import random
import time
import cv2
import numpy as np
from datetime import datetime
from math import sqrt
from PIL import Image
import tifffile
from scipy.optimize import least_squares
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist

def deg2utm(region, config, lon, lat):
    utm_system = config[f'{region}_UTM_SYSTEM']  # e.g., '49N'
    zone = int(utm_system[:-1])
    is_northern = utm_system[-1].upper() == 'N'
    epsg_code = 32600 + zone if is_northern else 32700 + zone
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg_code}", always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    return utm_x, utm_y

def utm2deg(region, config, utm_x, utm_y):
    utm_system = config[f'{region}_UTM_SYSTEM']
    zone = int(utm_system[:-1])
    is_northern = utm_system[-1].upper() == 'N'
    epsg_code = 32600 + zone if is_northern else 32700 + zone
    transformer = Transformer.from_crs(f"EPSG:{epsg_code}", "EPSG:4326", always_xy=True)
    lon, lat = transformer.transform(utm_x, utm_y)
    return lon, lat

import tifffile

def load_config_parameters_new(region_config, opt, region, load_images=True):
    Ref_type = getattr(opt, 'Ref_type', 'satellite')
    dsm_resolution = float(region_config.get(f'{region}_{Ref_type}_DSM_resolution', 0.3))
    ref_resolution = float(region_config.get(f'{region}_{Ref_type}_REF_resolution', 0.3))
    
    ref_image = None
    dsm_image = None
    
    if load_images:
        ref_path = region_config.get(f"{region}_{Ref_type}_REF_PATH")
        if ref_path and os.path.exists(ref_path):
            t0 = time.time()
            img = tifffile.imread(ref_path)
            ref_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 读取真实的物理分辨率 (ModelPixelScaleTag) 并更新
            try:
                with tifffile.TiffFile(ref_path) as tif:
                    tags = tif.pages[0].tags
                    if 'ModelPixelScaleTag' in tags:
                        scale = tags['ModelPixelScaleTag'].value
                        if len(scale) >= 2:
                            real_res_x = scale[0]
                            real_res_y = scale[1]
                            ref_resolution = (real_res_x + real_res_y) / 2.0
                            print(f"[读取TIF标签] 自动获得卫星图实际分辨率: X={real_res_x:.4f}, Y={real_res_y:.4f}, 平均={ref_resolution:.4f} m/px")
            except Exception as e:
                print(f"[警告] 无法读取卫星图分辨率标签: {e}")
                
            t1 = time.time()
            print(f"[timing] Read satellite map : {(t1 - t0) // 60:.0f}m {(t1 - t0) % 60:.1f}s")
        else:
            print("Satellite map path not found or missing:", ref_path)
            
    return ref_image, dsm_image, dsm_resolution, ref_resolution
    
import time
import cv2
from math import sqrt
import numpy as np
from PIL import Image
import tifffile
from scipy.optimize import least_squares
# 在 utils.py 顶部添加
from scipy.spatial import Voronoi, Delaunay
from scipy.spatial.distance import cdist

# === [新增] 角度归一化 ===

# ==========================================
# 流式 SLAM : Sim(3) 轨迹对齐与拼图核心工具
# ==========================================

def umeyama_alignment(source, target, estimate_scale=True):
    """
    使用 Umeyama 算法计算三维点集的 Sim(3) 相似变换矩阵。
    目标：寻找尺度 s、旋转 R、平移 t，使得 target 约等于 s * (source @ R.T) + t
    source: (N, 3) 的原始点集 (Mk 中的坐标)
    target: (N, 3) 的目标点集 (M0 中的坐标)
    """
    assert source.shape == target.shape, "Source and target point clouds must have the same shape"
    assert source.shape[0] >= 3, "At least 3 points are required for Sim(3) transformation"

    mean_src = np.mean(source, axis=0)
    mean_tar = np.mean(target, axis=0)

    # 去质心
    src_centered = source - mean_src
    tar_centered = target - mean_tar

    # 协方差矩阵
    K = src_centered.T @ tar_centered / source.shape[0]

    # SVD 分解
    U, D, V_T = np.linalg.svd(K)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        S[2, 2] = -1

    # 计算旋转矩阵 R
    R = V_T.T @ S @ U.T

    # 计算尺度 s
    if estimate_scale:
        var_src = np.var(source, axis=0).sum()
        s = np.trace(np.diag(D) @ S) / var_src
    else:
        s = 1.0

    # 计算平移向量 t
    t = mean_tar - s * (R @ mean_src)

    return s, R, t

def compute_Mk_to_M0_transform(mk_poses, m0_poses, overlap_names):
    """
    利用两个局部位姿字典中共有的重叠帧，使用四元数均值及多点平均计算 Mk 到 M0 的刚体/相似变换。
    这种基于相机姿态 R 的直接求解比仅使用光心 Umeyama 更加鲁棒，可避免共线飞行时的旋转退化（Roll 退化）。
    返回: s, R_align, t_align
    """
    from scipy.spatial.transform import Rotation
    
    src_centers = []
    tar_centers = []
    quats = []
    
    for name in overlap_names:
        # P_cam = R P_world + T
        R_mk = mk_poses[name]["R"]
        T_mk = mk_poses[name]["T"]
        R_m0 = m0_poses[name]["R"]
        T_m0 = m0_poses[name]["T"]
        
        c_mk = -R_mk.T @ T_mk
        c_m0 = -R_m0.T @ T_m0
        src_centers.append(c_mk)
        tar_centers.append(c_m0)
        
        # Rs_align = R_m0^T @ R_mk
        R_a_curr = R_m0.T @ R_mk
        quats.append(Rotation.from_matrix(R_a_curr).as_quat())
        
    src_centers = np.array(src_centers)
    tar_centers = np.array(tar_centers)
    quats = np.array(quats)
    
    # 1. 求解 R_align：利用姿态的四元数均值避免直线飞行的解算退化
    # 注意：同一个旋转有两种四元数表示 (q 和 -q)，在求均值前必须统一符号
    q0 = quats[0]
    for i in range(1, len(quats)):
        if np.dot(q0, quats[i]) < 0:
            quats[i] = -quats[i]
            
    mean_q = np.mean(quats, axis=0)
    mean_q /= np.linalg.norm(mean_q)
    R_align = Rotation.from_quat(mean_q).as_matrix()
    
    # 2. 求解 Scale：
    # Scal3R 名字即代表 Scale Recovery，其网络已经输出了具备公制尺度 (Metric Scale) 的距离。
    # 如果这里的子块之间有微小的尺度计算误差（因为重叠帧只有 3 个点，基线极短，方差除法极其不稳定），
    # 就会导致映射后的高度放大，使得地面点云出现 Z 方向的“分层”、“梯田”现象。
    # 所以直接强制锁定尺度为 1.0 (保持 Scal3R 局部的严密尺度即可)
    s = 1.0
    
    # 3. 求解 t_align: t_a = c_m0 - s * R_align @ c_mk
    t_aligns = tar_centers - s * (R_align @ src_centers.T).T
    t_align = np.mean(t_aligns, axis=0)
    
    return s, R_align, t_align

def apply_sim3_transform_to_poses(mk_poses, s, R_align, t_align):
    """
    利用计算好的 Sim(3) 映射将 Mk 的每一帧生位姿对齐进大基座 M0。
    """
    m0_poses_new = {}
    for name, pose in mk_poses.items():
        R_mk = pose["R"]
        T_mk = pose["T"]
        
        # 旋转需左乘新参考系的相对旋转逆（因为它是对世界进行映射被相机观测）
        R_new = R_mk @ R_align.T
        T_new = s * T_mk - R_new @ t_align
        
        m0_poses_new[name] = {"R": R_new, "T": T_new}
    return m0_poses_new

def project_point_cloud_sim3(points, s, R_align, t_align):
    """
    将 Mk 的生点云同样投射进 M0。
    """
    return s * (R_align @ points.T).T + t_align

def normalize_angle(angle):
    """将角度归一化到 [-180, 180]"""
    angle = (angle + 180) % 360 - 180
    return angle


def img_name(sensing_path):
    file_name_with_ext = os.path.basename(sensing_path)
    file_name_without_ext, _ = os.path.splitext(file_name_with_ext)
    return file_name_without_ext


def load_data(save_path, *args):
    with open(save_path, 'rb') as file:
        data = pickle.load(file)
    results = tuple(data[arg] for arg in args if arg in data)

    return results

def save_data(filename, **kwargs):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = kwargs
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def get_jpg_files(folder_path):
    # Ensure the input path exists
    if not os.path.exists(folder_path):
        print("Path does not exist. Please check your input!")
        return []

    # If the path is a file rather than a folder, return an empty list
    if not os.path.isdir(folder_path):
        print("The input is a file path, not a folder path!")
        return []

    # Use the glob module to search for all JPG files, including uppercase and lowercase extensions
    jpg_files = glob.glob(os.path.join(folder_path, '*.JPG'))

    return jpg_files


def copy_image(src_path, dest_path, new_name):
    dest_file_path = os.path.join(dest_path, new_name)
    shutil.copy2(src_path, dest_file_path)




def process_and_save_matches(Sen_pts, Ref_pts, inliers, uav_image, ref_image, save_path):
    # 1. 筛选内点 (兼容不同格式)
    if len(inliers) > 0:
        if isinstance(inliers[0], (list, np.ndarray, tuple)):
            Sen_pts_temp = [Sen_pts[i[0]] for i in inliers]
            Ref_pts_temp = [RefPts[i[0]] for i in inliers]
        else:
            Sen_pts_temp = [SenPts[i] for i in inliers]
            Ref_pts_temp = [RefPts[i] for i in inliers]
    else:
        Sen_pts_temp = []
        Ref_pts_temp = []

    # 2. 获取两张图的原始尺寸
    h1, w1 = uav_image.shape[:2]
    h2, w2 = ref_image.shape[:2]
    
    # --- 设置中间留白的宽度 (像素) ---
    gap = 50 
    
    # 画板高度取两者最大值，宽度为两者之和加上间距
    max_h = max(h1, h2)
    total_w = w1 + gap + w2
    
    # 【改动】创建一个全透明的背景画板 (4通道 BGRA，初始化为全 0)
    vis_img = np.zeros((max_h, total_w, 4), dtype=np.uint8)
    
    # 【改动】将原图由 3 通道 (BGR) 转换为 4 通道 (BGRA)，Alpha 设为 255 (完全不透明)
    uav_bgra = cv2.cvtColor(uav_image, cv2.COLOR_BGR2BGRA)
    ref_bgra = cv2.cvtColor(ref_image, cv2.COLOR_BGR2BGRA)
    
    # --- 计算垂直居中的 Y 轴偏移量 ---
    # 较高的图偏移量会是 0，较矮的图会被分配一个使其居中的下移量
    offset_y1 = (max_h - h1) // 2
    offset_y2 = (max_h - h2) // 2
    
    # 3. 将 BGRA 原图原封不动地贴到画板上 (垂直居中贴图)
    vis_img[offset_y1 : offset_y1+h1, :w1, :] = uav_bgra
    vis_img[offset_y2 : offset_y2+h2, w1+gap : w1+gap+w2, :] = ref_bgra

    # 4. 自定义绘制连线与特征点
    for pt1, pt2 in zip(Sen_pts_temp, Ref_pts_temp):
        # 左侧点坐标需要加上其专属的垂直偏移量
        pt1_x = int(pt1[0])
        pt1_y = int(pt1[1]) + offset_y1
        
        # 右侧点的 X 坐标需要加上左图宽度和间距
        pt2_x = int(pt2[0]) + w1 + gap
        pt2_y = int(pt2[1]) + offset_y2
        
        # 画线 (绿色，线宽1)
        cv2.line(vis_img, (pt1_x, pt1_y), (pt2_x, pt2_y), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        
        # 画点 (红色，半径2，实心)
        cv2.circle(vis_img, (pt1_x, pt1_y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis_img, (pt2_x, pt2_y), 2, (0, 0, 255), -1, lineType=cv2.LINE_AA)

    # 5. 保存
    cv2.imwrite(save_path, vis_img)


def computeCameraMatrix(truePos):
    imW0 = truePos['width']
    imH0 = truePos['height']
    cameraSize = truePos['cam_size']
    focal = truePos['focal_len']
    pixelSize_x = cameraSize / np.sqrt(imW0**2 + imH0**2)
    pixelSize_y = cameraSize / np.sqrt(imW0**2 + imH0**2)
    focalx = focal / pixelSize_x
    focaly = focal / pixelSize_y
    centerX = imW0 / 2
    centerY = imH0 / 2
    K = np.array([
        [focalx, 0, centerX],
        [0, focaly, centerY],
        [0, 0, 1]
    ])
    return K


def dumpRotateImage(img, degree):
    height, width = img.shape[:2]
    center = (width / 2, height / 2)
    matRotation = cv2.getRotationMatrix2D(center, degree, 1)
    cos_val = np.abs(matRotation[0, 0])
    sin_val = np.abs(matRotation[0, 1])
    new_w = int(height * sin_val + width * cos_val)
    new_h = int(height * cos_val + width * sin_val)
    matRotation[0, 2] += (new_w / 2) - center[0]
    matRotation[1, 2] += (new_h / 2) - center[1]
    
    if len(img.shape) == 2:
        imgRotation = cv2.warpAffine(img, matRotation, (new_w, new_h), borderValue=0)
    else:
        imgRotation = cv2.warpAffine(img, matRotation, (new_w, new_h), borderValue=(0, 0, 0))
    return imgRotation, matRotation

def crop_geo_images_with_offset(region, config, opt, truePos, ref_map_full, dsm_map_full):
    refLocX = truePos['initialX']
    refLocY = truePos['initialY']
    ref_resolution = opt.ref_resolution
    dsm_resolution = opt.dsm_resolution
    utm_x = truePos['utm_x']
    utm_y = truePos['utm_y']
    
    # Calculate cropping boundaries
    half_size = opt.crop_size / 2.0
    
    x_min = int((utm_x - half_size - refLocX) / ref_resolution)
    x_max = int((utm_x + half_size - refLocX) / ref_resolution)
    y_min = int((refLocY - utm_y - half_size) / ref_resolution)
    y_max = int((refLocY - utm_y + half_size) / ref_resolution)
    
    ref_map_crop = ref_map_full[y_min:y_max, x_min:x_max]
    
    if dsm_map_full is not None:
        dx_min = int((utm_x - half_size - refLocX) / dsm_resolution)
        dx_max = int((utm_x + half_size - refLocX) / dsm_resolution)
        dy_min = int((refLocY - utm_y - half_size) / dsm_resolution)
        dy_max = int((refLocY - utm_y + half_size) / dsm_resolution)
        dsm_map_crop = dsm_map_full[dy_min:dy_max, dx_min:dx_max]
    else:
        dsm_map_crop = None
        
    temp_config = config.copy()
    temp_config[f'{region}_HIGH_REF_initialX'] = refLocX + x_min * ref_resolution
    temp_config[f'{region}_HIGH_REF_initialY'] = refLocY - y_min * ref_resolution
    
    return ref_map_crop, dsm_map_crop, temp_config

def rotate_image_and_mask_fixed(img, mask, angle_deg):
    if abs(float(angle_deg)) < 1e-9:
        return img, mask
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_rot = None
    if mask is not None:
        mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_rot, mask_rot

def rotate_image_and_mask_fixed_with_matrix(img, mask, angle_deg):
    h, w = img.shape[:2]
    if abs(float(angle_deg)) < 1e-9:
        M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        return img, mask, M
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    img_rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask_rot = None
    if mask is not None:
        mask_rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img_rot, mask_rot, M

def affine_transform_points(points, M):
    if points is None or len(points) == 0:
        return []
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    M = np.asarray(M, dtype=np.float64).reshape(2, 3)
    out = pts @ M[:, :2].T + M[:, 2]
    return [tuple(map(float, p)) for p in out]

def transform_points_h(points_xy, homography):
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, homography)
    return transformed.reshape(-1, 2)

def save_minima_matches_vis(template_img, sat_img, sen_pts, ref_pts, out_path):
    if len(sen_pts) == 0:
        return
    kpts0 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in sen_pts]
    kpts1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1) for p in ref_pts]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(sen_pts))]
    vis = cv2.drawMatches(template_img, kpts0, sat_img, kpts1, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_path, vis)

def draw_minima_anchor_overlay(sat_crop, template_img, template_mask, homography,
                               anchor_px_sat, projected_center_px, out_path, overlay_alpha=0.55):
    overlay = sat_crop.copy()
    warped = cv2.warpPerspective(template_img, homography, (sat_crop.shape[1], sat_crop.shape[0]))
    warped_mask = cv2.warpPerspective(template_mask, homography, (sat_crop.shape[1], sat_crop.shape[0]), flags=cv2.INTER_NEAREST)
    mask = warped_mask > 127
    if np.any(mask):
        overlay[mask] = cv2.addWeighted(overlay[mask], 1.0 - overlay_alpha, warped[mask], overlay_alpha, 0.0)
    cv2.drawMarker(overlay, tuple(map(int, anchor_px_sat)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 28, 3, cv2.LINE_AA)
    cv2.drawMarker(overlay, tuple(map(int, projected_center_px)), (0, 255, 255), cv2.MARKER_CROSS, 28, 3, cv2.LINE_AA)
    cv2.imwrite(out_path, overlay)

def save_template_center_vis(template_img, template_mask, pointcloud_center_px, anchor_px, out_path, camera_anchor_px=None):
    vis = template_img.copy()
    if template_mask is not None:
        contours, _ = cv2.findContours((template_mask > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (80, 220, 220), 1, cv2.LINE_AA)
    cv2.drawMarker(vis, tuple(map(int, pointcloud_center_px)), (255, 0, 255), cv2.MARKER_CROSS, 30, 3, cv2.LINE_AA)
    cv2.putText(vis, "PC center", tuple(map(int, pointcloud_center_px + np.array([8, -8]))), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.drawMarker(vis, tuple(map(int, anchor_px)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 30, 3, cv2.LINE_AA)
    cv2.putText(vis, "anchor", tuple(map(int, anchor_px + np.array([8, 18]))), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
    if camera_anchor_px is not None:
        cv2.drawMarker(vis, tuple(map(int, camera_anchor_px)), (0, 255, 0), cv2.MARKER_DIAMOND, 26, 3, cv2.LINE_AA)
        cv2.line(vis, tuple(map(int, camera_anchor_px)), tuple(map(int, anchor_px)), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(out_path, vis)

def utm_to_crop_px(utm_xy, crop_origin_x, crop_origin_y, ref_res):
    return (
        int(round((float(utm_xy[0]) - float(crop_origin_x)) / float(ref_res))),
        int(round((float(crop_origin_y) - float(utm_xy[1])) / float(ref_res))),
    )

def draw_labeled_marker(img, pt, color, label, marker_type=cv2.MARKER_CROSS, marker_size=34):
    cv2.drawMarker(img, tuple(map(int, pt)), color, marker_type, marker_size, 4, cv2.LINE_AA)
    x, y = tuple(map(int, pt))
    cv2.putText(img, label, (x + 10, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, cv2.LINE_AA)
    cv2.putText(img, label, (x + 10, max(28, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def overlay_minima_pointcloud_on_crop(ref_crop, anchor, new_ref_initialX, new_ref_initialY, ref_res, alpha=0.38):
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
    to_compare_crop = np.array([[sx, 0.0, tx], [0.0, sy, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
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

def captioned_panel(img, title, size=(520, 360)):
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

def save_anchor_match_process_panel(frame_dir, frame_name, raw_count, valid_count, inlier_count):
    panels = []
    for filename, title in [
        ("template_center_marked.png", "Template center"),
        ("satellite_crop.jpg", "Satellite crop"),
        ("minima_filtered_matches.jpg", "Filtered matches"),
        ("minima_inlier_matches.jpg", "RANSAC inliers"),
        ("match_result_overlay.jpg", "Warp overlay"),
    ]:
        panels.append(captioned_panel(cv2.imread(os.path.join(frame_dir, filename)), title))

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
        title = f"{anchor['frame']}  inliers={anchor.get('inlier_count', 0)}  dYaw={anchor.get('yaw_delta_deg', 0.0):.2f}"
        thumbs.append(captioned_panel(img, title, size=(760, 520)))

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

def json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj

def append_dynamic_minima_log(save_dir, event):
    log_dir = os.path.join(save_dir, "dynamic_minima_feedback")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "dynamic_minima_feedback.log.jsonl")
    event = dict(event)
    event["time"] = datetime.now().isoformat(timespec="seconds")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(json_safe(event), ensure_ascii=False) + "\n")
    return log_path
