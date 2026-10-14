"""Original fast settings; all numerical defaults are preserved."""
from __future__ import annotations
import os
import json
import numpy as np
from pathlib import Path
from .paths import project_root

ROOT = project_root()
PACKAGE_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = ROOT / "third_party"
REGION_CONFIG_PATH = Path(os.environ.get("SCALLOC_REGION_CONFIG", str(ROOT / "configs/regions/changsha.yaml")))
GLOBAL_RANDOM_SEED = 42


# -----------------------------------------------------------------------------
# Experiment configuration
# -----------------------------------------------------------------------------
DATASET_NAME = "草地-直线-长" # 草地

DATASET_ROOT = Path(
    os.environ.get(
        "THERMAL_UAV_DATASET_ROOT",
        str(ROOT / "data" / "Thermal-Oblique" / "test" / "changsha"),
    )
)
REGION_NAME = "changsha"
REF_TYPE = "HIGH"
BATCH_DATASET_NAMES = [
    name.strip()
    for name in os.environ.get("BATCH_DATASET_NAMES", "").split(",")
    if name.strip()
]
BATCH_MAX_DATASETS = int(os.environ.get("BATCH_MAX_DATASETS", "0"))
# Optional per-frame robustness perturbation.  Disabled by default.
ENABLE_FRAME_PITCH_HEIGHT_PERTURBATION = (
    os.environ.get("ENABLE_FRAME_PITCH_HEIGHT_PERTURBATION", "0")
    .strip().lower() in {"1", "true", "yes", "on"}
)
USE_VGGT_PITCH_HEIGHT = (
    os.environ.get("USE_VGGT_PITCH_HEIGHT", "0")
    .strip().lower() in {"1", "true", "yes", "on"}
)
PITCH_NOISE_MAX_ABS_DEG = float(
    os.environ.get("PITCH_NOISE_MAX_ABS_DEG", "0")
)
HEIGHT_NOISE_MAX_ABS_FRACTION = float(
    os.environ.get("HEIGHT_NOISE_MAX_ABS_FRACTION", "0")
)
if PITCH_NOISE_MAX_ABS_DEG < 0 or HEIGHT_NOISE_MAX_ABS_FRACTION < 0:
    raise ValueError("noise amplitudes must be non-negative")
# Debug subset switch.  When True, batch execution is restricted to the
# datasets recorded in DEBUG_FAILURE_DATASETS_JSON.  If BATCH_DATASET_NAMES
# is also set, the two lists are intersected (preserving its explicit order).
RUN_DEBUG_FAILURE_DATASETS_ONLY = False
DEBUG_FAILURE_DATASETS_JSON = (
    ROOT / "Result" / "216-3_debug_failures_20260725_130022"
    / "debug_failure_datasets.json"
)

# Large-scale experiment mode.  True suppresses verbose per-window logs and
# nonessential structured outputs, and shows tqdm progress instead.  Visual
# artifacts are controlled independently and exclusively by the master switch
# below.
BATCH_EXPERIMENT_COMPACT_MODE = True

# Master switch for every visual/debug artifact: direct/BEV match images,
# per-window point clouds and intrinsics, and the final trajectory map.  Metric
# JSON/CSV/PKL files are always saved regardless of this value.
SAVE_ALL_VISUALIZATIONS = False

# Reproduce ly/ObliqueLoc's direct oblique-image -> satellite -> SWG chain
# while retaining the current MINIMA matcher.
ENABLE_LY_DIRECT_ANCHOR_PIPELINE = True
# True reproduces LY's VGGT point-surface settings exactly (2nd-percentile
# confidence + depth unprojection, without fixed-K track BA).  It is kept for
# ablation, but was substantially worse on the current thermal dataset.
LY_STRICT_VGGT_SURFACE_PROFILE = False

VGGT_WINDOW_SIZE = 12 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 15
# Dense overlap is an independent ablation.  It changes every VGGT window's
# image composition, so keep the established one-frame schedule by default and
# evaluate overlap=5 separately from the dual-anchor change.
ENABLE_DENSE_WINDOW_OVERLAP = False
VGGT_WINDOW_OVERLAP = (
    1 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE
    else (5 if ENABLE_DENSE_WINDOW_OVERLAP else 1) # 7
)
VGGT_WINDOW_STRIDE = VGGT_WINDOW_SIZE - VGGT_WINDOW_OVERLAP
# 0: fit Sim(2) from at least two overlap frames.
# 1: directly place the current window on the previous window's shared last pose.
VGGT_WINDOW_STITCH_MODE = (
    1 if (ENABLE_LY_DIRECT_ANCHOR_PIPELINE or ENABLE_DENSE_WINDOW_OVERLAP) else 1
)
VGGT_MAX_WINDOWS = int(os.environ.get("VGGT_MAX_WINDOWS", "0"))
VGGT_TARGET_SIZE = 518
VGGT_CONF_PERCENTILE = (
    2.0 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE and LY_STRICT_VGGT_SURFACE_PROFILE else 25.0
)
# VGGT_MAX_POINTS = 350_000
VGGT_MAX_POINTS = 0
VGGT_USE_DEPTH_UNPROJECT = bool(
    ENABLE_LY_DIRECT_ANCHOR_PIPELINE and LY_STRICT_VGGT_SURFACE_PROFILE
)
# Physical thermal-camera calibration at the 1280 x 1024 source resolution.
# 9.83 mm is the sensor diagonal (not its width): f=12 mm gives fx=fy=2001.06 px,
# and its 35.47-degree horizontal FOV agrees with the supplied 35.4-degree value.
# VGGT cannot consume a known K directly. When enabled, its output point rays are
# recast through this K while retaining VGGT's predicted per-pixel depth.
VGGT_FIXED_SOURCE_INTRINSICS_ENABLED = False
# Experimental multi-view calibration refinement. This uses VGGT's native
# cross-frame tracks and fixed K, but deliberately does not use the dense
# per-frame ray-recast above.
VGGT_FIXED_K_TRACK_BA_ENABLED = not bool(
    ENABLE_LY_DIRECT_ANCHOR_PIPELINE and LY_STRICT_VGGT_SURFACE_PROFILE
)
VGGT_TRACK_BA_QUERY_COLUMNS = 18
VGGT_TRACK_BA_QUERY_ROWS = 14
VGGT_FIXED_SOURCE_INTRINSICS = {
    "fx": 2001.0577474459521,
    "fy": 2001.0577474459521,
    "cx": 640.0,
    "cy": 512.0,
    "width": 1280.0,
    "height": 1024.0,
    "focal_len_mm": 12.0,
    "sensor_diagonal_mm": 9.83,
    "horizontal_fov_deg": 35.4,
}
# True: rasterize every finite point returned by the window reconstruction.
# False: use the legacy radius/percentile/side-length crop below.


BEV_USE_FULL_POINT_CLOUD = True
BEV_MAX_RADIUS_M = 360.0
BEV_MAX_SIDE_PX = 1800
# Extra satellite-map context around every side of the BEV footprint, in metres.
SATELLITE_FOOTPRINT_EXPAND_M = 20
SATELLITE_MAX_PITCH_GROUND_RANGE_M = 1200.0
SATELLITE_BEV_OVERLAY_ALPHA = 0.45

ANCHOR_MATCH_METHOD = "minima"  # "template": template_match_optimizer peak-shape; "minima": MINIMA_Roma
# Two logically parallel satellite observations.  Both branches estimate a
# candidate for the same window; the fused result is applied exactly once.
ENABLE_BEV_ANCHOR = False
ENABLE_MULTIFRAME_3D_UTM_ANCHOR = True
ENABLE_DUAL_ANCHOR_FUSION = True
# Conditional cross-check/relocalization: LY remains the primary branch.  BEV
# runs only when LY is rejected, makes a large innovation, or the sequence has
# already lost recent anchors.  Set False to restore the original behavior.
ENABLE_BEV_RESCUE_MODE = True
BEV_RESCUE_LOST_AFTER_REJECTIONS = 2
# Keep BEV and satellite crops at their original metric scale.  Rescue may
# change the estimated pose, but it must not enlarge the search footprint.
BEV_RESCUE_EXTRA_MARGIN_STEP_M = 0.0
BEV_RESCUE_MAX_EXTRA_MARGIN_M = 0.0
BEV_RESCUE_AGREEMENT_POSITION_M = 20.0
BEV_RESCUE_AGREEMENT_YAW_DEG = 6.0
BEV_RESCUE_AGREEMENT_SCALE_LOG = 0.15
# A BEV-only rescue is not allowed to rewrite the sequential anchor state from
# one match.  The first hypothesis is kept pending and must agree with a later
# overlapping window.  Set False to restore the former one-shot lost recovery.
ENABLE_BEV_RESCUE_TEMPORAL_CONFIRMATION = False
BEV_RESCUE_CONFIRM_MIN_OVERLAP_FRAMES = 3
BEV_RESCUE_CONFIRM_POSITION_M = 20.0
BEV_RESCUE_CONFIRM_YAW_DEG = 6.0
BEV_RESCUE_CONFIRM_SCALE_LOG = 0.15
BEV_RESCUE_PENDING_MAX_AGE_WINDOWS = 1
BEV_RESCUE_CONFIRMED_APPLY_BLEND = 0.65
# When the direct LY oblique crop has drifted, use an accepted same-scale BEV
# match to recenter the oblique satellite tiles, then run LY once more as the
# geometric verifier.
ENABLE_BEV_GUIDED_LY_RETRY = True
BEV_GUIDED_LY_USE_TEMPLATE_COARSE = False
BEV_GUIDED_LY_EXTRA_MARGIN_M = 0.0
BEV_GUIDED_LY_MIN_INLIERS = 30
BEV_GUIDED_LY_MIN_INLIER_RATIO = 0.08
BEV_GUIDED_LY_MAX_RMSE_M = 6.5
BEV_GUIDED_LY_MAX_SCALE_LOG = 0.35
# Sequence-level state machine.  The existing LY + BEV-rescue path stays
# untouched for ordinary flights.  If the first reconstructed window moves
# tangentially to its camera/view direction (the typical centre-facing orbit),
# the whole sequence instead uses the legacy per-window BEV anchor behaviour.
ENABLE_ORBIT_LEGACY_BEV_PIPELINE = True
ORBIT_LEGACY_BEV_MIN_VALID_PAIRS = 5
ORBIT_LEGACY_BEV_MIN_SUPPORT_RATIO = 0.70
ORBIT_LEGACY_BEV_MIN_PERPENDICULAR_ANGLE_DEG = 65.0
ORBIT_LEGACY_BEV_MIN_NET_YAW_CHANGE_DEG = 12.0
ORBIT_LEGACY_BEV_MIN_TOTAL_YAW_CHANGE_DEG = 18.0
# Multi-frame image -> satellite -> VGGT point-map correspondence branch.
# A non-positive count uses every frame, matching ly's window-level solve.
# MULTIFRAME_KEYFRAME_COUNT = 0 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 5
MULTIFRAME_KEYFRAME_COUNT =  3
# Follow ly/ObliqueLoc: crop a fixed 512 px square around each propagated
# optical-axis ground point.  At 0.3 m/px this is 153.6 m per side.
MULTIFRAME_SATELLITE_TILE_SIZE_PX = 1024
MULTIFRAME_ROTATION_CONTEXT_RATIO = 0.25
MULTIFRAME_POINT_LOOKUP_RADIUS_PX = 5.0
MULTIFRAME_USE_RAYCAST = False
MULTIFRAME_RAYCAST_EXCLUDE_QUERY_FRAME_POINTS = False
MULTIFRAME_RAYCAST_DEPTH_EPSILON = 0.0
MULTIFRAME_RAYCAST_DEPTH_WEIGHT = 0.0
MULTIFRAME_GRID_SIZE = 12 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 6
MULTIFRAME_MAX_MATCHES_PER_GRID_PAIR = 8 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 4
MULTIFRAME_MAX_CORRESPONDENCES_PER_FRAME = 1000 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 450
MULTIFRAME_MIN_CORRESPONDENCES = 20 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 36
MULTIFRAME_MIN_INLIERS = 80 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 24
MULTIFRAME_MIN_INLIER_RATIO = 0.01 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 0.08
MULTIFRAME_RANSAC_THRESHOLD_M = 12.0 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 10.0
MULTIFRAME_MAX_RMSE_M = 15.0 if ENABLE_LY_DIRECT_ANCHOR_PIPELINE else 8.0
# A weak/occluded selected keyframe must not reject a robust consensus from
# the other frames.  Keep this only for strict all-views ablations.
MULTIFRAME_REQUIRE_ALL_SELECTED_INLIER_FRAMES = False
MULTIFRAME_RANSAC_ITERATIONS = 5000
MULTIFRAME_RANSAC_SAMPLE_SIZE = 4
MULTIFRAME_RANSAC_MIN_UAV_GRIDS = 4
MULTIFRAME_RANSAC_MIN_SAT_GRIDS = 4
MULTIFRAME_RANSAC_MIN_HULL_AREA_FRAC = 0.01
MULTIFRAME_RANSAC_COLLINEAR_EPS = 0.03
MULTIFRAME_RANSAC_MIN_LATERAL_SPAN_FRAC = 0.08
MULTIFRAME_RANSAC_MAX_SAMPLE_RETRIES = 100
MULTIFRAME_ROBUST_ITERS = 5
MULTIFRAME_HUBER_DELTA_M = 6.0
# Do not retry the direct UAV-to-satellite match with alternative yaw angles.
# Each window now runs MINIMA once for each selected keyframe at the propagated
# yaw only.
MULTIFRAME_YAW_OFFSETS_DEG = (0.0,)
MULTIFRAME_MAX_ACCEPTED_YAW_OFFSET_DEG = 15.0
MULTIFRAME_GATE_STRONG_DISTANCE_M = 10.0
MULTIFRAME_GATE_WEAK_DISTANCE_M = 30.0
MULTIFRAME_GATE_WEAK_MIN_INLIERS = 150
MULTIFRAME_GATE_WEAK_MAX_RMSE_M = 10.0
MULTIFRAME_GATE_WEAK_MIN_SAT_GRIDS = 6
MULTIFRAME_GATE_OVERRIDE_MAX_DISTANCE_M = 180
MULTIFRAME_GATE_OVERRIDE_MIN_SAT_GRIDS = 80
# MINIMA can keep excellent positional support while its window-level SWG yaw
# becomes ambiguous on repetitive thermal texture.  Position still uses full
# SWG; attitude is fused independently around the stable no-anchor VGGT yaw.
LY_DIRECT_YAW_CORRECTION_LIMIT_DEG = 8.0

# Agreement thresholds decide whether two valid candidates may be averaged.
# They are not hard rejection gates: disagreement selects the better-supported
# branch conservatively instead of averaging incompatible transforms.
DUAL_ANCHOR_AGREEMENT_POSITION_M = 35.0
DUAL_ANCHOR_AGREEMENT_YAW_DEG = 8.0
DUAL_ANCHOR_AGREEMENT_SCALE_LOG = 0.25
DUAL_ANCHOR_CONFLICT_MULTIFRAME_SCORE_RATIO = 0.85
ANCHOR_APPLY_SATELLITE_YAW = True
# LY applies the complete accepted SWG Sim(2) to the current window.  Keep the
# legacy causal scale filter only for the original BEV pipeline, where scale
# was intentionally withheld from each individual satellite observation.
ANCHOR_APPLY_SATELLITE_SCALE = bool(ENABLE_LY_DIRECT_ANCHOR_PIPELINE)
SEQUENTIAL_ANCHOR_SCALE_FILTER_ENABLED = not bool(ENABLE_LY_DIRECT_ANCHOR_PIPELINE)
SEQUENTIAL_ANCHOR_SCALE_MIN_RAW = 0.65
SEQUENTIAL_ANCHOR_SCALE_MAX_RAW = 1.35
SEQUENTIAL_ANCHOR_SCALE_MIN_MULTIPLIER = 0.60
SEQUENTIAL_ANCHOR_SCALE_MAX_MULTIPLIER = 1.60
SEQUENTIAL_ANCHOR_SCALE_INITIAL_STD_LOG = 0.35
SEQUENTIAL_ANCHOR_SCALE_PROCESS_STD_LOG = 0.015
SEQUENTIAL_ANCHOR_SCALE_MIN_MEASUREMENT_STD_LOG = 0.065
SEQUENTIAL_ANCHOR_SCALE_MAX_MEASUREMENT_STD_LOG = 0.25
# Do not start changing metric scale from a lone plausible match.  First
# collect several high-quality, mutually consistent anchors in one direction.
SEQUENTIAL_ANCHOR_SCALE_WARMUP_REQUIRED = 3
SEQUENTIAL_ANCHOR_SCALE_WARMUP_MIN_RATIO = 0.20
SEQUENTIAL_ANCHOR_SCALE_WARMUP_MIN_INLIERS = 500
SEQUENTIAL_ANCHOR_SCALE_WARMUP_MAX_RMSE_M = 5.4
SEQUENTIAL_ANCHOR_SCALE_WARMUP_MIN_DEVIATION = 0.06
SEQUENTIAL_ANCHOR_SCALE_WARMUP_MAX_LOG_SPREAD = 0.12
# A sequence-level correction of the height-derived metric scale.  `1.0`
# preserves the original behaviour.  It is intentionally independent from
# `ANCHOR_APPLY_SATELLITE_SCALE`: the former corrects the local VGGT-to-metre
# conversion once, whereas the latter applies a fresh Sim(2) scale at every
# satellite anchor.
SEQUENCE_METRIC_SCALE_MULTIPLIER = float(
    os.environ.get("SEQUENCE_METRIC_SCALE_MULTIPLIER", "1.0")
)
if not np.isfinite(SEQUENCE_METRIC_SCALE_MULTIPLIER) or SEQUENCE_METRIC_SCALE_MULTIPLIER <= 0.0:
    raise ValueError("SEQUENCE_METRIC_SCALE_MULTIPLIER must be finite and > 0")
# Optional second-pass map, e.g. {"DJI_..._0007": 1.2048}.  This lets one
# batch use a distinct, sequence-level scale without confusing it with a
# global correction shared by every flight.
SEQUENCE_METRIC_SCALE_MAP_PATH = os.environ.get("SEQUENCE_METRIC_SCALE_MAP_PATH", "").strip()
SEQUENCE_METRIC_SCALE_MAP: dict[str, float] = {}
if SEQUENCE_METRIC_SCALE_MAP_PATH:
    with open(SEQUENCE_METRIC_SCALE_MAP_PATH, "r", encoding="utf-8") as file:
        raw_scale_map = json.load(file)
    if not isinstance(raw_scale_map, dict):
        raise ValueError("SEQUENCE_METRIC_SCALE_MAP_PATH must contain a JSON object")
    for _name, _scale in raw_scale_map.items():
        _value = float(_scale)
        if not np.isfinite(_value) or _value <= 0.0:
            raise ValueError(f"Invalid sequence metric scale for {_name}: {_scale}")
        SEQUENCE_METRIC_SCALE_MAP[str(_name)] = _value
# A per-window scale estimate is the least stable part of a satellite Sim(2)
# match.  It must agree with earlier accepted anchors before it can alter the
# propagated VGGT trajectory.  Translation is always retained at the window
# pivot, and yaw remains independently controlled by its own consistency test.
# The full 19-sequence ablation is currently worse with this enabled.  Keep
# the implementation and diagnostics available for targeted experiments, but
# preserve the established Sim(2) behaviour as the default.
ANCHOR_ROBUST_COMPONENT_FUSION = False
ANCHOR_COMPONENT_CONSENSUS_MIN_HISTORY = 2
ANCHOR_SCALE_CONSENSUS_REL_TOLERANCE = 0.05
ANCHOR_YAW_CONSENSUS_DEG = 3.0
ANCHOR_MAX_APPLIED_YAW_DEG = 5.0
# Evaluation-only ablation: keep the VGGT height/yaw but replace its pitch
# when back-projecting a satellite-matched ground point to the camera center.
# Leave this False for normal inference.
USE_GT_PITCH_FOR_ANCHOR_BACKPROJECTION = False
USE_GT_HEIGHT_FOR_ANCHOR_BACKPROJECTION = False
MINIMA_METHOD = "MINIMA_Roma"
MINIMA_MIN_CONF = 0.0
MINIMA_MIN_MATCHES = 12
MINIMA_INLIER_RATIO_GATE_ENABLED = True
MINIMA_MIN_INLIER_RATIO = 0.16
MINIMA_RANSAC_THRESHOLD_M = 8.0
MINIMA_MAX_RMSE_M = 5.5
# Normal-scale anchors may correct position/yaw.  A high-quality match outside
# this narrow range is not discarded: it becomes a scale-only observation for
# the causal filter, never a current-window pose correction.
MINIMA_RESIDUAL_SCALE_MIN = 0.85
MINIMA_RESIDUAL_SCALE_MAX = 1.42
MINIMA_SCALE_OBSERVATION_MIN = 0.65
MINIMA_SCALE_OBSERVATION_MAX = 1.55

TEMPLATE_SEARCH_MARGIN_M = 160.0
TEMPLATE_YAW_MIN_DEG = -18.0
TEMPLATE_YAW_MAX_DEG = 18.0
TEMPLATE_YAW_STEP_DEG = 3.0
TEMPLATE_TOPK_PER_YAW = 5
TEMPLATE_NMS_RADIUS_M = 45.0
TEMPLATE_MIN_NCC_SCORE = -1.0
OVERLAP_MAX_RMSE_M = 20.0
ANCHOR_CONTINUITY_GATE_ENABLED = True
TEMPLATE_ANCHOR_CONTINUITY_GATE_ENABLED = True
MINIMA_ANCHOR_CONTINUITY_GATE_ENABLED = True
# Position innovation is diagnostic-only. Large corrections are blended into
# later windows instead of being rejected outright.
ANCHOR_LARGE_POSITION_INNOVATION_M = 60.0
ANCHOR_LARGE_POSITION_INNOVATION_BLEND = 0.50
ANCHOR_MAX_YAW_INNOVATION_DEG = 8.0
# Ablation switch: when False, a large yaw innovation is recorded but does not
# reject the whole anchor.  Keep the threshold itself for diagnostics/blending.
ANCHOR_YAW_INNOVATION_HARD_REJECT_ENABLED = False
ANCHOR_LARGE_YAW_INNOVATION_BLEND = 0.50
# Keep extrapolation diagnostics, but do not hard-reject satellite anchors
# from an already drifting propagated trajectory.
ANCHOR_EXTRAPOLATION_GATE_ENABLED = False
ANCHOR_EXTRAPOLATION_HISTORY = 3
ANCHOR_EXTRAPOLATION_MIN_HISTORY = 2
ANCHOR_EXTRAPOLATION_MAX_POSITION_ERROR_M = 45.0
ANCHOR_EXTRAPOLATION_MAX_YAW_ERROR_DEG = 10.0

ANCHOR_ACCEPTED_WEIGHT = 5.0
PROPAGATED_WINDOW_WEIGHT = 1.0
# Final global position-graph refinement. Set to True to enable it.
USE_FINAL_GLOBAL_POSE_GRAPH = False
# True: keep VGGT-only relative motion as odometry and use satellite matches only
# as soft anchor observations. False restores the previous behavior, where each
# anchor Sim(2) scale also changes the odometry inside its window.
POSE_GRAPH_USE_ANCHOR_FREE_ODOMETRY = True
POSE_GRAPH_ODOM_SIGMA_M = 1.5
POSE_GRAPH_ANCHOR_SIGMA_M = 6.0
POSE_GRAPH_OPTIMIZE_GLOBAL_SCALE = True
POSE_GRAPH_GLOBAL_SCALE_PRIOR_SIGMA = 0.25
POSE_GRAPH_GLOBAL_SCALE_WARN_MIN = 0.55
POSE_GRAPH_GLOBAL_SCALE_WARN_MAX = 1.80


