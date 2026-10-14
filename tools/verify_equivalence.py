"""Run untouched fast ScalLoc and compare full-precision release outputs.

No dependency on the refactored package. Original source remains read-only.
Tolerance is frozen here, never fitted to observed experiment results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ATOL = 1e-8
RTOL = 1e-10
REQUIRED_FILES = (
    "trajectory_per_frame_comparison.csv", "global_poses_utm.json",
    "global_poses_utm_uncorrected.json", "sequence_metric_scale_calibration.json",
    "window_anchor_summary.json", "anchor_diagnostics.csv", "trajectory_metrics_summary.csv",
)
# No acceptance decision, count, pose, scale or residual is excluded.
IGNORED_KEYS = frozenset({
    "save_dir", "out_dir", "output_dir", "output_path", "source_root", "dataset_root",
    "result_dir", "python", "timestamp", "created_at", "started_at", "finished_at",
    "elapsed", "elapsed_seconds", "runtime",
})
IGNORED_SUFFIXES = ("_sec", "_seconds", "_path", "_dir")


def ignored_key(key: str) -> bool:
    return key in IGNORED_KEYS or key.endswith(IGNORED_SUFFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def install_source_write_guard(source: Path) -> None:
    """Reject Python filesystem write attempts inside the original project."""
    source = source.resolve()

    def check(path: Any) -> None:
        if isinstance(path, (str, bytes, os.PathLike)):
            candidate = Path(os.fsdecode(path)).absolute()
            if is_within(candidate, source):
                raise PermissionError(f"Original project is read-only during validation: {candidate}")

    def audit(event: str, args: tuple[Any, ...]) -> None:
        if event == "open":
            path, mode, flags = args
            write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            if (isinstance(mode, str) and any(char in mode for char in "wax+")) or (int(flags) & write_flags):
                check(path)
        elif event in {"os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime", "os.truncate"}:
            check(args[0])
        elif event in {"os.rename", "os.replace", "os.link"}:
            check(args[0])
            check(args[1])
        elif event == "os.symlink":
            check(args[1])

    sys.addaudithook(audit)


def original_environment(source: Path, dataset: str, max_windows: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONHASHSEED": "42", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1",
        "MPLBACKEND": "Agg", "BATCH_DATASET_NAMES": dataset, "BATCH_MAX_DATASETS": "0",
        "VGGT_MAX_WINDOWS": str(max_windows), "ENABLE_FRAME_PITCH_HEIGHT_PERTURBATION": "0",
        "PITCH_NOISE_MAX_ABS_DEG": "0", "HEIGHT_NOISE_MAX_ABS_FRACTION": "0",
        "USE_VGGT_PITCH_HEIGHT": "0", "SEQUENCE_METRIC_SCALE_MULTIPLIER": "1.0",
        "SEQUENCE_METRIC_SCALE_MAP_PATH": "",
        "THERMAL_UAV_DATASET_ROOT": str(source / "data" / "Thermal-UAV-v2" / "test" / "changsha"),
        "VGGT_WEIGHTS_PATH": str(source / "weights" / "VGGT" / "model.safetensors"),
        "DINOV2_WEIGHTS_PATH": str(source / "Matching_Models" / "MINIMA" / "weights" / "dinov2_vitl14_pretrain.pth"),
    })
    return env


def run_original(args: argparse.Namespace) -> int:
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    if not (source / "ScalLoc1.py").is_file():
        raise FileNotFoundError(source / "ScalLoc1.py")
    if is_within(output, source):
        raise ValueError("Validation output must be outside the original project")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty validation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "_run-original-worker",
               "--source-root", str(source), "--output-dir", str(output),
               "--dataset", args.dataset, "--max-windows", str(args.max_windows)]
    return subprocess.run(command, cwd=source,
                          env=original_environment(source, args.dataset, args.max_windows),
                          check=False).returncode


def original_worker(args: argparse.Namespace) -> int:
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    sys.dont_write_bytecode = True
    os.chdir(source)
    sys.path.insert(0, str(source))
    install_source_write_guard(source)
    names = ("ScalLoc1.py", "local_image_io.py", "绝了卧槽_MINIMA_SE2.py",
             "参考/vggt_reconstructor.py", "Matching_Models/MINIMA/MINIMA_match1.py")
    hashes = {name: sha256(source / name) for name in names}
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "status": "running", "source_root": str(source), "output_dir": str(output),
        "dataset": args.dataset, "max_windows": args.max_windows,
        "python": sys.executable, "python_version": platform.python_version(),
        "source_sha256_before": hashes, "source_write_guard": True,
        "configuration": {key: os.environ.get(key) for key in (
            "PYTHONHASHSEED", "VGGT_MAX_WINDOWS", "ENABLE_FRAME_PITCH_HEIGHT_PERTURBATION",
            "PITCH_NOISE_MAX_ABS_DEG", "HEIGHT_NOISE_MAX_ABS_FRACTION", "USE_VGGT_PITCH_HEIGHT",
            "SEQUENCE_METRIC_SCALE_MULTIPLIER", "SEQUENCE_METRIC_SCALE_MAP_PATH",
            "THERMAL_UAV_DATASET_ROOT", "VGGT_WEIGHTS_PATH", "DINOV2_WEIGHTS_PATH")},
        "comparison_policy": {"atol": ATOL, "rtol": RTOL, "exact_first": True},
    }
    write_json(output / "run_metadata.json", metadata)
    results, failures = [], []
    pipeline = None
    try:
        pipeline = runpy.run_path(str(source / "ScalLoc1.py"), run_name="scalloc_original_validation")
        metadata["pipeline_constants"] = {key: pipeline[key] for key in (
            "GLOBAL_RANDOM_SEED", "VGGT_WINDOW_SIZE", "VGGT_WINDOW_OVERLAP", "VGGT_WINDOW_STRIDE",
            "VGGT_WINDOW_STITCH_MODE", "VGGT_FIXED_K_TRACK_BA_ENABLED", "SAVE_ALL_VISUALIZATIONS",
            "BATCH_EXPERIMENT_COMPACT_MODE", "ANCHOR_MATCH_METHOD", "USE_FINAL_GLOBAL_POSE_GRAPH")}
        write_json(output / "batch_datasets.json", [args.dataset])
        runtime = pipeline["create_runtime_context"](output)
        results.append(pipeline["run_pipeline"](args.dataset, runtime, output))
        # Even metadata-only device queries can initialize CUDA. The original
        # runtime sets its allocator environment while importing its helpers;
        # querying the GPU before that import changes execution conditions.
        # Collect all environment information only after original inference.
        import torch
        import cv2
        import numpy
        metadata["environment"] = {
            "torch": torch.__version__, "numpy": numpy.__version__, "opencv": cv2.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "collection_phase": "after_original_inference",
        }
        metadata["status"] = "completed"
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        failures.append({"dataset_name": args.dataset, "error": repr(exc)})
        traceback.print_exc()
    finally:
        if pipeline is not None:
            pipeline["save_batch_reports"](output, results, failures)
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["source_sha256_after"] = {name: sha256(source / name) for name in names}
        metadata["source_unchanged"] = hashes == metadata["source_sha256_after"]
        if not metadata["source_unchanged"]:
            metadata["status"] = "failed"
        write_json(output / "run_metadata.json", metadata)
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0 if metadata["status"] == "completed" else 1


def csv_value(value: str) -> Any:
    if value in {"True", "False"}:
        return value == "True"
    if value == "":
        return None
    if value[:1] in {"[", "{"}:
        try:
            return json.loads(value)
        except ValueError:
            pass
    try:
        return float(value)
    except ValueError:
        return value


def read_artifact(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8-sig"))
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return {"columns": [key for key in reader.fieldnames or [] if not ignored_key(key)],
                "rows": [{key: csv_value(value) for key, value in row.items()
                          if not ignored_key(key)} for row in reader]}


class Comparison:
    def __init__(self) -> None:
        self.checked = 0
        self.numeric = 0
        self.exact_numeric = 0
        self.within_tolerance = 0
        self.max_absolute_difference = 0.0
        self.ignored: list[str] = []
        self.differences: list[dict[str, Any]] = []
        self.difference_count = 0

    def fail(self, path: str, left: Any, right: Any, reason: str) -> None:
        self.difference_count += 1
        if len(self.differences) < 100:
            self.differences.append({"path": path, "original": repr(left),
                                     "refactored": repr(right), "reason": reason})

    def compare(self, left: Any, right: Any, path: str = "$") -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            all_keys = set(left) | set(right)
            for key in sorted(all_keys):
                child = f"{path}.{key}"
                if ignored_key(key):
                    self.ignored.append(child)
                elif key not in left or key not in right:
                    self.fail(child, left.get(key), right.get(key), "missing_key")
                else:
                    self.compare(left[key], right[key], child)
            return
        if isinstance(left, list) and isinstance(right, list):
            if len(left) != len(right):
                self.fail(path, len(left), len(right), "list_length")
            for index, (old, new) in enumerate(zip(left, right)):
                self.compare(old, new, f"{path}[{index}]")
            return
        self.checked += 1
        if isinstance(left, bool) or isinstance(right, bool):
            if type(left) is not type(right) or left != right:
                self.fail(path, left, right, "boolean_mismatch")
            return
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            self.numeric += 1
            if left == right or (math.isnan(left) and math.isnan(right)):
                self.exact_numeric += 1
                return
            if not math.isfinite(left) or not math.isfinite(right):
                self.fail(path, left, right, "nonfinite_mismatch")
                return
            difference = abs(left - right)
            self.max_absolute_difference = max(self.max_absolute_difference, difference)
            if difference <= ATOL + RTOL * abs(left):
                self.within_tolerance += 1
            else:
                self.fail(path, left, right, "numeric_tolerance")
            return
        if type(left) is not type(right) or left != right:
            self.fail(path, left, right, "exact_value_mismatch")

    def result(self) -> dict[str, Any]:
        return {
            "pass": self.difference_count == 0,
            "exact_numeric_equal": self.difference_count == 0 and self.within_tolerance == 0,
            "checked_leaves": self.checked, "numeric_leaves": self.numeric,
            "exact_numeric_leaves": self.exact_numeric, "tolerance_only_numeric_leaves": self.within_tolerance,
            "max_absolute_difference": self.max_absolute_difference,
            "difference_count": self.difference_count, "first_differences": self.differences,
            "ignored_fields": self.ignored,
        }


def discover_results(root: Path) -> dict[str, Path]:
    if (root / REQUIRED_FILES[0]).is_file():
        return {root.name: root}
    return {path.parent.name: path.parent for path in sorted(root.glob("*/" + REQUIRED_FILES[0]))}


def compare_runs(original: Path, refactored: Path) -> dict[str, Any]:
    old, new = discover_results(original), discover_results(refactored)
    report: dict[str, Any] = {
        "original": str(original.resolve()), "refactored": str(refactored.resolve()),
        "policy": {"atol": ATOL, "rtol": RTOL, "exact_first": True,
                   "ignored_keys": sorted(IGNORED_KEYS), "ignored_suffixes": list(IGNORED_SUFFIXES),
                   "comparison": "abs(new - original) <= atol + rtol * abs(original)",
                   "nan_policy": "Only NaN/NaN pairs are equal; missing, infinity and NaN do not interchange."},
        "original_sequences": sorted(old), "refactored_sequences": sorted(new), "sequences": {},
        "pass": bool(old) and set(old) == set(new), "exact_numeric_equal": True,
    }
    for name in sorted(set(old) & set(new)):
        files: dict[str, Any] = {}
        for filename in REQUIRED_FILES:
            left, right = old[name] / filename, new[name] / filename
            if not left.is_file() or not right.is_file():
                result = {"pass": False, "exact_numeric_equal": False, "error": "required_artifact_missing",
                          "original_exists": left.is_file(), "refactored_exists": right.is_file()}
            else:
                comparison = Comparison()
                comparison.compare(read_artifact(left), read_artifact(right))
                result = comparison.result()
                result.update({"original_sha256": sha256(left), "refactored_sha256": sha256(right)})
            files[filename] = result
            report["pass"] = report["pass"] and result["pass"]
            report["exact_numeric_equal"] = report["exact_numeric_equal"] and result["exact_numeric_equal"]
        report["sequences"][name] = files
    report["exact_numeric_equal"] = report["pass"] and report["exact_numeric_equal"]
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run-original", "_run-original-worker"):
        run = commands.add_parser(name)
        run.add_argument("--source-root", required=True, type=Path)
        run.add_argument("--output-dir", required=True, type=Path)
        run.add_argument("--dataset", default="Rural_Day_Curve_H150_P30_S01")
        run.add_argument("--max-windows", type=int, default=0)
    compare = commands.add_parser("compare")
    compare.add_argument("--original", required=True, type=Path)
    compare.add_argument("--refactored", required=True, type=Path)
    compare.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "compare":
        output = args.output_dir.resolve()
        if (output / "comparison.json").exists():
            raise FileExistsError(output / "comparison.json")
        output.mkdir(parents=True, exist_ok=True)
        report = compare_runs(args.original, args.refactored)
        write_json(output / "comparison.json", report)
        print(json.dumps({key: report[key] for key in ("pass", "exact_numeric_equal", "original_sequences", "refactored_sequences")}, ensure_ascii=False))
        return 0 if report["pass"] else 1
    if args.max_windows < 0:
        parser.error("--max-windows must be nonnegative")
    return original_worker(args) if args.command == "_run-original-worker" else run_original(args)


if __name__ == "__main__":
    raise SystemExit(main())
