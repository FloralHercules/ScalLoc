"""CPU-only checks for the release equivalence gate; no model imports."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


_TOOL = Path(__file__).resolve().parents[1] / "tools" / "verify_equivalence.py"
_SPEC = importlib.util.spec_from_file_location("equivalence_tool", _TOOL)
equivalence = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(equivalence)


def compare(left, right):
    checker = equivalence.Comparison()
    checker.compare(left, right)
    return checker.result()


def test_exact_values_and_nan():
    report = compare({"pose": [1.0, float("nan")], "accepted": True},
                     {"pose": [1.0, float("nan")], "accepted": True})
    assert report["pass"] and report["exact_numeric_equal"]


def test_frozen_numeric_tolerance_is_reported_not_exact():
    report = compare(1.0, 1.0 + 5e-9)
    assert report["pass"] and not report["exact_numeric_equal"]
    assert report["tolerance_only_numeric_leaves"] == 1
    assert equivalence.ATOL == 1e-8 and equivalence.RTOL == 1e-10


@pytest.mark.parametrize("left,right", [(1.0, 1.001), (True, 1), (False, True),
                                          (float("nan"), None), (float("inf"), float("nan")),
                                          ("frame_001", "frame_002"), ([1], [1, 2]),
                                          ({"accepted": True}, {})])
def test_mismatch_fails(left, right):
    assert not compare(left, right)["pass"]


def test_time_and_path_exclusion_does_not_ignore_algorithm_fields():
    left = {"runtime_sec": 12.0, "save_dir": "old", "scale": 1.0, "inlier_count": 80}
    right = {"runtime_sec": 90.0, "save_dir": "new", "scale": 1.0, "inlier_count": 80}
    assert compare(left, right)["pass"]
    right["scale"] = 1.01
    assert not compare(left, right)["pass"]


def test_csv_identity_fields_and_boolean_types():
    assert equivalence.csv_value("True") is True
    assert equivalence.csv_value("frame_001") == "frame_001"
    assert equivalence.csv_value("") is None
    assert equivalence.csv_value('["frame_001"]') == ["frame_001"]


def create_results(root):
    sequence = root / "sequence"
    sequence.mkdir(parents=True)
    for filename in equivalence.REQUIRED_FILES:
        if filename.endswith(".json"):
            (sequence / filename).write_text(json.dumps({"pose": [1, 2, 3], "accepted": True}), encoding="utf-8")
        else:
            (sequence / filename).write_text("frame,position_error_m,runtime_sec\nf0,1.0,20\n", encoding="utf-8")
    return sequence


def test_run_comparison_requires_every_artifact_and_detects_pose_change(tmp_path):
    left = create_results(tmp_path / "old")
    right = create_results(tmp_path / "new")
    assert equivalence.compare_runs(left.parent, right.parent)["exact_numeric_equal"]
    (right / "global_poses_utm.json").write_text('{"pose": [1, 2, 4], "accepted": true}', encoding="utf-8")
    assert not equivalence.compare_runs(left.parent, right.parent)["pass"]
    (right / "global_poses_utm.json").unlink()
    assert not equivalence.compare_runs(left.parent, right.parent)["pass"]


def test_empty_and_missing_sequences_fail(tmp_path):
    assert not equivalence.compare_runs(tmp_path, tmp_path)["pass"]
    create_results(tmp_path / "old")
    assert not equivalence.compare_runs(tmp_path / "old", tmp_path / "new")["pass"]


def test_original_environment_fixes_original_run_settings(tmp_path):
    env = equivalence.original_environment(tmp_path, "curve", 0)
    assert env["PYTHONHASHSEED"] == "42"
    assert env["USE_VGGT_PITCH_HEIGHT"] == "0"
    assert env["ENABLE_FRAME_PITCH_HEIGHT_PERTURBATION"] == "0"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["BATCH_DATASET_NAMES"] == "curve"


def test_source_write_guard_blocks_write_but_allows_reads(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    probe = source / "keep.txt"
    probe.write_text("preserved", encoding="utf-8")
    code = (
        "import runpy, pathlib\n"
        f"ns=runpy.run_path({str(_TOOL)!r})\n"
        f"ns['install_source_write_guard'](pathlib.Path({str(source)!r}))\n"
        f"p=pathlib.Path({str(probe)!r})\n"
        "assert p.read_text(encoding='utf-8') == 'preserved'\n"
        "try:\n"
        "    p.write_text('modified',encoding='utf-8')\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('guard did not block')\n"
    )
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert probe.read_text(encoding="utf-8") == "preserved"


def test_original_worker_never_queries_cuda_before_original_inference(tmp_path, monkeypatch):
    """Metadata must not initialize CUDA before the legacy allocator setup."""
    source, output = tmp_path / "original", tmp_path / "output"
    source.mkdir()
    output.mkdir()
    for name in ("ScalLoc1.py", "local_image_io.py", "绝了卧槽_MINIMA_SE2.py",
                 "参考/vggt_reconstructor.py", "Matching_Models/MINIMA/MINIMA_match1.py"):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# untouched source fixture\n", encoding="utf-8")
    events = []

    def create_runtime(directory):
        assert events == []
        events.append("create_runtime")
        return "runtime"

    def inference(dataset, runtime, directory):
        assert events == ["create_runtime"]
        assert runtime == "runtime"
        events.append("inference_completed")
        return {"dataset_name": dataset}

    def cuda_available():
        assert "inference_completed" in events
        events.append("cuda_metadata_query")
        return True

    def device_name(index):
        assert "inference_completed" in events
        events.append("device_metadata_query")
        return "mock GPU, no CUDA initialization"

    pipeline = {key: 0 for key in (
        "GLOBAL_RANDOM_SEED", "VGGT_WINDOW_SIZE", "VGGT_WINDOW_OVERLAP", "VGGT_WINDOW_STRIDE",
        "VGGT_WINDOW_STITCH_MODE", "VGGT_FIXED_K_TRACK_BA_ENABLED", "SAVE_ALL_VISUALIZATIONS",
        "BATCH_EXPERIMENT_COMPACT_MODE", "ANCHOR_MATCH_METHOD", "USE_FINAL_GLOBAL_POSE_GRAPH")}
    pipeline.update(create_runtime_context=create_runtime, run_pipeline=inference,
                    save_batch_reports=lambda *args: None)
    fake_torch = SimpleNamespace(
        __version__="mock", version=SimpleNamespace(cuda="mock"),
        cuda=SimpleNamespace(is_available=cuda_available, get_device_name=device_name),
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=False, allow_tf32=False),
                                 cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))),
        are_deterministic_algorithms_enabled=lambda: False,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(__version__="mock"))
    monkeypatch.setitem(sys.modules, "numpy", SimpleNamespace(__version__="mock"))
    monkeypatch.setattr(equivalence.runpy, "run_path", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(equivalence, "install_source_write_guard", lambda source: None)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(source_root=source, output_dir=output, dataset="curve", max_windows=0)
    assert equivalence.original_worker(args) == 0
    assert events == ["create_runtime", "inference_completed", "cuda_metadata_query", "device_metadata_query"]
    report = json.loads((output / "run_metadata.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["source_unchanged"] is True
    assert report["environment"]["collection_phase"] == "after_original_inference"
