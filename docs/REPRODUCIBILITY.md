# Refactoring equivalence

## Verified result — 2026-10-14

The reorganized fast ScalLoc implementation was run on the **complete**
`Rural_Day_Curve_H150_P30_S01` sequence: **138 frames, 13 windows**.
It was compared with both:

1. A fresh execution of the unmodified original `ScalLoc1.py`.
2. The previously saved original run `VGGT_Window_MINIMA_BATCH_20261002_113700`.

**Both comparisons passed with exact numerical equality**, not merely matching
rounded metrics. Across the seven artifact families below, all **23,558 numeric
leaves** matched exactly. No tolerance-only matches were needed.

| Implementation | ME (m) | MedE (m) | PosAcc@20 (%) | YawAcc@10 (%) |
| --- | ---: | ---: | ---: | ---: |
| Saved original, 2026-10-02 | 15.65 | 16.85 | 66.67 | 100.00 |
| Original rerun | 15.65 | 16.85 | 66.67 | 100.00 |
| Refactored package | 15.65 | 16.85 | 66.67 | 100.00 |

Unrounded corrected ME: `15.652822411925504` m.
Unrounded corrected MedE: `16.852320791606296` m.
The three `trajectory_per_frame_comparison.csv` files also have identical bytes:

```text
SHA256 23ba38ef0a824c598f7ed93ad0ea5cc71573281ce06a6f1689d815dc68efadea
```

This validates **one complete trajectory and the default fast profile**, not all
216 trajectories, every ablation, or another machine. It is not an evaluation of
the later pure-VGGT prototype and does not establish a new accuracy benchmark.

## Preserved experiment

- Original fast pipeline, not `ScalLoc-v1_end`.
- Window size 12, overlap 1, stride 11; complete sequence (`max_windows: 0`).
- Original fixed-intrinsics track BA, map matching, anchor checks and propagation.
- Original per-frame pitch and relative-height priors; perturbations disabled.
- Seed 42; original GPU/backend settings retained, without forcing a different
  deterministic algorithm configuration.
- Same input files, map and checkpoints, read from their existing locations.
  No dataset, map or weight file was copied into the new project.

Of 104 top-level functions/classes extracted from the original entry point,
100 retain the same abstract syntax tree. Four have import/resource-path
adaptations only. This audit is recorded in
[`provenance/refactor_manifest.json`](../provenance/refactor_manifest.json).
The original entry-point SHA256 is:

```text
c376903fbb6b4570fef3ed2d548c854e3225e5336aaed075706973fef328e28d
```

The original execution helper rejects filesystem writes under the source
directory. Recorded source hashes before and after execution were identical.

Checkpoint SHA256 values (hashes only; checkpoint files are not included):

| File | SHA256 |
| --- | --- |
| `weights/vggt/model.safetensors` | `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` |
| `weights/minima/minima_roma.pth` | `17f3923bd780e8f1450792706e0643f70c8864ad11bf5fd4f2b54714bac23538` |
| `weights/dinov2/dinov2_vitl14_pretrain.pth` | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |

## Comparison policy and artifacts

[`tools/verify_equivalence.py`](../tools/verify_equivalence.py) compares:

1. `trajectory_per_frame_comparison.csv`
2. `global_poses_utm.json`
3. `global_poses_utm_uncorrected.json`
4. `sequence_metric_scale_calibration.json`
5. `window_anchor_summary.json`
6. `anchor_diagnostics.csv`
7. `trajectory_metrics_summary.csv`

Missing files, frame changes, differing structure, decisions or numeric values
are not silently ignored. Timing, timestamps and filesystem-path fields are
excluded; the full exclusion list and every excluded field appear in the JSON
comparison report. The predeclared numeric fallback is
`abs(new - old) <= 1e-8 + 1e-10 * abs(old)`.
For the successful comparisons, **maximum absolute difference was zero**.

Local-only evidence is retained under `validation/` (excluded from Git):

| Location | Contents |
| --- | --- |
| `original_runtime_order/` | Fresh original result and source-integrity metadata |
| `refactored/` | New package result and resolved run configuration |
| `runtime_order_comparison/comparison.json` | Exact fresh-original comparison |
| `historical_comparison/comparison.json` | Exact saved-original comparison |
| `original/`, `comparison/` | Initial diagnostic run and failed comparison, retained for audit |

### Initialization-order finding

The initial verification helper queried CUDA device information **before** the
original runtime initialized. That run produced small differences in the final
anchored result (ME approximately 15.668 m versus 15.653 m). The source and
refactored algorithms were not changed to compensate.

All device/environment queries were moved **after** original inference, restoring
its initialization order. The rerun then matched both the refactored package and
the saved original result exactly. A CPU regression test now forbids premature
CUDA introspection in this helper. This identifies initialization order as the
observed trigger; it does not claim a proof of the specific low-level CUDA cause.

## Reproduce the checks

Use the same checkpoint files, inputs and environment. Run the original and
refactored implementations in **separate fresh processes**. Prepare a private
`configs/local.yaml` using [`configs/local.example.yaml`](../configs/local.example.yaml)
to point to existing external assets. Never commit that private configuration.
Use new output directories, not directories containing earlier results.

```bash
# Requires the original source tree locally; does not alter that tree.
python tools/verify_equivalence.py run-original --source-root /path/to/original --output-dir validation/original_new --dataset Rural_Day_Curve_H150_P30_S01 --max-windows 0

python run.py --config configs/local.yaml --check
python run.py --config configs/local.yaml --output-dir validation/refactored_new

python tools/verify_equivalence.py compare --original validation/original_new --refactored validation/refactored_new --output-dir validation/comparison_new

# CPU-only interface, provenance and comparator tests; no model inference.
python -m pip install -r requirements-dev.txt
python -m pytest -q

# Inspect the code-only upload inventory; no archive or duplicate assets created.
python tools/check_release.py --manifest validation/release_manifest.json
```

For the recorded runs, `PYTHONHASHSEED=42` and `PYTHONDONTWRITEBYTECODE=1`
were set before starting Python. On Windows PowerShell, use
`$env:PYTHONHASHSEED='42'` and `$env:PYTHONDONTWRITEBYTECODE='1'`.
On a POSIX shell, use `export PYTHONHASHSEED=42 PYTHONDONTWRITEBYTECODE=1`.

## Tested environment and limits

- Windows, Python 3.12.9; NVIDIA GeForce RTX 4060 Ti, 16 GB.
- PyTorch 2.6.0+cu126; torchvision 0.21.0+cu126; CUDA runtime 12.6.
- Exact non-PyTorch versions: [`requirements-tested.txt`](../requirements-tested.txt).
- **36 CPU tests passed** after the final error-handling and verification-tool
  changes. They do not replace the full GPU comparison above.
- The final CLI-only error handling also records failed status when the original
  MINIMA loader exits on incompatible weights; successful numerical paths are
  unchanged.

The refactored complete run took 151.05 s wall time including model/runtime
initialization; its internal trajectory timer reported 122.47 s. The original
rerun took 147.02 s wall time / 124.23 s internal trajectory time. These are single
runs, not a speed benchmark; different timers must not be compared as if they
measure the same work.

GPU algorithms, library versions and initialization order can affect numerical
reproducibility. Exact equality on this machine is not a cross-platform promise.
Keep this tested profile separate from algorithmic improvements or model updates.
