# ScalLoc

Thermal-oblique UAV localization with VGGT geometry and satellite-map matching.

This repository packages the original **fast ScalLoc pipeline** for the **Thermal-Oblique** dataset. It reorganizes the runnable implementation into focused Python modules, configurable asset paths, and a command-line entry point. Datasets, satellite maps, model weights, caches, and experiment outputs are **not included**.

> This is the original fast pipeline, not `ScalLoc-v1_end` and not the later experimental pose-refinement prototype. Its original numerical settings and processing logic are intended to remain unchanged. See [reproducibility](docs/REPRODUCIBILITY.md) for the actual regression-test status and results; restructuring alone is not evidence of equivalent predictions.

[中文说明](docs/README_zh.md) · [Data and weights](docs/DATA_AND_WEIGHTS.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [License status](LICENSE_NOTICE.md)

## Pipeline

```text
Thermal frames + frame metadata             Satellite reference map
                 │                                   │
         VGGT window reconstruction                  │
                 │                                   │
       Fixed-intrinsics track BA                     │
                 │                                   │
        Metric alignment and stitching               │
                 └────── MINIMA-RoMa matching ────────┘
                                  │
                       Anchor-based correction
                                  │
                      Trajectory and evaluation
```

The default profile preserves the original 12-frame windows with one-frame overlap, fixed-intrinsics track BA, anchor logic, and per-frame pitch/relative-height priors used by that implementation. Perturbations are disabled by default. This is **not** a pure-VGGT or first-frame-prior-only experiment. Keep this distinction when reporting or comparing results.

## Quick start

### 1. Prepare Python and dependencies

Use an isolated Python environment. The previous local reference run used Python 3.12.9, PyTorch 2.6.0+cu126, torchvision 0.21.0+cu126, and an NVIDIA RTX 4060 Ti with 16 GB VRAM. Those are reference-environment details, not a guarantee of identical results on other hardware or library versions.

Install a matching PyTorch/torchvision pair for your CUDA environment, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

For the exact non-PyTorch versions used in the local equivalence test, use
`requirements-tested.txt` instead. This is a reference snapshot for Python 3.12,
not a cross-platform lockfile; install the matching PyTorch pair separately.

The `run.py` entry point works directly from this checkout; installing the project itself is optional. For development:

```bash
python -m pip install -e . --no-deps
```

CUDA is strongly recommended for these large models. CPU execution is not the validated performance target. The vendored dependencies contain only the inference code needed by the default profile, not their complete upstream training toolchains.

### 2. Prepare external assets

Follow [Data and weights](docs/DATA_AND_WEIGHTS.md). The expected layout is:

```text
data/
├── Thermal-Oblique/test/changsha/
│   └── <sequence>/
│       ├── Thermal/
│       └── Thermal_info.csv
└── maps/changsha/
    ├── ref.tif
    └── dsm.tif                 # Optional; not read by the default fast pipeline
weights/
├── vggt/model.safetensors
├── minima/minima_roma.pth
└── dinov2/dinov2_vitl14_pretrain.pth
```

The default fast pipeline does **not read the DSM**. You can run it without downloading `dsm.tif`. Keep the `paths.dsm_map` field in the configuration as a compatibility placeholder; its file does not need to exist for the default run.

To reuse an existing dataset or checkpoint without duplicating it, edit the placeholder paths in `configs/local.example.yaml` and pass that configuration to `run.py`. All relative paths resolve against the **project root**, not the configuration directory or the shell's current directory.

Download locations for the project's Hugging Face and Baidu Netdisk releases are **to be supplied by the author**. No unpublished download URL is implied by this repository.

### 3. Check resources, then run

Check the configured resources without loading either model:

```bash
python run.py --config configs/default.yaml --check
```

Run the default complete curve sequence:

```bash
python run.py --config configs/default.yaml
```

The default sequence is `Rural_Day_Curve_H150_P30_S01` (138 frames in the reference dataset). `max_windows: 0` means all windows. This command does **not** start all 216 sequences.

Common alternatives:

```bash
# Reuse assets stored outside this checkout.
python run.py --config configs/local.example.yaml --check
python run.py --config configs/local.example.yaml

# Choose one sequence, or repeat --dataset for several sequences.
python run.py --config configs/default.yaml --dataset Rural_Day_Curve_H150_P30_S01

# Short smoke test; not a complete-trajectory evaluation.
python run.py --config configs/default.yaml --max-windows 1

# Explicitly request every available sequence.
python run.py --config configs/default.yaml --all-datasets

# Choose a new output directory.
python run.py --config configs/default.yaml --output-dir outputs/my_run
```

A non-empty explicit output directory is rejected to avoid overwriting earlier results. Use a fresh directory for each run. Detailed visual/debug artifacts remain disabled in the default fast profile; do not compare timings between runs with different saving policies.

## Configuration

`configs/default.yaml` controls asset locations and the supported run options. `configs/regions/changsha.yaml` contains the geographic reference settings for the supplied Changsha satellite map.

| Setting | Purpose |
| --- | --- |
| `paths.dataset_root` | Directory containing the sequence folders |
| `paths.reference_map` | Required satellite reference image |
| `paths.dsm_map` | Compatibility path for an optional DSM; not read by the default fast pipeline |
| `paths.vggt_weights` | VGGT safetensors checkpoint |
| `paths.minima_weights` | MINIMA-RoMa checkpoint |
| `paths.dinov2_weights` | DINOv2 ViT-L/14 backbone checkpoint |
| `run.datasets` | Explicit sequence names |
| `run.max_windows` | `0` for complete sequences; positive values limit windows |
| `run.pitch_max_deg` | Original optional pitch-perturbation amplitude; default `0` |
| `run.height_max_fraction` | Original optional relative-height perturbation amplitude; default `0` |
| `run.vggt_predictions` | Original VGGT-prediction ablation; default `false` |

Changing perturbation or prediction-ablation settings changes the experiment. For a refactoring-equivalence test, keep every run option, checkpoint, map, frame order, and dataset file identical.

## Repository structure

```text
ScalLoc/
├── run.py                     # Public command-line entry point
├── configs/                   # Run and map-reference configuration
├── src/scalloc/
│   ├── configuration.py       # Preserved algorithm defaults
│   ├── pipeline.py            # Window and sequence orchestration
│   ├── reconstruction.py      # VGGT reconstruction and track refinement
│   ├── matching.py            # Satellite matching and anchor decisions
│   ├── geometry.py            # Geometric transformations and fitting
│   ├── geometry_io.py         # Original geometry/helper implementation
│   ├── map_utils.py           # Map coordinates, crops, and transforms
│   ├── reporting.py           # Trajectory evaluation and output
│   └── image_io.py            # Image I/O compatibility
├── third_party/               # Minimal vendored inference dependencies
├── tests/                     # Refactoring and interface tests
├── tools/                     # Validation utilities
├── provenance/                # Source provenance / comparison records
└── docs/                      # Setup and reproducibility documentation
```

Some helper routines remain intentionally close to their original implementation: preserving the numerical behavior takes priority over algorithmic cleanup. Future changes to preprocessing, geometry, random sampling, or model loading should be accompanied by a numerical regression comparison.

## Results and reproducibility

Use [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the measured comparison between the original and refactored implementations. A successful resource check or import test is not a full inference test. A one-window smoke test is not a complete-trajectory result.

The original implementation uses stochastic matching and GPU kernels. Seeds are preserved, but reproducibility across different drivers, hardware, CUDA, or library versions is not promised. Do not replace a checkpoint with a newer similarly named checkpoint when reproducing results.

When reporting a run, retain its configuration, dependency versions, selected sequence names, checkpoint hashes, and final metrics. Do not upload personal absolute-path configurations, dataset metadata, or large outputs as part of the code release.

## Contributing and license status

Keep data and weights external, make behavior changes explicit, and separate refactoring-only changes from algorithm changes. Please include a reproducible command and regression evidence with numerical changes. No publication metadata or citation entry is invented here; the author can add the final paper information when available.

The project's own license has **not yet been selected by the author**. See [LICENSE_NOTICE.md](LICENSE_NOTICE.md) before redistribution or public release. Included third-party code remains subject to its own licenses, listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Code and checkpoint licenses can differ.
