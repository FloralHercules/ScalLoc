# Data and weights

The code release deliberately excludes all datasets, maps, checkpoints, and generated results. This avoids duplicating large files and keeps the repository suitable for Git hosting.

## Download status

| Resource | Release location |
| --- | --- |
| Thermal-Oblique images and sequence metadata | Hugging Face / Baidu Netdisk links to be supplied by the author |
| Changsha satellite reference map | Distribution details and permissions to be supplied by the author |
| Optional Changsha DSM, unused by the default fast pipeline | No download required for the default run; release details to be supplied if distributed |
| Checkpoint bundle or upstream-download instructions | Final release links to be supplied by the author; upstream projects are listed below |

Do not interpret these placeholders as published downloads. The dataset, map imagery, and each model may have different redistribution terms. Hosting a checkpoint on another platform does not change its upstream license.

## Standard directory layout

Paths below are relative to the repository root:

```text
data/
├── Thermal-Oblique/
│   └── test/
│       └── changsha/
│           ├── Rural_Day_Curve_H150_P30_S01/
│           │   ├── Thermal/
│           │   │   └── <original frame files>
│           │   └── Thermal_info.csv
│           └── <other sequence folders>/
│               ├── Thermal/
│               └── Thermal_info.csv
└── maps/
    └── changsha/
        ├── ref.tif
        └── dsm.tif              # Optional; not read by the default fast pipeline
weights/
├── vggt/
│   └── model.safetensors
├── minima/
│   └── minima_roma.pth
└── dinov2/
    └── dinov2_vitl14_pretrain.pth
```

The public dataset name is **Thermal-Oblique**. The `test/changsha` path preserves the input archive's organizational layout; that folder name alone is not a claim of a new, geographically held-out machine-learning split.

Keep the original image filenames, image contents, metadata columns, units, and row associations. Do not resize images, reorder or rewrite metadata, or rename individual frames to prepare this release. The provided pipeline expects the original relationship between `Thermal/` and `Thermal_info.csv`.

The reference local dataset contains 216 sequences and 28,112 thermal frames. A partial installation can run the selected sequences only. The default command selects one complete sequence, `Rural_Day_Curve_H150_P30_S01`, rather than requiring an all-sequence run.

## Maps and geographic configuration

`ref.tif` is the required satellite reference image. `dsm.tif` is an optional digital surface model: the default fast pipeline **does not read it**, so there is no need to download or copy it to run the default profile. Retain the `paths.dsm_map` configuration field as a compatibility placeholder; its target file may be absent.

Geographic settings live in `configs/regions/changsha.yaml` and must match the actual satellite reference map. The supplied configuration describes the original Changsha reference: UTM zone 49N and 0.3 m/pixel. Do not change the image origin, map dimensions, georeferencing, or resolution independently.

Do not substitute an arbitrary satellite image with the same filename. A map mismatch can produce plausible-looking but incorrect crops and locations. When using the standard release assets, keep the provided geographic configuration unchanged.

## Required checkpoints

The default fast profile requires all three files:

| File | Role | Reference SHA256 |
| --- | --- | --- |
| `weights/vggt/model.safetensors` | VGGT reconstruction, depth/point and camera/track outputs | `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e` |
| `weights/minima/minima_roma.pth` | MINIMA-RoMa cross-modal matcher | `17f3923bd780e8f1450792706e0643f70c8864ad11bf5fd4f2b54714bac23538` |
| `weights/dinov2/dinov2_vitl14_pretrain.pth` | RoMa's DINOv2 ViT-L/14 backbone | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |

These hashes identify the checkpoints recorded for the original local reference installation. They are not generic hashes for every release of the corresponding model. A different VGGT checkpoint, DINOv2 variant, or matcher checkpoint changes the reproduction target.

Upstream sources:

- [VGGT](https://github.com/facebookresearch/vggt)
- [MINIMA](https://github.com/LSXI7/MINIMA)
- [RoMa](https://github.com/Parskatt/RoMa)
- [DINOv2](https://github.com/facebookresearch/dinov2)

Read the upstream terms before downloading or redistributing weights. In particular, the vendored VGGT documentation distinguishes its updated code license from the original VGGT-1B checkpoint license. A commercially licensed replacement checkpoint is not automatically numerically interchangeable with the original one.

## Reuse assets without copying

Use `configs/local.example.yaml` for external paths. Its placeholders must be replaced before running. The relevant keys are:

```yaml
paths:
  dataset_root: /path/to/Thermal-Oblique/test/changsha
  reference_map: /path/to/maps/changsha/ref.tif
  dsm_map: /path/to/maps/changsha/dsm.tif  # Compatibility placeholder; file optional
  vggt_weights: /path/to/weights/vggt/model.safetensors
  minima_weights: /path/to/weights/minima/minima_roma.pth
  dinov2_weights: /path/to/weights/dinov2/dinov2_vitl14_pretrain.pth
```

On Windows, quoted forward-slash paths such as `"D:/assets/Thermal-Oblique/test/changsha"` avoid YAML backslash-escape pitfalls. These are examples, not machine-specific defaults. Relative paths resolve from the project root.

Then check resources without loading models:

```bash
python run.py --config configs/local.example.yaml --check
```

Checking resource availability does not validate numerical equivalence or prove the input metadata matches the imagery. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full regression procedure and results.

## What not to commit

Do not add image archives, `Thermal_info.csv` files, maps, weights, model caches, generated trajectories, logs, or personal local configurations to the public code repository. Use the documented external downloads and ignore rules. Keep model and dataset release permissions separate from the project's source-code licensing decision.
