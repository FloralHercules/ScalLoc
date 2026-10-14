# Third-party notices

ScalLoc includes a minimal subset of third-party inference code. Each component retains its own license and copyright notices. The list below does not grant a new license to upstream code, checkpoints, or datasets.

| Component | Source | Included location | License copy |
| --- | --- | --- | --- |
| VGGT | [facebookresearch/vggt](https://github.com/facebookresearch/vggt) | `third_party/vggt/` | [VGGT License](third_party/vggt/LICENSE.txt) |
| MINIMA | [LSXI7/MINIMA](https://github.com/LSXI7/MINIMA) | `third_party/minima/` | [Apache License 2.0](third_party/minima/LICENSE) |
| RoMa | [Parskatt/RoMa](https://github.com/Parskatt/RoMa), as included in MINIMA | `third_party/minima/third_party/RoMa_minima/` | [MIT License](third_party/minima/third_party/RoMa_minima/LICENSE) |
| DINOv2 implementation embedded in RoMa | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | RoMa's `romatch/models/transformer/` | [Apache License 2.0](third_party/minima/third_party/RoMa_minima/LICENSE.DINOv2) |

The source headers preserve their original copyright attributions. RoMa's upstream documentation explicitly separates its MIT-licensed implementation from the Apache-licensed DINOv2 code. Do not remove the DINOv2 license merely because a standalone DINOv2 repository is not vendored.

The copied upstream README files are available as `README.upstream.md` alongside the respective component. These preserve upstream context; their training examples and optional dependencies are not a promise that the full upstream repository is included here. Only the default inference dependency subset is packaged. Local adapters and import/path integration are provided for this ScalLoc release.

## Checkpoints and other assets

No model checkpoint, dataset, satellite image, or DSM is distributed in this code package. Downloading or redistributing those assets is governed by their own terms. The VGGT source snapshot contains the license dated July 29, 2025; its README separately describes restrictions on the original VGGT-1B checkpoint. Do not infer a checkpoint's commercial-use rights from a source-code license alone.

The Thermal-Oblique dataset and map-asset release terms and download links must be supplied by the author. This notice does not assign ownership or permissions for those assets.

## Other dependencies and acknowledgments

Python dependencies installed through `requirements.txt`, PyTorch, and torchvision remain subject to their respective licenses. They are not relicensed by this repository.

The original ScalLoc implementation contains comments referring to earlier geometry/localization implementations. Those comments are retained. Before public release, the author should verify additional provenance and acknowledgments for contributed or adapted project-specific helper code; this notice does not invent missing authorship or publication metadata.
