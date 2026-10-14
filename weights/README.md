# External checkpoints only

Expected files:

```text
vggt/model.safetensors
minima/minima_roma.pth
dinov2/dinov2_vitl14_pretrain.pth
```

The resource release is hosted on
[Hugging Face](https://huggingface.co/datasets/FloralHercules/Thermal-Oblique);
see the asset guide for current availability. No checkpoint is included in Git.
Configure external paths to reuse existing weights
without copying them. Download locations and checksums are documented in
[DATA_AND_WEIGHTS.md](../docs/DATA_AND_WEIGHTS.md). Follow each upstream model's
license; this repository does not change checkpoint redistribution permissions.
