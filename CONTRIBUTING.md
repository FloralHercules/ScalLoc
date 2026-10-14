# Contributing

Keep algorithm changes separate from packaging or path changes. The initial
release preserves the validated ScalLoc numerical pipeline.

- Run `python -m pytest tests` for asset-free CPU checks.
- Never commit data, checkpoints, generated results, private download tokens or
  personal configurations. Run `python tools/check_release.py` before a release.
- A numerical change requires a new protocol and complete-trajectory evaluation;
  do not relax the original equivalence tolerances to make a refactor pass.
- Preserve third-party copyright and license notices.
- File issues with the command, software versions and traceback. Do not include
  credentials or upload private images without permission.

The license for original ScalLoc code remains pending. See `LICENSE_NOTICE.md`.
The official Thermal-Oblique resource page and placement instructions are in
`docs/DATA_AND_WEIGHTS.md`.
