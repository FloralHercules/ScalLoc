"""Configuration and resource validation without importing inference models."""
from __future__ import annotations
import math
from pathlib import Path
import yaml
from .paths import project_root, resolve_path

PATH_FIELDS = {'dataset_root', 'reference_map', 'dsm_map', 'vggt_weights',
               'minima_weights', 'dinov2_weights', 'region_config', 'output_root'}
RUN_FIELDS = {'datasets', 'max_windows', 'pitch_max_deg', 'height_max_fraction', 'vggt_predictions'}


def load_config(path: str | Path) -> dict:
    root = project_root()
    path = resolve_path(str(path), root)
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or set(config) - {'paths', 'run'}:
        raise ValueError('Configuration must contain only paths and run sections')
    paths, run = config.get('paths', {}), config.get('run', {})
    if not isinstance(paths, dict) or set(paths) != PATH_FIELDS:
        raise ValueError(f'paths must contain exactly: {sorted(PATH_FIELDS)}')
    if not isinstance(run, dict) or set(run) - RUN_FIELDS:
        raise ValueError(f'Unknown run options; allowed: {sorted(RUN_FIELDS)}')
    for name, value in paths.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'paths.{name} must be a nonempty path')
    resolved = {name: resolve_path(value, root) for name, value in paths.items()}
    defaults = dict(datasets=['Rural_Day_Curve_H150_P30_S01'], max_windows=0,
                    pitch_max_deg=0.0, height_max_fraction=0.0, vggt_predictions=False)
    defaults.update(run)
    validate_run(defaults)
    return dict(paths=resolved, run=defaults, config_path=path)


def validate_run(run: dict) -> None:
    names = run['datasets']
    if not isinstance(names, list) or any(not isinstance(n, str) or not n.strip() or n in {'.', '..'} or any(c in n for c in '/\\:') for n in names):
        raise ValueError('datasets must be a list of trajectory folder names, not paths')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate trajectory names are not allowed')
    if type(run['max_windows']) is not int or run['max_windows'] < 0:
        raise ValueError('max_windows must be a nonnegative integer')
    for name in ('pitch_max_deg', 'height_max_fraction'):
        if isinstance(run[name], bool) or not isinstance(run[name], (int, float)) or not math.isfinite(run[name]) or run[name] < 0:
            raise ValueError(f'{name} must be a finite nonnegative number')
    if type(run['vggt_predictions']) is not bool:
        raise ValueError('vggt_predictions must be a Boolean')
    if run['vggt_predictions'] and (run['pitch_max_deg'] or run['height_max_fraction']):
        raise ValueError('VGGT predicted pitch/height cannot be combined with metadata perturbations')


def check_assets(config: dict) -> list[str]:
    paths, run = config['paths'], config['run']
    errors = []
    for name in ('reference_map', 'vggt_weights', 'minima_weights', 'dinov2_weights', 'region_config'):
        if not paths[name].is_file():
            errors.append(f'Missing {name}: {paths[name]}')
    # DSM is documented but not read by the default fast pipeline.
    if not paths['dataset_root'].is_dir():
        errors.append(f'Missing dataset_root: {paths["dataset_root"]}')
        return errors
    names = run['datasets'] or sorted(p.name for p in paths['dataset_root'].iterdir() if p.is_dir())
    if not names:
        errors.append('No trajectory directories found')
    for name in names:
        sequence = paths['dataset_root'] / name
        if not (sequence / 'Thermal_info.csv').is_file():
            errors.append(f'Missing trajectory metadata: {name}/Thermal_info.csv')
        images = sequence / 'Thermal'
        if not images.is_dir() or not any(p.suffix in {'.JPG', '.jpg', '.png'} for p in images.iterdir() if p.is_file()):
            errors.append(f'Missing thermal images: {name}/Thermal/')
    for folder in ('third_party/vggt/vggt', 'third_party/minima/third_party/RoMa_minima/romatch'):
        if not (project_root() / folder).is_dir():
            errors.append(f'Missing vendored code: {folder}; keep the complete source checkout')
    return errors
