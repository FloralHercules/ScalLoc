"""Public command line; model loading occurs only after resource validation."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import yaml
from .assets import load_config, validate_run, check_assets
from .paths import project_root, resolve_path


def configure_environment(config: dict, region_file: Path) -> None:
    paths, run = config['paths'], config['run']
    values = {
        'THERMAL_UAV_DATASET_ROOT': paths['dataset_root'],
        'VGGT_WEIGHTS_PATH': paths['vggt_weights'], 'MINIMA_WEIGHTS_PATH': paths['minima_weights'],
        'DINOV2_WEIGHTS_PATH': paths['dinov2_weights'], 'SCALLOC_REGION_CONFIG': region_file,
        'BATCH_DATASET_NAMES': ','.join(run['datasets']), 'BATCH_MAX_DATASETS': 0,
        'VGGT_MAX_WINDOWS': run['max_windows'],
        'ENABLE_FRAME_PITCH_HEIGHT_PERTURBATION': int(not run['vggt_predictions'] and bool(run['pitch_max_deg'] or run['height_max_fraction'])),
        'PITCH_NOISE_MAX_ABS_DEG': run['pitch_max_deg'], 'HEIGHT_NOISE_MAX_ABS_FRACTION': run['height_max_fraction'],
        'USE_VGGT_PITCH_HEIGHT': int(run['vggt_predictions']),
        'SEQUENCE_METRIC_SCALE_MULTIPLIER': '1.0', 'SEQUENCE_METRIC_SCALE_MAP_PATH': '',
        'HF_HUB_OFFLINE': '1', 'MPLBACKEND': 'Agg', 'PYTHONDONTWRITEBYTECODE': '1',
    }
    os.environ.update({key: str(value) for key, value in values.items()})
    sys.dont_write_bytecode = True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description='ScalLoc fast pipeline for Thermal-Oblique')
    parser.add_argument('--config', default='configs/default.yaml')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--dataset', action='append', help='Trajectory folder name; repeat for multiple trajectories')
    group.add_argument('--all-datasets', action='store_true', help='Explicitly run every trajectory')
    parser.add_argument('--max-windows', type=int, default=None, help='0 = complete trajectories')
    parser.add_argument('--pitch-max-deg', type=float, default=None)
    parser.add_argument('--height-max-fraction', type=float, default=None)
    parser.add_argument('--vggt-predictions', action='store_true', default=None)
    parser.add_argument('--output-dir', help='New or empty directory; existing results are never replaced')
    parser.add_argument('--check', action='store_true', help='Validate resources without loading models or writing outputs')
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        for name in ('max_windows', 'pitch_max_deg', 'height_max_fraction', 'vggt_predictions'):
            if getattr(args, name) is not None:
                config['run'][name] = getattr(args, name)
        if args.dataset:
            config['run']['datasets'] = args.dataset
        elif args.all_datasets:
            config['run']['datasets'] = []
        validate_run(config['run'])
        errors = check_assets(config)
        if errors:
            raise ValueError('\n'.join(errors))
        if args.check:
            print('Resources verified. No models loaded; no experiment started.')
            return 0
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f'ScalLoc configuration error: {exc}', file=sys.stderr)
        return 2
    # One runtime per process: computational modules bind the frozen configuration.
    if 'scalloc.pipeline' in sys.modules:
        raise RuntimeError('Start a new Python process for each experiment')
    output = (resolve_path(args.output_dir) if args.output_dir else config['paths']['output_root'] /
              datetime.now().strftime('run_%Y%m%d_%H%M%S_%f'))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        print(f'Refusing to overwrite nonempty output: {output}', file=sys.stderr)
        return 2
    output.mkdir(parents=True, exist_ok=True)
    resolved_region = yaml.safe_load(config['paths']['region_config'].read_text(encoding='utf-8'))
    resolved_region['changsha_HIGH_REF_PATH'] = str(config['paths']['reference_map'])
    resolved_region['changsha_HIGH_DSM_PATH'] = str(config['paths']['dsm_map'])
    region_file = output / 'resolved_region.yaml'
    region_file.write_text(yaml.safe_dump(resolved_region, allow_unicode=True, sort_keys=False), encoding='utf-8')
    configure_environment(config, region_file)
    status = dict(state='running', training=False, dataset_name='Thermal-Oblique',
                  configuration={k: str(v) for k, v in config['paths'].items()}, run=config['run'],
                  source_sha256={str(p.relative_to(project_root())): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((project_root()/'src/scalloc').rglob('*.py'))})
    def save_status():
        (output/'run_metadata.json').write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding='utf-8')
    save_status()
    started = time.perf_counter()
    try:
        from . import pipeline
        # Explicit output paths retain every original computation while avoiding
        # old ROOT/Result and timestamp-dependent global output directories.
        names = pipeline.discover_batch_datasets()
        if not names or (config['run']['datasets'] and names != config['run']['datasets']):
            raise ValueError('Requested dataset list was not resolved exactly')
        (output/'batch_datasets.json').write_text(json.dumps(names, indent=2), encoding='utf-8')
        runtime = pipeline.create_runtime_context(output)
        results, failures = [], []
        for name in names:
            print(f'[ScalLoc] Starting {name}', flush=True)
            try:
                result = pipeline.run_pipeline(name, runtime, output)
                results.append(result)
                print(f'[ScalLoc] Completed {name}', flush=True)
            except Exception as exc:
                failures.append(dict(dataset_name=name, error=repr(exc)))
                traceback.print_exc()
        pipeline.save_batch_reports(output, results, failures)
        reports = sorted(output.glob('*/trajectory_metrics_summary.csv'))
        status.update(state='completed' if not failures and len(reports)==len(names) else 'failed',
                      datasets=names, completed_trajectories=len(reports), failures=failures)
    except (Exception, SystemExit) as exc:
        # The preserved MINIMA loader exits on incompatible weights. Persist
        # failure rather than leaving this run permanently marked as running.
        status.update(state='failed', error=repr(exc))
        traceback.print_exc()
    status['elapsed_seconds'] = time.perf_counter()-started
    save_status()
    print(f'[ScalLoc] {status["state"]}: {output}', flush=True)
    return 0 if status['state']=='completed' else 1
