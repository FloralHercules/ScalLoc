"""Lightweight release checks; no GPU, dataset, weights or inference required."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import pytest
from scalloc.assets import load_config, validate_run, check_assets
from scalloc.paths import project_root, resolve_path

ROOT = project_root()


def test_default_protocol():
    config = load_config('configs/default.yaml')
    assert config['run']['datasets'] == ['Rural_Day_Curve_H150_P30_S01']
    assert config['run']['max_windows'] == 0
    assert config['run']['pitch_max_deg'] == 0
    assert config['run']['height_max_fraction'] == 0
    assert config['run']['vggt_predictions'] is False
    assert 'Thermal-Oblique' in str(config['paths']['dataset_root'])


def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert resolve_path('configs/default.yaml') == ROOT/'configs/default.yaml'
    assert load_config('configs/default.yaml')['paths']['output_root'] == ROOT/'outputs'


@pytest.mark.parametrize('field,value', [('max_windows',-1), ('max_windows',1.5), ('pitch_max_deg',float('nan')),
    ('height_max_fraction',-0.1), ('vggt_predictions','false'), ('datasets',['../escape']), ('datasets',['x','x'])])
def test_invalid_run_options(field,value):
    run=load_config('configs/default.yaml')['run']
    run[field]=value
    with pytest.raises(ValueError): validate_run(run)


def test_conflicting_perturbation():
    run=load_config('configs/default.yaml')['run']
    run.update(vggt_predictions=True,pitch_max_deg=5)
    with pytest.raises(ValueError): validate_run(run)


def test_help_does_not_load_models(tmp_path):
    result=subprocess.run([sys.executable,'-B',str(ROOT/'run.py'),'--help'],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0
    assert 'Thermal-Oblique' in result.stdout
    assert not list(tmp_path.iterdir())


def test_missing_assets_are_explicit(tmp_path):
    config=load_config('configs/default.yaml')
    config['paths']['dataset_root']=tmp_path/'missing'
    assert any('dataset_root' in error for error in check_assets(config))


def test_all_python_parses():
    for directory in ('src','tools','third_party'):
        for path in (ROOT/directory).rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))


def test_original_algorithm_asts_are_unchanged():
    manifest=json.loads((ROOT/'provenance/refactor_manifest.json').read_text())
    observed={}
    for module in ('geometry','matching','reporting','runtime_utils','pipeline'):
        tree=ast.parse((ROOT/f'src/scalloc/{module}.py').read_text(encoding='utf-8'))
        observed.update({n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))})
    assert len(manifest['functions'])==104
    assert set(observed)==set(manifest['functions'])
    count=0
    for name,entry in manifest['functions'].items():
        if entry['path_adapter']: continue
        digest=hashlib.sha256(ast.dump(observed[name],include_attributes=False).encode()).hexdigest()
        assert digest==entry['original_ast_sha256'], name
        count+=1
    assert count==100


def test_vendored_code_is_unmodified():
    manifest=json.loads((ROOT/'provenance/refactor_manifest.json').read_text())
    for record in manifest['vendor_files']:
        assert hashlib.sha256((ROOT/record['destination']).read_bytes()).hexdigest()==record['sha256']


def test_no_models_or_dataset_copied():
    forbidden={'.pt','.pth','.ckpt','.safetensors','.tif','.tiff','.jpg','.jpeg','.png','.mp4'}
    for folder in ('src','third_party','data','weights'):
        assert not [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix.lower() in forbidden]


def test_source_layout_has_no_original_project_import():
    for path in (ROOT/'src').rglob('*.py'):
        text=path.read_text(encoding='utf-8')
        # Reject references to the old checkout on any drive or platform.
        assert 'ScalLoc_最终版本' not in text


@pytest.mark.parametrize('error', [SystemExit(1), RuntimeError('incompatible weights')])
def test_model_load_failure_is_recorded(tmp_path, monkeypatch, error):
    import scalloc
    from scalloc import cli
    from types import ModuleType

    fake_pipeline = ModuleType('scalloc.pipeline')
    fake_pipeline.discover_batch_datasets = lambda: ['Rural_Day_Curve_H150_P30_S01']

    def fail_loading(_output):
        raise error

    fake_pipeline.create_runtime_context = fail_loading

    def install_fake_runtime(_config, _region):
        # Install only after the CLI has checked its one-runtime-per-process
        # guard, without loading torch or any actual model.
        monkeypatch.setitem(sys.modules, 'scalloc.pipeline', fake_pipeline)
        monkeypatch.setattr(scalloc, 'pipeline', fake_pipeline, raising=False)

    monkeypatch.setattr(cli, 'check_assets', lambda _config: [])
    monkeypatch.setattr(cli, 'configure_environment', install_fake_runtime)
    output = tmp_path/'failed_run'
    assert cli.main(['--output-dir', str(output)]) == 1
    status = json.loads((output/'run_metadata.json').read_text(encoding='utf-8'))
    assert status['state'] == 'failed'
    assert status['error'] == repr(error)
    assert status['elapsed_seconds'] >= 0
