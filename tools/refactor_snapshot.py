"""Mechanically split the validated fast snapshot; copy code, never assets."""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source, dest = args.source_root.resolve(), args.destination.resolve()
    if source == dest or source in dest.parents:
        raise ValueError('Destination must be separate from source')
    package = dest / 'src/scalloc'
    if (package / 'pipeline.py').exists():
        raise FileExistsError('Refusing to overwrite existing refactor')
    package.mkdir(parents=True, exist_ok=True)
    origin = source / 'ScalLoc1.py'
    raw = origin.read_text(encoding='utf-8-sig')
    lines = raw.splitlines(keepends=True)
    definitions = {n.name:n for n in ast.parse(raw).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    geometry_extra = {'yaw_from_forward','interpolate_sim2_per_frame','apply_framewise_sim2',
        'yaw_from_forward_framewise','camera_to_body_rotation_opencv','body_to_gt_ypr','vggt_window_ypr','circular_weighted_mean'}
    report_extra = {'save_batch_reports','record_stage_timing','build_stage_timing_rows','aggregate_per_frame_metrics','flatten_sequence_metric'}
    match_extra = {'make_bev_rescue_snapshot','compare_bev_rescue_snapshots','build_middle_frame_pitch_crop_prior'}
    groups = {k:('geometry' if 494<=n.lineno<=1399 or k in geometry_extra else
        'reporting' if 4149<=n.lineno<=4764 or k in report_extra else
        'matching' if 1400<=n.lineno<=2590 or 2762<=n.lineno<=3872 or k in match_extra else
        'runtime_utils' if k in {'load_module_from_path','template_match_optimizer_path'} else 'pipeline')
        for k,n in definitions.items()}
    def replace_once(text,before,after):
        if text.count(before)!=1: raise ValueError(f'Expected one path adapter: {before}')
        return text.replace(before,after)
    def extract(node):
        start=min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
        return ''.join(lines[start-1:node.end_lineno]).rstrip()+'\n'
    config=''.join(lines[36:410])
    config=replace_once(config,'ROOT = Path(__file__).resolve().parent',
        'ROOT = project_root()\nPACKAGE_ROOT = Path(__file__).resolve().parent\nTHIRD_PARTY_ROOT = ROOT / "third_party"\nREGION_CONFIG_PATH = Path(os.environ.get("SCALLOC_REGION_CONFIG", str(ROOT / "configs/regions/changsha.yaml")))')
    config=config.replace('"Thermal-UAV-v2"','"Thermal-Oblique"')
    config='\n'.join(l for l in config.splitlines() if not l.startswith('# DATASET_NAME = ') and not l.startswith('# BATCH_DATASET_NAMES = '))+'\n'
    (package/'configuration.py').write_text('"""Original fast settings; all numerical defaults are preserved."""\n'
        'from __future__ import annotations\nimport os\nimport json\nfrom pathlib import Path\nfrom .paths import project_root\n\n'+config,encoding='utf-8')
    common=('from __future__ import annotations\nimport glob\nimport importlib.util\nimport io\nimport json\nimport math\n'
        'import os\nimport random\nimport sys\nimport tempfile\nimport time\nfrom dataclasses import dataclass, field\n'
        'from datetime import datetime\nfrom pathlib import Path\nfrom types import SimpleNamespace\nfrom typing import Any\n'
        'from contextlib import redirect_stdout\nos.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)\n'
        'os.environ.setdefault("PYTHONHASHSEED", "42")\nimport cv2\nimport numpy as np\nimport pandas as pd\nimport torch\nimport yaml\n'
        'from scipy.spatial import cKDTree\nfrom scipy.spatial.transform import Rotation as SciRotation\nfrom tqdm.auto import tqdm\n'
        'from .configuration import *\n')
    adaptations,functions={},{}
    for name,node in definitions.items():
        body=extract(node)
        adapted=body.replace('from Matching_Models.MINIMA.MINIMA_match1 import ','from scalloc.matchers.minima import ')
        if name=='load_vggt_once':
            adapted=replace_once(adapted,'str(ROOT / "vggt")','str(THIRD_PARTY_ROOT / "vggt")')
        if name=='create_runtime_context':
            adapted=replace_once(adapted,'ROOT / "绝了卧槽_MINIMA_SE2.py"','PACKAGE_ROOT / "geometry_io.py"')
            adapted=replace_once(adapted,'ROOT / "参考" / "vggt_reconstructor.py"','PACKAGE_ROOT / "reconstruction.py"')
            adapted=replace_once(adapted,'ROOT / "data" / "Regions_params" / f"{REGION_NAME}.yaml"','REGION_CONFIG_PATH')
        if ast.dump(ast.parse(body),include_attributes=False)!=ast.dump(ast.parse(adapted),include_attributes=False):
            adaptations[name]='Only import/resource paths were adapted.'
        functions[name]=adapted
    for group in ('geometry','reporting','runtime_utils','matching','pipeline'):
        members=[k for k in definitions if groups[k]==group]
        dependencies={}
        for name in members:
            used={n.id for n in ast.walk(definitions[name]) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load)}
            for other in used & definitions.keys():
                if groups[other]!=group: dependencies.setdefault(groups[other],set()).add(other)
        imports=''.join('from .'+mod+' import ('+', '.join(sorted(names))+')\n' for mod,names in sorted(dependencies.items()))
        content=f'"""{group.title()} from the validated fast ScalLoc pipeline."""\n'+common+imports
        if group=='pipeline': content+='from .image_io import install_windows_unicode_io\ninstall_windows_unicode_io()\n'
        content+='\n\n'+'\n\n'.join(functions[name] for name in members)
        if group=='pipeline': content+='\n\npipeline_start = time.perf_counter()\n'
        ast.parse(content)
        (package/f'{group}.py').write_text(content,encoding='utf-8')
    copies={'参考/vggt_reconstructor.py':'src/scalloc/reconstruction.py','utils.py':'src/scalloc/map_utils.py',
        'local_image_io.py':'src/scalloc/image_io.py','绝了卧槽_MINIMA_SE2.py':'src/scalloc/geometry_io.py'}
    for old,new in copies.items():
        content=(source/old).read_text(encoding='utf-8-sig')
        content=content.replace('from utils import (','from scalloc.map_utils import (')
        content=content.replace('from Matching_Models.MINIMA.MINIMA_match1 import ','from scalloc.matchers.minima import ')
        (dest/new).write_text(content,encoding='utf-8')
    vendor=[]
    def copy_text(path,target):
        if path.suffix.lower() not in {'.py','.yaml','.yml','.toml','.md','.txt','.json'} and not path.name.upper().startswith(('LICENSE','NOTICE')): return
        if any(p in {'__pycache__','.git','.pytest_cache','weights','data','assets','tests','benchmarks','experiments','docs','training'} for p in path.relative_to(source).parts): return
        if path.stat().st_size>2_000_000: raise ValueError(f'Unexpected large code file: {path}')
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        vendor.append(dict(source=path.relative_to(source).as_posix(),destination=target.relative_to(dest).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for old,new in [('vggt/vggt','third_party/vggt/vggt'),('Matching_Models/MINIMA/third_party/RoMa_minima/romatch','third_party/minima/third_party/RoMa_minima/romatch')]:
        for path in sorted((source/old).rglob('*')):
            if path.is_file(): copy_text(path,dest/new/path.relative_to(source/old))
    for old,new in [
        ('vggt/LICENSE.txt','third_party/vggt/LICENSE.txt'),('vggt/README.md','third_party/vggt/README.upstream.md'),
        ('Matching_Models/MINIMA/LICENSE','third_party/minima/LICENSE'),('Matching_Models/MINIMA/README.md','third_party/minima/README.upstream.md'),
        ('Matching_Models/MINIMA/load_model.py','third_party/minima/load_model.py'),
        ('Matching_Models/MINIMA/src/config/default.py','third_party/minima/src/config/default.py'),
        ('Matching_Models/MINIMA/src/utils/data_io_roma.py','third_party/minima/src/utils/data_io_roma.py'),
        ('Matching_Models/MINIMA/third_party/RoMa_minima/LICENSE','third_party/minima/third_party/RoMa_minima/LICENSE'),
        ('Matching_Models/MINIMA/third_party/RoMa_minima/README.md','third_party/minima/third_party/RoMa_minima/README.upstream.md'),
        ('DINOv2/LICENSE','third_party/minima/third_party/RoMa_minima/LICENSE.DINOv2')]: copy_text(source/old,dest/new)
    for relative in ['__init__.py','src/__init__.py','src/config/__init__.py','src/utils/__init__.py','third_party/__init__.py','third_party/RoMa_minima/__init__.py']:
        path=source/'Matching_Models/MINIMA'/relative
        if path.is_file(): copy_text(path,dest/'third_party/minima'/relative)
    provenance=dict(source_entrypoint='ScalLoc1.py',source_sha256=hashlib.sha256(origin.read_bytes()).hexdigest(),dataset_public_name='Thermal-Oblique',
        functions={k:dict(module=groups[k],original_ast_sha256=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest(),path_adapter=k in adaptations) for k,n in definitions.items()},
        path_adapters=adaptations,support_files={new:hashlib.sha256((source/old).read_bytes()).hexdigest() for old,new in copies.items()},vendor_files=vendor,weights_or_dataset_copied=False)
    (dest/'provenance').mkdir(exist_ok=True)
    (dest/'provenance/refactor_manifest.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print(json.dumps(dict(definitions=len(definitions),unchanged_definition_asts=len(definitions)-len(adaptations),path_adapters=list(adaptations),vendor_files=len(vendor)),indent=2))


if __name__=='__main__': main()
