"""Audit source and approved documentation figures without creating an archive."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'validation', 'outputs', 'Result', '.git', '__pycache__', '.pytest_cache',
            '.ruff_cache', '.venv', 'venv', 'dist', 'build'}
FORBIDDEN = {'.pth', '.pt', '.ckpt', '.safetensors', '.tif', '.tiff', '.jpg', '.jpeg',
             '.png', '.gif', '.mp4', '.npz', '.npy', '.pkl', '.zip', '.tar', '.gz'}
DOCUMENTATION_ASSETS = {
    'assets/ScalLoc.png': {
        'bytes': 6_342_656,
        'sha256': '9634b54751cf79dd0d7b4726231c8166c5525a3955678727ba03a84200001376',
    },
    'assets/Thermal-Oblique.png': {
        'bytes': 9_962_132,
        'sha256': '131d77af24da69ac224cdd8f34950da7e8a1578b8c2c4b357eb1455bcbec91a2',
    },
}
SECRET_PATTERNS = [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                   r'\bhf_[A-Za-z0-9]{25,}\b', r'\bgh[pousr]_[A-Za-z0-9]{30,}\b']


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED or part.endswith('.egg-info') for part in relative.parts):
        return False
    if relative.parts[0] in {'data', 'weights'} and relative.as_posix() not in {'data/README.md', 'weights/README.md'}:
        return False
    if relative.parts[0] == 'configs' and path.name.startswith('local') and path.name != 'local.example.yaml':
        return False
    return path.suffix not in {'.pyc', '.pyo', '.log'} and not path.name.startswith('.env')


def validate_file(relative_path: str, data: bytes) -> list[str]:
    """Exempt only the two byte-identical public figures from text-only rules."""
    expected = DOCUMENTATION_ASSETS.get(relative_path)
    if (expected is not None
            and len(data) == expected['bytes']
            and data.startswith(b'\x89PNG\r\n\x1a\n')
            and hashlib.sha256(data).hexdigest() == expected['sha256']):
        return []

    errors = []
    if expected is not None:
        errors.append(f'Documentation figure does not match approved bytes: {relative_path}')
    if Path(relative_path).suffix.lower() in FORBIDDEN or len(data) > 2_000_000:
        errors.append(f'Unexpected binary/large asset: {relative_path}')
    try:
        text = data.decode('utf-8-sig')
    except UnicodeDecodeError:
        errors.append(f'Unexpected non-text file: {relative_path}')
        text = ''
    if any(re.search(pattern, text) for pattern in SECRET_PATTERNS):
        errors.append(f'Potential secret (value suppressed): {relative_path}')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Optional JSON list; keep local audit artifacts outside the upload set')
    args = parser.parse_args()
    files, errors = [], []
    for path in sorted(ROOT.rglob('*')):
        if not included(path):
            continue
        if path.is_symlink():
            errors.append(f'Symlink is not upload-safe: {path.relative_to(ROOT)}')
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        errors.extend(validate_file(rel, data))
        files.append(dict(path=rel, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    result = dict(passed=not errors, file_count=len(files), total_bytes=sum(f['bytes'] for f in files),
                  errors=errors, excludes=sorted(EXCLUDED), files=files)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='files'}, indent=2))
    return 0 if not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())
