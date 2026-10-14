"""Audit upload contents; produce a code-only file list without creating an archive."""
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


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED or part.endswith('.egg-info') for part in relative.parts):
        return False
    if relative.parts[0] in {'data', 'weights'} and relative.as_posix() not in {'data/README.md', 'weights/README.md'}:
        return False
    if relative.parts[0] == 'configs' and path.name.startswith('local') and path.name != 'local.example.yaml':
        return False
    return path.suffix not in {'.pyc', '.pyo', '.log'} and not path.name.startswith('.env')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, help='Optional JSON list; keep local audit artifacts outside the upload set')
    args = parser.parse_args()
    files, errors = [], []
    secret_patterns = [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                       r'\bhf_[A-Za-z0-9]{25,}\b', r'\bgh[pousr]_[A-Za-z0-9]{30,}\b']
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
        if path.suffix.lower() in FORBIDDEN or len(data) > 2_000_000:
            errors.append(f'Unexpected binary/large asset: {rel}')
        try:
            text = data.decode('utf-8-sig')
        except UnicodeDecodeError:
            errors.append(f'Unexpected non-text file: {rel}')
            text = ''
        if any(re.search(pattern, text) for pattern in secret_patterns):
            errors.append(f'Potential secret (value suppressed): {rel}')
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
