"""Repository-relative resource paths, independent of the caller's directory."""
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(value: str, root: Path | None = None) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else ((root or project_root()) / path).resolve()
