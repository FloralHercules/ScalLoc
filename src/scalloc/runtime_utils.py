"""Runtime_Utils from the validated fast ScalLoc pipeline."""
from __future__ import annotations
import glob
import importlib.util
import io
import json
import math
import os
import random
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from contextlib import redirect_stdout
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2 ** 40)
os.environ.setdefault("PYTHONHASHSEED", "42")
import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRotation
from tqdm.auto import tqdm
from .configuration import *


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def template_match_optimizer_path() -> Path:
    preferred = [
        path for path in sorted(ROOT.glob("template_match_optimizer-*.py"))
        if "back" not in path.stem.lower()
    ]
    if preferred:
        return preferred[0]
    fallback = ROOT / "template_match_optimizer.py"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("Cannot find template_match_optimizer module")
