"""Allow the existing OpenCV calls to use Unicode paths on Windows."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np


def install_windows_unicode_io() -> None:
    if os.name != "nt" or getattr(cv2, "_scalloc_unicode_io", False):
        return

    def imread(filename, flags=cv2.IMREAD_COLOR):
        try:
            encoded = np.fromfile(os.fspath(filename), dtype=np.uint8)
        except OSError:
            return None
        if encoded.size == 0:
            return None
        return cv2.imdecode(encoded, flags)

    def imwrite(filename, image, params=None):
        path = os.fspath(filename)
        success, encoded = cv2.imencode(Path(path).suffix, image, params or [])
        if not success:
            return False
        try:
            encoded.tofile(path)
        except OSError:
            return False
        return True

    cv2.imread = imread
    cv2.imwrite = imwrite
    cv2._scalloc_unicode_io = True
