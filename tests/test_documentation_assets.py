"""Public figures are an exact-file exception, never a general binary allowlist."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('scalloc_release_audit', ROOT/'tools/check_release.py')
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
FIGURES = tuple(AUDIT.DOCUMENTATION_ASSETS)


@pytest.fixture(scope='module')
def figure_bytes():
    return {name: (ROOT/name).read_bytes() for name in FIGURES}


@pytest.mark.parametrize('name', FIGURES)
def test_approved_figure_is_accepted(name, figure_bytes):
    assert AUDIT.validate_file(name, figure_bytes[name]) == []


@pytest.mark.parametrize('name', FIGURES)
def test_same_size_modified_figure_is_rejected(name, figure_bytes):
    data = figure_bytes[name]
    changed = data[:-1] + bytes([data[-1] ^ 1])
    errors = AUDIT.validate_file(name, changed)
    assert any('does not match approved bytes' in error for error in errors)
    assert any('Unexpected binary/large asset' in error for error in errors)
    assert any('Unexpected non-text file' in error for error in errors)


@pytest.mark.parametrize('name', FIGURES)
def test_truncated_figure_is_rejected(name, figure_bytes):
    assert AUDIT.validate_file(name, figure_bytes[name][:-1])


@pytest.mark.parametrize('name', FIGURES)
def test_forged_figure_with_expected_size_is_rejected(name):
    data = b'\x89PNG\r\n\x1a\n' + b'x' * (AUDIT.DOCUMENTATION_ASSETS[name]['bytes'] - 8)
    assert AUDIT.validate_file(name, data)


def test_approved_bytes_at_another_path_are_rejected(figure_bytes):
    assert AUDIT.validate_file('assets/unapproved.png', figure_bytes[FIGURES[0]])


def test_other_png_and_binary_formats_remain_rejected():
    assert AUDIT.validate_file('assets/extra.png', b'\x89PNG\r\n\x1a\n')
    assert AUDIT.validate_file('assets/extra.bin', b'\xff\xfe\x80')
    assert AUDIT.validate_file('weights/model.pth', b'not a checkpoint')


def test_large_text_is_not_exempt():
    assert AUDIT.validate_file('assets/large.txt', b'x' * 2_000_001)


def test_small_utf8_text_is_accepted():
    assert AUDIT.validate_file('docs/example.md', 'ScalLoc 文档\n'.encode('utf-8')) == []


def test_image_filename_does_not_bypass_text_secret_checks():
    secret = ('hf_' + 'a' * 30).encode('ascii')
    errors = AUDIT.validate_file(FIGURES[0], secret)
    assert any('Potential secret' in error for error in errors)
