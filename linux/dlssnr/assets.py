"""Portable package integrity checks (stdlib, no downloads)."""
from __future__ import annotations

from pathlib import Path
import json
import re

from .package import sha256


def verify_assets(package_root) -> dict:
    """Verify every file listed in the release manifest.json against its SHA256."""
    root = Path(package_root)
    manifest_path = root / 'manifest.json'
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as e:
        raise RuntimeError(f'Cannot read package manifest: {e}') from e
    if type(manifest) is not dict:
        raise RuntimeError('Invalid package manifest')
    files = manifest.get('files')
    if type(files) is not dict or not files:
        raise RuntimeError('Invalid package manifest: empty files map')
    for name, expected in files.items():
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_./-]+', name)
                or name.startswith('/') or '..' in Path(name).parts):
            raise RuntimeError(f'Disallowed path in manifest: {name}')
        if (not isinstance(expected, dict) or not isinstance(expected.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', expected['sha256'])):
            raise RuntimeError(f'Invalid checksum in manifest: {name}')
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f'Missing/modified asset: {path}')
        if sha256(path) != expected['sha256']:
            raise RuntimeError(f'Missing/modified asset: {path}')
        if isinstance(expected.get('bytes'), int) and path.stat().st_size != expected['bytes']:
            raise RuntimeError(f'Size mismatch for asset: {path}')
    return manifest
