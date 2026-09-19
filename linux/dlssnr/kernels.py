"""GPU targets supported by the bundled native HIP network library."""
from __future__ import annotations

from pathlib import Path
import re

FALLBACK_TARGETS = frozenset(('gfx1201',))
_TARGET_RE = re.compile(rb'(?<![A-Za-z0-9])gfx[0-9a-f]{3,5}(?![0-9a-z])')


def bundled_targets(so_path) -> frozenset:
    """gfx targets embedded in the bundled libdlss5_hip.so kernel images."""
    try:
        data = Path(so_path).read_bytes()
    except OSError as e:
        raise RuntimeError(f'Cannot read bundled HIP library {so_path}: {e}') from e
    found = frozenset(target.decode('ascii') for target in _TARGET_RE.findall(data))
    return found or FALLBACK_TARGETS
