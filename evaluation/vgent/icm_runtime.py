"""Load the pinned upstream Vgent checkout and register ICM-Bench backends."""

from __future__ import annotations

import os
import sys
from pathlib import Path


INTEGRATION_ROOT = Path(__file__).resolve().parent


def upstream_root() -> Path:
    configured = os.environ.get("VGENT_ROOT")
    root = Path(configured).expanduser().resolve() if configured else INTEGRATION_ROOT / "upstream"
    if not (root / "utils" / "vgent.py").is_file():
        raise RuntimeError(
            f"Vgent upstream checkout not found at {root}. "
            "Run `git submodule update --init evaluation/vgent/upstream` or set VGENT_ROOT."
        )
    return root


def activate_upstream() -> Path:
    root = upstream_root()
    for path in (str(INTEGRATION_ROOT), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    return root


def load_vgent_class():
    activate_upstream()
    from utils import vgent as upstream_vgent

    upstream_vgent.MODEL_MAP.update(
        {
            "qwen35_9b": (
                "icm_backends.qwenvl",
                os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-9B"),
            ),
            "qwen3vl_8b": (
                "icm_backends.qwenvl",
                os.environ.get("QWEN3VL_MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct"),
            ),
        }
    )
    return upstream_vgent.Vgent
