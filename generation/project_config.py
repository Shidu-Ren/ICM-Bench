from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from datetime import datetime

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_VIDEO_CONFIG_PATH = CONFIGS_DIR / "video_config_pilot.yaml"
DEFAULT_API_CONFIG_PATH = CONFIGS_DIR / "api_config.json"
DEFAULT_OUTPUT_BASE_DIR = PROJECT_ROOT / "output"
DEFAULT_VIDEO_OUTPUT_BASE_DIR = DEFAULT_OUTPUT_BASE_DIR / "video_runs"

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON config must be an object: {path}")

    return data


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")

    return data


def load_api_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_API_CONFIG_PATH
    return _read_json(path)


def load_video_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_VIDEO_CONFIG_PATH
    return _read_yaml_with_extends(path)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml_with_extends(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve()
    seen = seen or set()
    if path in seen:
        raise ValueError(f"Cyclic YAML extends detected: {path}")
    seen.add(path)

    data = _read_yaml(path)
    extends = data.pop("extends", None)
    if not extends:
        return data

    if isinstance(extends, (str, os.PathLike)):
        base_paths = [extends]
    elif isinstance(extends, list):
        base_paths = extends
    else:
        raise ValueError(f"YAML extends must be a string or list: {path}")

    merged: dict[str, Any] = {}
    for base_ref in base_paths:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        merged = _deep_merge(merged, _read_yaml_with_extends(base_path, seen))
    return _deep_merge(merged, data)


def slugify(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return cleaned or "album"


def ensure_unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    index = 2
    while True:
        candidate = path.parent / f"{path.name}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def build_staging_output_root(base_dir: Path | None = None) -> Path:
    base = base_dir or DEFAULT_OUTPUT_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_unique_path(base / f"_staging_{timestamp}")


def build_named_output_root(protagonist_name: str, base_dir: Path | None = None) -> Path:
    base = base_dir or DEFAULT_OUTPUT_BASE_DIR
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = slugify(protagonist_name)
    return ensure_unique_path(base / f"{slug}_{timestamp}")


def build_stable_named_output_root(name: str, base_dir: Path | None = None) -> Path:
    base = base_dir or DEFAULT_OUTPUT_BASE_DIR
    slug = slugify(name)
    return base / slug


def find_latest_output_root(
    base_dir: Path | None = None,
    marker_filename: str = "01_protagonist.json",
) -> Path:
    base = base_dir or DEFAULT_OUTPUT_BASE_DIR
    latest_path: Path | None = None
    latest_mtime = -1.0

    if not base.exists():
        raise FileNotFoundError(f"Output base directory does not exist: {base}")

    # 兼容新的 output/<run_name>/metadata 结构
    for child in base.iterdir():
        metadata_file = child / "metadata" / marker_filename
        if child.is_dir() and metadata_file.exists():
            mtime = metadata_file.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime = mtime
                latest_path = child

    # 兼容旧的 output/metadata 扁平结构
    legacy_metadata = base / "metadata" / marker_filename
    if legacy_metadata.exists() and legacy_metadata.stat().st_mtime > latest_mtime:
        latest_path = base

    if latest_path is None:
        raise FileNotFoundError(f"No generated output run found under: {base}")

    return latest_path


def get_google_api_key(config_path: str | os.PathLike[str] | None = None) -> str | None:
    """Return the Google AI API key from the environment.

    ``config_path`` remains in the signature for compatibility with existing
    callers, but secret keys are intentionally never loaded from files.
    """

    del config_path
    value = os.environ.get("GOOGLE_API_KEY", "").strip()
    return value or None


def _get_model_setting(
    env_name: str,
    config_key: str,
    config_path: str | os.PathLike[str] | None = None,
) -> str | None:
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    value = load_api_config(config_path).get(config_key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def get_image_model(config_path: str | os.PathLike[str] | None = None) -> str | None:
    return _get_model_setting("ICM_IMAGE_MODEL", "image_model", config_path)


def get_text_model(config_path: str | os.PathLike[str] | None = None) -> str | None:
    return _get_model_setting("ICM_TEXT_MODEL", "text_model", config_path)


def get_video_api_key(config_path: str | os.PathLike[str] | None = None) -> str | None:
    return get_google_api_key(config_path)


def get_video_model(config_path: str | os.PathLike[str] | None = None) -> str | None:
    return _get_model_setting("ICM_VIDEO_MODEL", "video_model", config_path)
