import importlib
import sys
from pathlib import Path

INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INTEGRATION_ROOT))

from icm_runtime import activate_upstream, load_vgent_class

activate_upstream()


MODULES = [
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "flash_attn",
    "decord",
    "pysubs2",
    "pyarrow",
    "pandas",
    "networkx",
    "accelerate",
]


def main() -> int:
    for name in MODULES:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "ok")
        print(f"{name}: {version}")

    import torch

    print(f"python: {sys.version.split()[0]}")
    print(f"torch cuda: {torch.version.cuda}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu count: {torch.cuda.device_count()}")
        print(f"gpu0: {torch.cuda.get_device_name(0)}")

    from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    load_vgent_class()

    print("Vgent imports: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
