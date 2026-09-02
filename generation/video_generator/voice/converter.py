"""Seed-VC voice conversion wrapper.

Calls Seed-VC inference as a subprocess for robustness.
Falls back to a no-op copy if Seed-VC is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import os
import sys
from pathlib import Path


class VoiceConverter:
    """Zero-shot voice conversion using Seed-VC."""

    def __init__(
        self,
        seed_vc_dir: str | Path = "external/seed-vc",
        device: str = "cuda:0",
        diffusion_steps: int = 25,
        length_adjust: float = 1.0,
        inference_cfg_rate: float = 0.7,
        python_executable: str | Path | None = None,
        allow_fallback: bool = True,
    ) -> None:
        self.seed_vc_dir = Path(seed_vc_dir).expanduser().resolve()
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.length_adjust = length_adjust
        self.inference_cfg_rate = inference_cfg_rate
        self.python_executable = str(python_executable or sys.executable)
        self.allow_fallback = allow_fallback
        self._available = None

    def is_available(self) -> bool:
        """Check if Seed-VC is installed and usable."""
        if self._available is not None:
            return self._available

        inference_script = self.seed_vc_dir / "inference.py"
        if not inference_script.exists():
            message = (
                f"Seed-VC not found at {self.seed_vc_dir}. "
                "Install Seed-VC and set voice.seed_vc_dir or SEED_VC_DIR to its directory."
            )
            print(f"⚠️  {message}")
            self._available = False
            return False

        self._available = True
        return True

    def convert(
        self,
        source_path: str | Path,
        reference_path: str | Path,
        output_path: str | Path,
    ) -> Path:
        """
        Convert voice in source audio to match the reference voice.

        Args:
            source_path: Path to the source audio segment.
            reference_path: Path to the reference voice audio.
            output_path: Path to save the converted audio.

        Returns:
            Path to the converted audio file.
        """
        source_path = Path(source_path)
        reference_path = Path(reference_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.is_available():
            if self.allow_fallback:
                print(f"   ↩️  Seed-VC not available, copying source as-is: {source_path.name}")
                shutil.copyfile(source_path, output_path)
                return output_path
            raise RuntimeError(f"Seed-VC is not available at {self.seed_vc_dir}")

        # Seed-VC outputs to a directory; we need to extract the result
        temp_output_dir = output_path.parent / f"_seedvc_tmp_{output_path.stem}"
        temp_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            self.python_executable, str(self.seed_vc_dir / "inference.py"),
            "--source", str(source_path),
            "--target", str(reference_path),
            "--output", str(temp_output_dir),
            "--diffusion-steps", str(self.diffusion_steps),
            "--length-adjust", str(self.length_adjust),
            "--inference-cfg-rate", str(self.inference_cfg_rate),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=str(self.seed_vc_dir),
            env=self._subprocess_env(),
        )

        if result.returncode != 0:
            message = (
                f"Seed-VC conversion failed for {source_path.name}: "
                f"{result.stderr[:500] or result.stdout[:500]}"
            )
            print(f"⚠️  {message}")
            if self.allow_fallback:
                shutil.copyfile(source_path, output_path)
                return output_path
            raise RuntimeError(message)

        # Find the output file (Seed-VC saves as .wav in output dir)
        output_files = list(temp_output_dir.glob("*.wav"))
        if not output_files:
            output_files = list(temp_output_dir.glob("*.flac"))
        if not output_files:
            output_files = list(temp_output_dir.glob("*.*"))

        if output_files:
            shutil.move(str(output_files[0]), str(output_path))
            print(f"   ✅ Converted: {source_path.name} → {output_path.name}")
        else:
            message = f"No output from Seed-VC for {source_path.name}"
            print(f"   ⚠️  {message}")
            if self.allow_fallback:
                shutil.copyfile(source_path, output_path)
            else:
                raise RuntimeError(message)

        # Clean up temp dir
        shutil.rmtree(temp_output_dir, ignore_errors=True)

        return output_path

    def _subprocess_env(self) -> dict[str, str]:
        env = dict(os.environ)
        device = str(self.device).strip().lower()
        env["PYTHONNOUSERSITE"] = "1"
        if device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        elif device.startswith("cuda:"):
            env["CUDA_VISIBLE_DEVICES"] = device.split(":", 1)[1]
        return env

    def convert_batch(
        self,
        segments: list[tuple[Path, Path, Path]],
    ) -> list[Path]:
        """
        Convert a batch of voice segments.

        Args:
            segments: List of (source_path, reference_path, output_path) tuples.

        Returns:
            List of output paths.
        """
        results = []
        for i, (source, reference, output) in enumerate(segments, 1):
            print(f"🔄 Voice conversion [{i}/{len(segments)}]")
            result = self.convert(source, reference, output)
            results.append(result)
        return results
