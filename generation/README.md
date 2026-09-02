# Long-form Video Generation Pipeline

This directory contains the public synthetic-video production workflow used to plan and render recurring-cast, life-album-style video series. It covers:

- theme, timeline, cast, scene, wardrobe, clip, and shot planning;
- reusable character, scene, and shot-anchor image generation;
- Veo clip rendering;
- optional Gemini TTS redubbing and subtitle generation;
- local human shot review and filtered video/metadata export.

## Requirements

- Python 3.11 or 3.12
- FFmpeg on `PATH`
- a Google AI API key with access to the configured Gemini image/text models and Veo

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export GOOGLE_API_KEY="your-google-ai-api-key"
```

The pipeline reads API credentials only from `GOOGLE_API_KEY`. Optional model overrides are:

| Environment variable | Purpose |
| --- | --- |
| `ICM_TEXT_MODEL` | Default planning model when a YAML stage override is absent |
| `ICM_IMAGE_MODEL` | Character, scene, and shot-anchor image model |
| `ICM_VIDEO_MODEL` | Veo rendering model |
| `SEED_VC_DIR` | Optional Seed-VC checkout used by voice conversion |

Model names can instead be stored in a local `configs/api_config.json` copied from `configs/api_config.example.json`. That JSON contains model names only; do not put secrets in it. Environment model overrides take precedence.

## Quick start

Validate the small pilot configuration without calling an API:

```bash
python -m video_generator.pipeline \
  --config configs/video_config_pilot.yaml \
  --preflight-only
```

Generate planning metadata only:

```bash
python -m video_generator.pipeline \
  --config configs/video_config_pilot.yaml \
  --metadata-only
```

Run planning, anchors, rendering, and any voice stage enabled by the selected YAML:

```bash
python -m video_generator.pipeline \
  --config configs/video_config_pilot.yaml
```

Useful controls:

- `--batch-outline-only`: stop after global batch planning.
- `--anchors-only`: stop after reusable assets and shot anchors.
- `--skip-voice`: render Veo clips without voice post-processing.
- `--skip-planning --output-root PATH`: reuse an existing run.
- `--include-clips clip_001,clip_002`: render or post-process a subset.
- `--resume-from-output PATH --resume-stage clip-outline|shot-blueprint|shot-plan`: resume planning from saved metadata.

Longer topic presets live in `configs/video_series_presets/`.

## Outputs

Runs are written below `output/video_runs/`. A run contains:

- `metadata/`: validated planning JSON and API-usage summaries;
- `assets/`: character, scene, and shot-anchor images;
- `renders/`: rendered and optionally voiced clips;
- `voice_work/` and `subtitles/`: optional TTS manifests and subtitles.

## TTS and subtitles

Voice processing is controlled by the `voice` section of the selected YAML. To rerun a clip range:

Install the optional voice dependencies first:

```bash
python -m pip install -r requirements-voice.txt
```

```bash
python scripts/rerun_tts_batches.py \
  --output-root output/video_runs/YOUR_RUN \
  --start 1 --end 8
```

Generate SRT files and review copies from existing TTS manifests:

```bash
python scripts/generate_tts_subtitles.py \
  --output-root output/video_runs/YOUR_RUN \
  --clips clip_001-clip_008
```

Seed-VC-based conversion is optional and requires its own local installation. Point to it with `SEED_VC_DIR` or the YAML `voice.seed_vc_dir` setting.

## Human review and export

Initialize a review manifest and open the local browser UI:

```bash
python scripts/review_shots.py \
  --output-root output/video_runs/YOUR_RUN \
  init --clips clip_001-clip_008

python scripts/review_shots.py \
  --output-root output/video_runs/YOUR_RUN \
  serve
```

Export review-filtered metadata:

```bash
python scripts/review_shots.py \
  --output-root output/video_runs/YOUR_RUN \
  export --export-root output/releases/YOUR_RUN
```

To reassemble only passing shots into release clips:

```bash
python scripts/export_filtered_reassembled_clips.py \
  --output-root output/video_runs/YOUR_RUN \
  --export-root output/releases/YOUR_RUN \
  --start 1 --end 8
```

All review and export tools leave the source renders and source metadata unchanged.
