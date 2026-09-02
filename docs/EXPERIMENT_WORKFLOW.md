# ICM-Bench public workflow

This document covers the released synthetic-video pipeline and the M3-Agent, Vgent, and HippoRAG2 evaluation integrations. Run each component in its own environment and pass local paths explicitly; checked-in examples do not assume a particular host, mount point, GPU index, or model directory.

## 1. Prepare the benchmark

```bash
cp .env.example .env
cp configs/paths.example.yaml configs/paths.local.yaml
set -a
source .env
set +a

hf download ryanren0330/ICM-Bench \
  --repo-type dataset \
  --local-dir "$ICM_BENCH_DATA_ROOT"
tar -xf "$ICM_BENCH_DATA_ROOT/videos.tar" -C "$ICM_BENCH_DATA_ROOT"
```

Confirm that `videos/clip_000.mp4` through `videos/clip_838.mp4` exist after extraction. Recall and Retrieval items may access clips only through their `before_clip`; Profile items use the full timeline. Keep all answer keys, evidence pointers, character metadata, and speaker-labeled reference transcripts outside model input.

## 2. Generate synthetic videos

The public generation path covers structured album planning, reusable character and scene anchors, shot rendering, voice/subtitle synthesis, and date-stamped video output:

```bash
cd generation
python -m pip install -r requirements.txt
PYTHONPATH="$PWD" python -m video_generator.pipeline \
  --config configs/video_config.yaml
```

Use a preset under `configs/video_series_presets/` when creating a different synthetic album. Generation calls external services and may incur cost; review the component README and run its preflight mode before a full render.

## 3. Evaluate M3-Agent

M3-Agent constructs a multimodal memory graph and then answers the open-ended benchmark questions through its control loop. The relevant public entry points are:

- `m3_agent/memorization_memory_graphs.py` for ordered memory construction;
- `m3_agent/control.py` for retrieval/reasoning and answer generation;
- `scripts/run_album_m3_original_strict_ordered.sh` as the orchestration template.

The launcher requires `ICM_BENCH_ROOT` and accepts portable overrides including `WORK_DIR`, `PY_BIN`, `GPU_LIST`, `SOURCE_DIR`, `ANNOTATION_IN`, `MEM_PATH`, `INTERMEDIATE_DIR`, `DATA_JSONL`, `ANNOTATION_OUT`, `RESULT_NAME`, and `LOG_DIR`. Keep the public clip order and enforce each question's cutoff when preparing the system-native annotation wrapper.

## 4. Evaluate Vgent

Initialize the pinned upstream checkout, create an album manifest, construct one graph per clip, and run the ICM-Bench open-ended retrieval/refinement adapter:

```bash
git submodule update --init evaluation/vgent/upstream
python -m pip install -r evaluation/vgent/requirements.txt
export VGENT_ROOT="$PWD/evaluation/vgent/upstream"

python evaluation/vgent/scripts/prepare_album_manifest.py \
  --source-dir "$ICM_BENCH_DATA_ROOT/videos" \
  --output-dir "$ICM_BENCH_OUTPUT_ROOT/vgent/manifest"

torchrun --standalone --nproc-per-node=1 \
  evaluation/vgent/scripts/build_album_graph.py \
  --model_name MODEL_NAME \
  --manifest_path "$ICM_BENCH_OUTPUT_ROOT/vgent/manifest/album_manifest.jsonl" \
  --graph_path "$ICM_BENCH_OUTPUT_ROOT/vgent/graphs"
```

If transcripts are used, populate each manifest row's `subtitle` field from `resources/asr_transcripts/` before graph construction. Never use `resources/transcripts_with_speakers/` in an evaluated run. The answer adapter is `scripts/eval_album_openqa_refine.py`. Pass the graph directory created above (normally `$ICM_BENCH_OUTPUT_ROOT/vgent/graphs/album_1.0fps_64`), the public manifest and QA file, and an output path; inspect `--help` for model-specific arguments. The adapter applies each non-Profile item's released `before_clip` cutoff.

## 5. Evaluate HippoRAG2

HippoRAG2 indexes a text corpus derived from the released speakerless transcripts and answers the same open-ended questions:

```bash
cd evaluation/hipporag2
python -m pip install -r requirements.txt
python -m pip install -e .

python album_tools/prepare_album_corpus_from_transcripts.py \
  --manifest MANIFEST_WITH_SPEAKERLESS_SRT_PATHS \
  --output-dir "$ICM_BENCH_OUTPUT_ROOT/hipporag2/corpus" \
  --dataset-name icm_bench_speakerless

python album_tools/eval_album_openqa.py \
  --stage all \
  --corpus "$ICM_BENCH_OUTPUT_ROOT/hipporag2/corpus/icm_bench_speakerless_corpus.json" \
  --qa-file SYSTEM_NATIVE_QA_FILE \
  --dataset-id ICM-Bench \
  --save-dir "$ICM_BENCH_OUTPUT_ROOT/hipporag2/index" \
  --output "$ICM_BENCH_OUTPUT_ROOT/hipporag2/answers.jsonl" \
  --llm-name MODEL_NAME \
  --llm-base-url "$OPENAI_BASE_URL"
```

Use `--timeline-mode before_clip` for Recall/Retrieval and `--user-profile-timeline-mode all` for Profile, which are the adapter defaults.

## 6. Judge open-ended answers

The reported metric uses semantic equivalence rather than exact string match. Run the released judge on system answers without passing any evaluator-side metadata to the system that produced them:

```bash
python evaluation/judging/scripts/rejudge_openqa_m3agent_style.py \
  --input "$ICM_BENCH_OUTPUT_ROOT/SYSTEM/answers.jsonl" \
  --output "$ICM_BENCH_OUTPUT_ROOT/SYSTEM/answers.judged.jsonl" \
  --summary "$ICM_BENCH_OUTPUT_ROOT/SYSTEM/answers.judged.summary.json"
```

An independent OpenAI-compatible cross-check is available in `rejudge_openqa_m3agent_style_openai.py`. Record the judge model, prompt version, and decoding settings with every reported score.

## 7. Validate a packaged dataset

The release tool only packs finalized public-format inputs and validates the result:

```bash
python tools/build_huggingface_release.py pack --help
python tools/build_huggingface_release.py validate RELEASE_DIR
```

It rejects unexpected annotation fields, non-public clip IDs, incorrect category totals, invalid cutoffs, mismatched transcript variants, unsafe tar members, and checksum failures. It performs no upload.
