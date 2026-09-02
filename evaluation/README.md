# Evaluation

This guide runs the M3-Agent, Vgent, and HippoRAG2 integrations on [ICM-Bench](https://arxiv.org/abs/2609.04438). Commands start from the repository root unless noted. Install each framework in a separate environment using its component README; upstream sources and revisions are listed in [UPSTREAMS.md](UPSTREAMS.md).

## 1. Prepare the benchmark

```bash
cp .env.example .env
set -a
source .env
set +a

python -m pip install huggingface_hub
hf download ryanren0330/ICM-Bench \
  --repo-type dataset \
  --local-dir "$ICM_BENCH_DATA_ROOT"
tar -xf "$ICM_BENCH_DATA_ROOT/videos.tar" -C "$ICM_BENCH_DATA_ROOT"
```

Confirm that `videos/clip_000.mp4` through `videos/clip_838.mp4` exist after extraction. Recall and Retrieval items may access memory clips only through their `before_clip`; Profile items use the full timeline. Video-based settings additionally use `clip_000` as a face–voice calibration reference. Keep all answer keys, evidence pointers, character metadata, and speaker-labeled reference transcripts outside model input.

## 2. M3-Agent

M3-Agent constructs a multimodal memory graph and then answers the open-ended benchmark questions through its control loop. The relevant public entry points are:

- `m3_agent/memorization_memory_graphs.py` for ordered memory construction;
- `m3_agent/control.py` for retrieval/reasoning and answer generation;
- `scripts/run_album_m3_original_strict_ordered.sh` as the orchestration template.

After completing the [M3-Agent setup](m3_agent/README.md), launch the original setting from the repository root:

```bash
export ICM_BENCH_ROOT="$ICM_BENCH_DATA_ROOT"
bash evaluation/m3_agent/scripts/run_album_m3_original_strict_ordered.sh
```

The launcher requires `ICM_BENCH_ROOT` and accepts portable overrides including `WORK_DIR`, `PY_BIN`, `GPU_LIST`, `SOURCE_DIR`, `ANNOTATION_IN`, `MEM_PATH`, `INTERMEDIATE_DIR`, `DATA_JSONL`, `ANNOTATION_OUT`, `RESULT_NAME`, and `LOG_DIR`. Keep the public clip order and enforce each question's cutoff when preparing the system-native annotation wrapper.

## 3. Vgent

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

The answer adapter applies each non-Profile item's released `before_clip` cutoff:

```bash
python evaluation/vgent/scripts/eval_album_openqa_refine.py \
  --model_name MODEL_NAME \
  --qa_path "$ICM_BENCH_DATA_ROOT/annotations/qa_test.json" \
  --manifest_path "$ICM_BENCH_OUTPUT_ROOT/vgent/manifest/album_manifest.jsonl" \
  --graph_dir "$ICM_BENCH_OUTPUT_ROOT/vgent/graphs/album_1.0fps_64" \
  --output_path "$ICM_BENCH_OUTPUT_ROOT/vgent/answers.jsonl"
```

To use speakerless transcripts, attach them before graph construction and pass the resulting manifest to both stages:

```bash
python evaluation/vgent/scripts/prepare_album_transcript_manifest.py \
  --base-manifest "$ICM_BENCH_OUTPUT_ROOT/vgent/manifest/album_manifest.jsonl" \
  --subtitle-dir "$ICM_BENCH_DATA_ROOT/resources/asr_transcripts" \
  --output "$ICM_BENCH_OUTPUT_ROOT/vgent/manifest/album_manifest_asr.jsonl"
```

## 4. HippoRAG2

HippoRAG2 supports speakerless transcript corpora and caption corpora. The example below runs the transcript-only control. The caption-corpus adapter is `album_tools/prepare_album_corpus_from_captions.py`; provide your prepared captions for that setting.

```bash
cd evaluation/hipporag2
python -m pip install -r requirements.txt
python -m pip install -e .

python ../vgent/scripts/prepare_album_manifest.py \
  --source-dir "$ICM_BENCH_DATA_ROOT/videos" \
  --output-dir "$ICM_BENCH_OUTPUT_ROOT/hipporag2/manifest"

python ../vgent/scripts/prepare_album_transcript_manifest.py \
  --base-manifest "$ICM_BENCH_OUTPUT_ROOT/hipporag2/manifest/album_manifest.jsonl" \
  --subtitle-dir "$ICM_BENCH_DATA_ROOT/resources/asr_transcripts" \
  --output "$ICM_BENCH_OUTPUT_ROOT/hipporag2/manifest/album_manifest_asr.jsonl"

python album_tools/prepare_album_corpus_from_transcripts.py \
  --manifest "$ICM_BENCH_OUTPUT_ROOT/hipporag2/manifest/album_manifest_asr.jsonl" \
  --output-dir "$ICM_BENCH_OUTPUT_ROOT/hipporag2/corpus" \
  --dataset-name icm_bench_speakerless

python album_tools/eval_album_openqa.py \
  --stage all \
  --corpus "$ICM_BENCH_OUTPUT_ROOT/hipporag2/corpus/icm_bench_speakerless_corpus.json" \
  --qa-file "$ICM_BENCH_DATA_ROOT/annotations/qa_test.json" \
  --dataset-id ICM-Bench \
  --save-dir "$ICM_BENCH_OUTPUT_ROOT/hipporag2/index" \
  --output "$ICM_BENCH_OUTPUT_ROOT/hipporag2/answers.jsonl" \
  --llm-name MODEL_NAME \
  --llm-base-url "$OPENAI_BASE_URL"

cd ../..
```

Use `--timeline-mode before_clip` for Recall/Retrieval and `--user-profile-timeline-mode all` for Profile, which are the adapter defaults.

## 5. Answer judging

The reported metric uses semantic equivalence rather than exact string match. Run the released judge on system answers without passing any evaluator-side metadata to the system that produced them:

```bash
python evaluation/judging/scripts/rejudge_openqa_m3agent_style.py \
  --input "$ICM_BENCH_OUTPUT_ROOT/SYSTEM/answers.jsonl" \
  --output "$ICM_BENCH_OUTPUT_ROOT/SYSTEM/answers.judged.jsonl" \
  --summary "$ICM_BENCH_OUTPUT_ROOT/SYSTEM/answers.judged.summary.json"
```

An independent OpenAI-compatible cross-check is available in `rejudge_openqa_m3agent_style_openai.py`. Record the judge model, prompt version, and decoding settings with every reported score.
