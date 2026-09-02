# ICM-Bench

Official code repository for **ICM-Bench: Person-Level Identity Reasoning in Multimodal Agents with Long-Term Memory**.

ICM-Bench evaluates whether multimodal agents can retrieve and aggregate person-centered evidence across a year-long episodic video timeline. The benchmark data are hosted in the [ICM-Bench Hugging Face dataset](https://huggingface.co/datasets/ryanren0330/ICM-Bench); this repository contains the synthetic-video generation pipeline, three memory-system evaluation integrations, semantic-equivalence judging, and a clean release packager.

## Released components

| Component | Location | Public-release scope |
|---|---|---|
| Synthetic video generation | `generation/` | Theme and timeline planning, reusable character/scene anchors, shot rendering, recurring-character speech, subtitles, and date-stamped video export |
| M3-Agent evaluation | `evaluation/m3_agent/` | Official M3-Agent baseline adapted for ordered album-memory construction, timeline cutoffs, an audio-only control, and open-ended evaluation |
| Vgent evaluation | `evaluation/vgent/` | Manifest, graph construction, retrieval, refinement, and answering adapters, with the pinned upstream checkout beside them in `upstream/` |
| HippoRAG2 evaluation | `evaluation/hipporag2/` | HippoRAG2 snapshot plus transcript-corpus and open-ended evaluation adapters |
| Answer judging | `evaluation/judging/` | Semantic-equivalence judging and independent cross-judge verification |
| Dataset packaging | `tools/build_huggingface_release.py` | Copies finalized public videos, annotations, character metadata, and transcripts into the release layout and validates schema, inventory, cutoffs, and checksums |

Large assets are intentionally kept outside Git: videos, released annotations, transcripts, model weights, memory graphs, answers, logs, and caches.

The direct caption-memory baseline implementation is intentionally not included in this public code release. Caption and transcript preparation utilities remain only where they are required by the released memory-system integrations.

## Benchmark data

The public test release contains:

- 839 videos (`clip_000.mp4` through `clip_838.mp4`), including one calibration clip and 838 date-stamped memory clips;
- 1,217 open-ended questions: 400 Identity Recall, 500 Cross-Episode Identity Retrieval, and 317 Long-Term Identity Profile Inference;
- 829 speakerless ASR transcripts used by transcript-based evaluation settings and 829 speaker-labeled transcripts supplied only for reference;
- public clip-level evidence annotations, with question-specific `before_clip` cutoffs for Recall and Retrieval.

Download and extract the videos with:

```bash
hf download ryanren0330/ICM-Bench \
  --repo-type dataset \
  --local-dir "$ICM_BENCH_DATA_ROOT"
tar -xf "$ICM_BENCH_DATA_ROOT/videos.tar" -C "$ICM_BENCH_DATA_ROOT"
```

The annotation files contain evaluator-side fields. Do not expose `reference_answer`, `target_character_ids`, `evidence_video_ids`, `annotations/characters.json`, or `resources/transcripts_with_speakers/` to an evaluated system. Use `resources/asr_transcripts/` for the released speakerless transcript setting.

## Repository layout

```text
ICM-Bench/
├── generation/
├── evaluation/
│   ├── m3_agent/
│   ├── vgent/
│   │   └── upstream/           # pinned Vgent submodule
│   ├── hipporag2/
│   └── judging/
├── configs/
├── docs/
└── tools/
```

## Setup

The generation pipeline and the three evaluated systems have incompatible dependency stacks, so use a separate virtual environment for each component. Install the requirements or setup file inside the component you plan to run.

```bash
git clone --recurse-submodules https://github.com/Shidu-Ren/ICM-Bench.git
cd ICM-Bench
cp .env.example .env
cp configs/paths.example.yaml configs/paths.local.yaml
set -a
source .env
set +a
```

Fill in the generic roots and any required credentials in `.env`, then source it in each new shell as shown above. Use `configs/paths.local.yaml` as a local path worksheet. The checked-in example files contain no credentials or machine-specific locations. See [the experiment workflow](docs/EXPERIMENT_WORKFLOW.md) for the public entry points.

## Packaging a prepared release

`tools/build_huggingface_release.py` accepts finalized public-format inputs and produces a local release directory. Run `python tools/build_huggingface_release.py pack --help` for the input contract and `python tools/build_huggingface_release.py validate RELEASE_DIR` for an integrity check. The tool has no upload command.

## Licensing and upstream code

Original ICM-Bench project code and documentation are available under the [MIT License](LICENSE). The separately distributed dataset is licensed under [CC BY-NC-SA 4.0](LICENSE-DATASET). These licenses do not override upstream terms.

M3-Agent is included under Apache-2.0 and HippoRAG2 under MIT. Vgent is referenced through a pinned upstream submodule because the audited upstream repository did not provide a repository-level license. See [NOTICE](NOTICE) and [evaluation/UPSTREAMS.md](evaluation/UPSTREAMS.md) before reusing upstream code.

## Citation

```bibtex
@misc{ren2026icmbench,
  title  = {ICM-Bench: Person-Level Identity Reasoning in Multimodal Agents with Long-Term Memory},
  author = {Ren, Shidu and Liu, Yunze and Liu, Xing and Wu, Chi-Hao and Zhou, Enmin and Shen, Junxiao},
  year   = {2026},
  note   = {arXiv preprint}
}
```
