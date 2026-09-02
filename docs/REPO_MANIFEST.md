# Repository manifest

## Included

- The synthetic-video generation pipeline under `generation/`.
- M3-Agent and HippoRAG2 evaluation snapshots, a pinned Vgent submodule, and ICM-Bench adapters for all three frameworks.
- Semantic-equivalence answer judging and independent judge cross-checks.
- Component requirements, portable configuration examples, and a clean-input dataset packager/validator.
- Upstream attribution and license notices.

## Excluded

- Released videos, annotations, transcripts, model weights, memory graphs, answer files, logs, and caches; obtain the benchmark from [Hugging Face](https://huggingface.co/datasets/ryanren0330/ICM-Bench).
- Private credentials, machine-local paths, environment-history exports, scheduler state, and host-specific launch state.
- Internal curation utilities, unpublished experiment artifacts, historical result dumps, and paper-editing files.
- The direct caption-memory baseline implementation, which is outside this public code release.

The project does not provide a single combined environment because the generation pipeline and three evaluated systems have incompatible dependency stacks. Install dependencies from the relevant component and supply paths through environment variables or an untracked `configs/paths.local.yaml`.
