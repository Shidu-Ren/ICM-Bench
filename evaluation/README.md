# Evaluation

Each evaluated memory framework has one self-contained directory containing its ICM-Bench adapters, setup notes, and provenance information:

- [`m3_agent/`](m3_agent/): the original M3-Agent memory pipeline, an audio-only control, and ICM-Bench timeline adapters;
- [`vgent/`](vgent/): ICM-Bench graph construction and answer adapters, with the pinned upstream checkout in `vgent/upstream/`;
- [`hipporag2/`](hipporag2/): HippoRAG2 plus the transcript-corpus and open-ended QA adapters;
- [`judging/`](judging/): the shared semantic-equivalence evaluator used to score system answers.

The evaluated systems use separate dependency environments. Follow each method README and the [end-to-end workflow](../docs/EXPERIMENT_WORKFLOW.md). All systems should emit question IDs and open-ended answers before they are passed to the shared judge. Upstream repositories, audited revisions, and license status are recorded in [UPSTREAMS.md](UPSTREAMS.md).
