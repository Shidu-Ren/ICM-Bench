# Vgent integration

This directory contains the ICM-Bench adapters for the [official Vgent repository](https://github.com/xiaoqian-shen/Vgent). The upstream checkout is pinned beside the adapters in `upstream/` at revision `c35c4b7731f22b962de24f31f842f6d84e183f8f`.

The ICM-Bench additions prepare the public video manifest, optionally attach released speakerless transcripts, construct per-clip graphs, apply question-specific timeline cutoffs, retrieve and refine evidence, and produce open-ended answers. See the [public experiment workflow](../../docs/EXPERIMENT_WORKFLOW.md) for commands and data-handling rules.

## Main entry points

- `scripts/prepare_album_manifest.py`: video-only public manifest.
- `scripts/prepare_album_transcript_manifest.py`: manifest using released speakerless transcripts.
- `scripts/build_album_graph.py`: offline graph construction through the pinned upstream framework.
- `scripts/eval_album_openqa.py`: open-ended retrieval and answering.
- `scripts/eval_album_openqa_refine.py`: retrieval, evidence refinement, and answering.
- `scripts/inject_album_subtitles_into_graph.py`: optional speakerless-subtitle attachment.

Initialize the submodule and install the adapter environment from the repository root:

```bash
git submodule update --init evaluation/vgent/upstream
python -m pip install -r evaluation/vgent/requirements.txt
export VGENT_ROOT="$PWD/evaluation/vgent/upstream"
```

Model checkpoints and benchmark assets are downloaded separately and are not stored in this repository.

## License status

The audited upstream Vgent revision did not contain a repository-level license. This repository therefore records a Git reference rather than redistributing the upstream source tree. Initializing the submodule fetches code directly from its original repository and does not itself grant reuse rights. See [`NOTICE`](../../NOTICE) and [`evaluation/UPSTREAMS.md`](../UPSTREAMS.md).

## Upstream citation

```bibtex
@inproceedings{shen2025vgent,
  title={Vgent: Graph-based Retrieval-Reasoning-Augmented Generation for Long Video Understanding},
  author={Shen, Xiaoqian and Zhang, Wenxuan and Chen, Jun and Elhoseiny, Mohamed},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```
