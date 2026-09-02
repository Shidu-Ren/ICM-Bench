# Evaluation framework provenance

The experiment directories preserve upstream research code together with ICM-Bench-specific adaptations. The top-level project license does not replace the license, copyright, or permission requirements of these subtrees.

| System | Upstream | Audited upstream revision | License status |
|---|---|---|---|
| M3-Agent | https://github.com/ByteDance-Seed/m3-agent | `0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c` | Apache-2.0; the upstream license is retained at `evaluation/m3_agent/LICENSE` |
| HippoRAG2 | https://github.com/OSU-NLP-Group/HippoRAG | `2f52a86dd04e4633703bd2fb3bb6a37683ac3cfb` | MIT; the upstream license is retained at `evaluation/hipporag2/LICENSE` |
| Vgent | https://github.com/xiaoqian-shen/Vgent | `c35c4b7731f22b962de24f31f842f6d84e183f8f` | Pinned at `evaluation/vgent/upstream`; no repository-level license was present at the audited revision |

The revisions above are the upstream heads checked on 2026-09-01. M3-Agent and HippoRAG2 are retained snapshots with ICM-Bench changes; Vgent is a Git submodule fetched directly from its original repository.

## Vgent permission risk

Public source code without an explicit license remains subject to copyright by default. The Vgent paper and public repository are useful scholarly references, but they do not by themselves authorize copying, modification, redistribution, or sublicensing. For this reason, ICM-Bench distributes only its independently authored integration code and records the upstream revision as a submodule. Users who initialize that submodule receive Vgent from its original repository and remain responsible for obtaining any rights needed for reuse.
