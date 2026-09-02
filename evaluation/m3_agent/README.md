# M3-Agent integration

> Modified by the ICM-Bench authors in 2026 for benchmark integration.

This directory contains the M3-Agent implementation used for ICM-Bench, based on the [official M3-Agent repository](https://github.com/ByteDance-Seed/m3-agent) at revision `0e3e41939bd8a0b66d756e7b7eb8d5fe9992da5c`. The upstream Apache-2.0 license is retained in [`LICENSE`](LICENSE).

The ICM-Bench additions provide ordered album-memory construction, question-specific timeline cutoffs, an audio-only control, result analysis, and portable launch scripts for the reported original M3-Agent setting. See the [public experiment workflow](../../docs/EXPERIMENT_WORKFLOW.md) for the released data layout and end-to-end procedure.

## Main entry points

- `m3_agent/memorization_intermediate_outputs.py`: face and speaker preprocessing.
- `m3_agent/memorization_memory_graphs.py`: multimodal memory construction.
- `m3_agent/memorization_memory_graphs_audio_only.py`: transcript/audio control.
- `m3_agent/control.py`: retrieval, reasoning, and open-ended answering.
- `scripts/run_album_m3_original_strict_ordered.sh`: ICM-Bench orchestration template.
- `scripts/run_memorization_audio_only.sh`: audio-only memory template.
- `scripts/retarget_album_annotation_mem_path.py`: public-QA to M3-Agent path adapter.

## Setup

Run `setup.sh` in a dedicated environment, then install the official M3-Agent runtime additions:

```bash
python -m pip install \
  git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8 \
  qwen-omni-utils==0.0.4

git clone --depth 1 https://github.com/modelscope/3D-Speaker.git external/3D-Speaker
export PYTHONPATH="$PWD/external/3D-Speaker:$PWD${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p models
curl -L \
  https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/resolve/master/pretrained_eres2netv2.ckpt \
  -o models/pretrained_eres2netv2.ckpt
```

The control stage additionally uses `transformers==4.51.0`, `vllm==0.8.4`, and `numpy==1.26.4`. Download the official [M3-Agent-Memorization](https://huggingface.co/ByteDance-Seed/M3-Agent-Memorization) and [M3-Agent-Control](https://huggingface.co/ByteDance-Seed/M3-Agent-Control) checkpoints separately.

Gemini-backed graph and retrieval utilities read credentials from `GOOGLE_API_KEY` or `GEMINI_API_KEY`. Local data, model, output, and device settings are supplied through the environment variables documented by the launcher; no credentials or machine paths are stored in this repository.

## Upstream citation

Please cite the M3-Agent paper when using this integration:

```bibtex
@misc{long2025seeing,
  title={Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory},
  author={Long, Lin and He, Yichen and Ye, Wentao and Pan, Yiyuan and Lin, Yuan and Li, Hang and Zhao, Junbo and Li, Wei},
  year={2025},
  eprint={2508.09736},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
