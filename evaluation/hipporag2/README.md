# HippoRAG2 integration

This directory contains the HippoRAG2 implementation used for ICM-Bench, based on the [official HippoRAG repository](https://github.com/OSU-NLP-Group/HippoRAG) at revision `2f52a86dd04e4633703bd2fb3bb6a37683ac3cfb`. The upstream MIT license is retained in [`LICENSE`](LICENSE).

The ICM-Bench additions build a chronological text corpus from released speakerless transcripts, enforce Recall/Retrieval timeline cutoffs, use full-timeline access for Profile questions, and evaluate open-ended answers. See the [public experiment workflow](../../docs/EXPERIMENT_WORKFLOW.md) for the complete command sequence.

## Main entry points

- `album_tools/prepare_album_corpus_from_transcripts.py`: build the speakerless transcript corpus.
- `album_tools/prepare_album_corpus_from_captions.py`: optional caption-corpus adapter for memory-system experiments.
- `album_tools/eval_album_openqa.py`: index, retrieve, and answer ICM-Bench questions.
- `album_tools/qwen_image_text_openai_server.py`: optional OpenAI-compatible local model wrapper.

## Setup

Use an isolated Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Model endpoints, embedding endpoints, benchmark paths, and output directories are passed explicitly to the evaluation command. No credentials or machine-specific paths are stored in this repository.

## Upstream citation

```bibtex
@misc{gutierrez2025ragmemory,
  title={From RAG to Memory: Non-Parametric Continual Learning for Large Language Models},
  author={Bernal Jimenez Gutierrez and Yiheng Shu and Weijian Qi and Sizhe Zhou and Yu Su},
  year={2025},
  eprint={2502.14802},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
