# Open-ended answer judging

This directory contains the two answer-level evaluators used with ICM-Bench:

- `scripts/rejudge_openqa_m3agent_style.py` runs the primary Gemini semantic-equivalence judge.
- `scripts/rejudge_openqa_m3agent_style_openai.py` runs the independent GPT cross-check.

Both scripts expect JSONL answer rows with `id`, `question`, `answer`, and
`response` fields and write judged JSONL plus a summary JSON. Set
`GOOGLE_API_KEY` (or `GEMINI_API_KEY`) for the primary judge and
`OPENAI_API_KEY` for the cross-check. Run either script with `--help` for the
complete command-line interface.
