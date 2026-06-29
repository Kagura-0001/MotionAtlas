# MotionAtlas-Bench

MotionAtlas-Bench evaluates target-object motion captioning. The model generates
one detailed caption for each highlighted target-object sample, then an official
Gemini judge answers all multiple-choice checks from that caption.

This task intentionally uses caption -> judge evaluation. It is not a direct
multiple-choice VQA task.

## Tasks

- `motionatlas_bench`: group
- `motionatlas_bench_first`: first-mask grounding setting

## Data

The default dataset is `maxLWSv2/motionatlas-bench`. The task needs:

- `mcqs.jsonl`
- `target_masks.jsonl.gz`
- `videos/`

For local development, point the task at a local release directory:

```bash
export MOTIONATLAS_BENCH_ROOT=/path/to/motionatlas-bench-v1
```

If `MOTIONATLAS_BENCH_ROOT` is not set, files are downloaded from the Hugging
Face dataset repo specified by `MOTIONATLAS_BENCH_REPO`, defaulting to
`maxLWSv2/motionatlas-bench`.

## Judge

The default judge is the official Gemini API.

```bash
export GEMINI_API_KEY=...
export MOTIONATLAS_JUDGE_MODEL=gemini-2.5-pro
```

Useful optional environment variables:

- `MOTIONATLAS_JUDGE_WORKERS`: per-sample MCQ judge workers, default `4`
- `MOTIONATLAS_JUDGE_MAX_RETRIES`: judge retry count, default `3`
- `MOTIONATLAS_JUDGE_MAX_TOKENS`: judge max output tokens, default `256`
- `MOTIONATLAS_JUDGE_CACHE`: optional JSON cache file for judge responses

## Example

```bash
lmms-eval \
  --model openai_compatible_chat \
  --model_args model_version=qwen3-vl-4b,base_url=http://127.0.0.1:8000/v1,api_key=EMPTY,batch_size=4 \
  --tasks motionatlas_bench_first \
  --batch_size 4 \
  --limit 2
```
