# MotionAtlas: Detailed Region Captioning for Motion-Centric Videos

by
[Weisong Liu](https://scholar.google.com/citations?user=a20rvfAAAAAJ),
[Haochen Wang](https://scholar.google.com/citations?user=oNlpTdcAAAAJ&hl=en),
Kuan Gao,
[Yuhao Wang](https://scholar.google.com/citations?user=BMWOkScAAAAJ&hl=en),
[Yikang Zhou](https://scholar.google.com/citations?user=dZikW2YAAAAJ&hl=en),
[Zhongwei Ren](https://scholar.google.com/citations?user=e5TJm-0AAAAJ&hl=en),
Jacky Mai,
Anna Wang,
[Yanwei Li](https://scholar.google.com/citations?user=I-UCPPcAAAAJ&hl=en),
Jason Li, and
[Zhaoxiang Zhang](https://scholar.google.com/citations?user=qxWfV6cAAAAJ).

[[Paper](TODO)] | [[Project Page](https://kagura-0001.github.io/projects/MotionAtlas/)] | [[HuggingFace Collection](https://huggingface.co/collections/maxLWSv2/motionatlas-6a3c4bb7c242d1946dc7fa41)] | [[Model](https://huggingface.co/maxLWSv2/MotionAtlas-4B)] | [[Citation](#citation)]

**TL;DR**: MotionAtlas is a system for **detailed captioning of motion-centric videos**, comprising (1) a human-annotated benchmark for *region-aware* motion understanding, (2) a scalable, high-quality data pipeline, and (3) a family of Video-MLLMs. Given a video and a spatiotemporal mask, MotionAtlas describes the motion *within the target region*, alleviating visual clutter and motion entanglement, and enabling reliable, quantifiable evaluation.

![](./assets/teaser.png)

![](./assets/data_gains.png)

> **Abstract.** Unlike conventional global motion captioning, we focus on region-aware motion captioning: given a video and a spatiotemporal mask, the model generates precise descriptions of motion within the target region. We first build **MotionAtlas-Bench**, a comprehensive benchmark of 2,073 multiple-choice questions over a curated set of high-quality, motion-centric videos, to evaluate fine-grained motion understanding of the objects in question. We then design a rigorous, scalable data pipeline that leverages self-bootstrap refinement to suppress fine-grained hallucinations, yielding 159K high-quality motion captioning samples (**MotionAtlas-Data**). Finally, a tailored training data composition strategy delivers consistent and substantial gains across diverse baseline Video-MLLMs, including Molmo2 and Qwen3-VL. For instance, MotionAtlas-4B surpasses Qwen3-VL-4B by an average of 5.2 points across general motion benchmarks.

## Updates

- **2026.06.30**: 🤗 Released the [MotionAtlas collection](https://huggingface.co/collections/maxLWSv2/motionatlas-6a3c4bb7c242d1946dc7fa41) on HuggingFace — [MotionAtlas-Bench](https://huggingface.co/datasets/maxLWSv2/motionatlas-bench), [MotionAtlas-Data](https://huggingface.co/datasets/maxLWSv2/motionatlas-data), and the [MotionAtlas-4B](https://huggingface.co/maxLWSv2/MotionAtlas-4B) model. Added MotionAtlas-Bench evaluation code under `evaluation/motionatlas_bench`.
- **2026.06.21**: Repository initialized.

## Resources

| Resource | Link | Description |
| --- | --- | --- |
| **MotionAtlas-Bench** | [🤗 maxLWSv2/motionatlas-bench](https://huggingface.co/datasets/maxLWSv2/motionatlas-bench) | 2,073 region-level motion MCQs over 107 videos for evaluation. |
| **MotionAtlas-Data** | [🤗 maxLWSv2/motionatlas-data](https://huggingface.co/datasets/maxLWSv2/motionatlas-data) | 159K high-quality region-level motion captioning samples for training. |
| **MotionAtlas-4B** | [🤗 maxLWSv2/MotionAtlas-4B](https://huggingface.co/maxLWSv2/MotionAtlas-4B) | Video-MLLM fine-tuned with MotionAtlas-Data. |

# Demos

Examples from **MotionAtlas-Data**. Given a user-specified region (highlighted in each clip), MotionAtlas produces a detailed, temporally grounded description of that region.

<p align="center">
  <img src="demo/previews/demo_gym.gif" width="48%" alt="MotionAtlas gym demo" />
  <img src="demo/previews/demo_dogs.gif" width="48%" alt="MotionAtlas dog interaction demo" />
</p>
<p align="center">
  <img src="demo/previews/demo_tunnel.gif" width="48%" alt="MotionAtlas tunnel driving demo" />
  <img src="demo/previews/demo_dance.gif" width="48%" alt="MotionAtlas dance demo" />
</p>

**[Interactive Demo](demo/index.html)** | **[Project Page](https://kagura-0001.github.io/projects/MotionAtlas/#demos)**

# Evaluation Quick Start

MotionAtlas-Bench evaluation code lives under `evaluation/motionatlas_bench`. It implements the **caption-to-judge** protocol used by the benchmark: a Video-MLLM first generates a target-object motion caption from highlighted video frames, then a text judge answers each MCQ given only that caption. The judge classifies every answer as `correct` / `wrong` / `miss`, from which we report **Accuracy**, **Recall** (tendency to describe the target motion rather than answer "not mentioned"), and **Precision** (correctness when the motion is explicitly mentioned), plus a `weighted_score`.

The benchmark provides two increasingly difficult **grounding settings**:

- **`first_mask`** (Single-Frame Grounding): the target mask is shown **only at its first visible frame**, so the model must track the entity through the clip — jointly testing spatial tracking and motion understanding.
- **`overlay_all`** (Full-Sequence Grounding): per-frame masks are overlaid on **every frame**, removing tracking ambiguity to strictly measure intrinsic motion captioning capacity.

## 1. Environment

Install dependencies from the provided `pyproject.toml` with `uv`:

```bash
uv sync
```

If you also serve the Qwen3-VL caption model with the provided vLLM script, install the optional serving group. The CUDA 12.9 PyTorch and vLLM wheel sources are pinned in `pyproject.toml`:

```bash
uv sync --group serve
```

The base environment covers the MotionAtlas-Bench runner, mask rendering, Gemini judging, and the `hf` download CLI used below. The `serve` group adds `vllm` for Step 3.

## 2. Download MotionAtlas-Bench

MotionAtlas-Bench is gated; accept the terms on the [dataset page](https://huggingface.co/datasets/maxLWSv2/motionatlas-bench), then log in and download:

```bash
uv run hf auth login
uv run hf download maxLWSv2/motionatlas-bench --repo-type dataset --local-dir ../motionatlas-bench-v1
```

The release contains `mcqs.jsonl` (one MCQ per line), the `videos/` media referenced by `video_path` (either `.mp4` files or directories of ordered image frames), and dataset bookkeeping (`manifest.json`, `checksums.sha256`).

## 3. Serve the Caption Model

Start a Qwen3-VL vLLM server. On an 8-GPU node, Qwen3-VL-4B is best served as eight data-parallel replicas behind a single OpenAI-compatible endpoint:

```bash
MODEL_PROFILE=qwen4b \
MODEL_PATH=Qwen/Qwen3-VL-4B-Instruct \
MODEL_NAME=qwen3-vl-4b \
TP_SIZE=1 \
DP_SIZE=8 \
DP_BACKEND=mp \
MAX_NUM_SEQS=8 \
PORT=8000 \
uv run bash scripts/serve_qwen3vl_vllm.sh
```

To evaluate our released model instead, point `MODEL_PATH` at [`maxLWSv2/MotionAtlas-4B`](https://huggingface.co/maxLWSv2/MotionAtlas-4B). The serve script defaults to `DP_SIZE=1` for portability; set `DP_SIZE=8` on 8-GPU nodes to expose one vLLM replica per GPU. Reduce `DP_SIZE` and `MAX_NUM_SEQS` if GPU memory is tight.

## 4. Run MotionAtlas-Bench

```bash
export GEMINI_API_KEY=YOUR_GEMINI_API_KEY

uv run python -m evaluation.motionatlas_bench.run_eval \
  --data-root ../motionatlas-bench-v1 \
  --output outputs/qwen3vl4b_first_mask_16_gemini25pro \
  --setting first_mask \
  --num-frames 16 \
  --caption-model qwen3-vl-4b \
  --caption-base-url http://127.0.0.1:8000/v1 \
  --caption-api-key EMPTY \
  --caption-workers 32 \
  --judge-provider gemini \
  --judge-model gemini-2.5-pro \
  --judge-workers 8
```

This writes `captions.jsonl`, `judge_predictions.jsonl`, `metrics.json`, and `run_config.json` under the output directory. Use `--setting overlay_all` for the all-mask (Full-Sequence) setting. The prompt labels images as `Frame 1:`, `Frame 2:`, and so on, matching the model input sequence; original zero-based public media frame indices are kept only in `render_metadata`. Reduce `--caption-workers` if GPU memory is tight.

To use a **local OpenAI-compatible text judge** instead of Gemini, serve it with vLLM and switch the judge provider:

```bash
uv run python -m evaluation.motionatlas_bench.run_eval \
  --data-root ../motionatlas-bench-v1 \
  --output outputs/qwen3vl4b_first_mask_16_qwen_judge \
  --setting first_mask \
  --num-frames 16 \
  --caption-model qwen3-vl-4b \
  --caption-base-url http://127.0.0.1:8000/v1 \
  --caption-api-key EMPTY \
  --caption-workers 16 \
  --judge-provider openai-compatible \
  --judge-model qwen3.6-27b \
  --judge-base-url http://127.0.0.1:8001/v1 \
  --judge-api-key EMPTY \
  --judge-workers 16
```

The reported metrics span six motion aspects — **Spatial**, **Parts**, **Kinematics**, **Interaction**, **State**, and **Camera**. See the paper for full benchmarked results across base models and grounding settings.

<p align="center">
  <img src="./assets/aspect_distribution.png" width="55%" alt="MotionAtlas-Bench aspect distribution" />
</p>

# Training

Training code and configs (with MotionAtlas-Data) will be released here. See [MotionAtlas-Data](https://huggingface.co/datasets/maxLWSv2/motionatlas-data) for the training corpus and its media-preparation guide.

# License

This project is licensed under the [Apache-2.0 License](LICENSE). Benchmark annotations and media are released for MotionAtlas-Bench evaluation; media files may also be subject to the licenses or terms of their original sources.

# Citation

If you find this project useful, please consider citing:

```bibtex
@article{liu2026motionatlas,
  title   = {MotionAtlas: Detailed Region Captioning for Motion-Centric Videos},
  author  = {Liu, Weisong and Wang, Haochen and Gao, Kuan and Wang, Yuhao and Zhou, Yikang and Ren, Zhongwei and Mai, Jacky and Wang, Anna and Li, Yanwei and Li, Jason and Zhang, Zhaoxiang},
  journal = {arXiv preprint},
  year    = {2026}
}
```

# Acknowledgements

We thank the data sources that make MotionAtlas possible, including [SA-V](https://ai.meta.com/datasets/segment-anything-video/), [MeViS](https://huggingface.co/datasets/FudanCVL/MeViS), [TAO](https://huggingface.co/datasets/chengyenhsieh/TAO-Amodal), [DanceTrack](https://huggingface.co/datasets/noahcao/dancetrack), [ViCaS](https://huggingface.co/datasets/Ali2500/ViCaS), [VastTrack](https://github.com/HengLan/VastTrack), and [GOT-10k](http://got-10k.aitestunion.com/). Our evaluation follows the protocols of [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) and [VLMEvalKit](https://github.com/open-compass/VLMEvalKit).
