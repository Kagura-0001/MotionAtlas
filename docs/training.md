# MotionAtlas Qwen3-VL Training

This release follows the GAR-style training flow:

```bash
bash tools/dist.sh train projects/motionatlas/configs/qwen3vl_4b_motionatlas.yaml 8
```

The YAML config is the only user-facing recipe. The launcher converts it to an
internal MMEngine/XTuner config under `work_dirs/.../.generated/`.

## 1. Environment

Use a separate training environment from the vLLM serving environment. XTuner
`0.2.0rc0` pins an older transformers release, so install it with `--no-deps`
and then install the Qwen3-VL-compatible runtime.

```bash
conda create -n motionatlas-train python=3.11 -y
conda activate motionatlas-train

pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install xtuner==0.2.0rc0 --no-deps
pip install transformers==4.57.3 deepspeed==0.18.2 peft==0.15.2 timm==1.0.19
pip install datasets decord==0.6.0 einops mmengine==0.10.6 numpy opencv-python-headless pillow pyarrow pycocotools pyyaml qwen-vl-utils safetensors tqdm
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Adjust the PyTorch CUDA wheel index for your cluster if it does not use CUDA
12.8-compatible drivers.

## 2. Data

Download MotionAtlas-Data metadata:

```bash
hf download maxLWSv2/motionatlas-data --repo-type dataset --local-dir data/motionatlas-data
```

Then set media roots for the public source keys used by the dataset:

```bash
export MOTIONATLAS_DATA_ROOT=data/motionatlas-data
export MOTIONATLAS_SAV_ROOT=/path/to/SA-V
export MOTIONATLAS_MEVIS_ROOT=/path/to/MeViS
export MOTIONATLAS_TAO_ROOT=/path/to/TAO-Amodal
export MOTIONATLAS_DANCETRACK_ROOT=/path/to/DanceTrack
export MOTIONATLAS_GOT10K_ROOT=/path/to/GOT-10k
export MOTIONATLAS_VICAS_ROOT=/path/to/ViCaS
```

The media path rule is fixed:

```text
<source root from YAML>/<record["video"]>
```

`media_type: video` points to a video file. `media_type: frame_dir` points to a
directory of ordered image frames.

## 3. Preflight

Before launching a multi-GPU job, materialize a few samples:

```bash
python projects/motionatlas/tools/inspect_dataset.py \
  projects/motionatlas/configs/qwen3vl_4b_motionatlas.yaml \
  --limit 32
```

For a tiny smoke subset:

```bash
python projects/motionatlas/tools/inspect_dataset.py \
  projects/motionatlas/configs/qwen3vl_4b_motionatlas.yaml \
  --limit 8 \
  --cfg-options data.max_samples=8
```

## 4. Train

Run on one 8-GPU node:

```bash
bash scripts/train_qwen3vl.sh
```

Useful overrides:

```bash
# small local smoke run
bash scripts/train_qwen3vl.sh projects/motionatlas/configs/qwen3vl_4b_motionatlas.yaml 1 \
  --cfg-options data.max_samples=8 train.max_steps=2

# custom output directory
MOTIONATLAS_WORK_DIR=work_dirs/my_motionatlas_run \
bash scripts/train_qwen3vl.sh
```

## 5. Convert To HuggingFace

If the base model is local, merge checkpoint tensors onto it:

```bash
python projects/motionatlas/tools/convert_to_hf.py \
  --config projects/motionatlas/configs/qwen3vl_4b_motionatlas.yaml \
  --checkpoint work_dirs/qwen3vl_4b_motionatlas/iter_xxx.pth \
  --base-model /path/to/Qwen3-VL-4B-Instruct \
  --output-dir outputs/MotionAtlas-4B
```

If the checkpoint already contains a complete model state dict, skip base merge:

```bash
python projects/motionatlas/tools/convert_to_hf.py \
  --config projects/motionatlas/configs/qwen3vl_4b_motionatlas.yaml \
  --checkpoint work_dirs/qwen3vl_4b_motionatlas/iter_xxx.pth \
  --output-dir outputs/MotionAtlas-4B \
  --no-merge-base
```
