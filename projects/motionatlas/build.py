from __future__ import annotations

from pathlib import Path
from pprint import pformat
from typing import Any


def _torch_dtype_expr(name: str) -> str:
    mapping = {
        "bf16": "torch.bfloat16",
        "bfloat16": "torch.bfloat16",
        "fp16": "torch.float16",
        "float16": "torch.float16",
        "fp32": "torch.float32",
        "float32": "torch.float32",
    }
    return mapping.get(str(name).lower(), "torch.bfloat16")


def build_mmengine_config_text(cfg: dict[str, Any], source_config: str | Path) -> str:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    ann_cfg = cfg.get("annotation", {})
    train_cfg = cfg.get("train", {})

    model_path = model_cfg.get("name_or_path", "Qwen/Qwen3-VL-4B-Instruct")
    work_dir = train_cfg.get("work_dir", "work_dirs/qwen3vl_4b_motionatlas")
    batch_size = int(train_cfg.get("batch_size", 2))
    accumulative_counts = int(train_cfg.get("gradient_accumulation_steps", 2))
    num_workers = int(train_cfg.get("num_workers", 8))
    epochs = int(train_cfg.get("epochs", 1))
    lr = float(train_cfg.get("lr", 1e-5))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    save_steps = int(train_cfg.get("save_steps", 2000))
    save_total_limit = int(train_cfg.get("save_total_limit", 3))
    seed = int(train_cfg.get("seed", 42))
    max_steps = int(train_cfg.get("max_steps", 0) or 0)

    data_kwargs = {
        "model_path": model_path,
        "hf_dataset": data_cfg.get("hf_dataset", "maxLWSv2/motionatlas-data"),
        "data_root": data_cfg.get("local_dir", "data/motionatlas-data"),
        "split": data_cfg.get("split", "train"),
        "source_roots": data_cfg.get("source_roots", {}),
        "max_frames": int(data_cfg.get("max_frames", 16)),
        "per_frame_tokens": int(data_cfg.get("per_frame_tokens", 256)),
        "max_seq_length": int(data_cfg.get("max_seq_length", 16384)),
        "max_samples": int(data_cfg.get("max_samples", 0) or 0),
        "annotation_mode": ann_cfg.get("mode", "highlight"),
        "annotation_prompt": ann_cfg.get("prompt", "Describe the highlighted object in detail."),
        "annotation_contour_color": ann_cfg.get("contour_color", [0, 255, 0]),
        "annotation_contour_thickness": int(ann_cfg.get("contour_thickness", 2)),
    }

    model_kwargs = {
        "mllm_name_or_path": model_path,
        "trust_remote_code": bool(model_cfg.get("trust_remote_code", True)),
        "attn_implementation": model_cfg.get("attn_implementation", "flash_attention_2"),
        "freeze_llm": bool(model_cfg.get("freeze_llm", False)),
        "freeze_visual_encoder": bool(model_cfg.get("freeze_visual_encoder", False)),
        "use_activation_checkpointing": bool(model_cfg.get("use_activation_checkpointing", True)),
        "pretrained_pth": model_cfg.get("pretrained_pth"),
    }

    train_loop_line = "train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)"
    if max_steps > 0:
        train_loop_line = (
            "train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, max_iters=max_steps)"
        )

    return f'''# Auto-generated from {source_config}. Do not edit.
import torch
from mmengine.hooks import CheckpointHook, DistSamplerSeedHook, IterTimerHook, LoggerHook, ParamSchedulerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop

from projects.motionatlas.datasets.motionatlas_qwen3vl_dataset import MotionAtlasQwen3VLDataset
from projects.motionatlas.datasets.collect_fns import qwen3vl_motionatlas_collect
from projects.motionatlas.models.qwen3vl import Qwen3VLForMotionAtlas

mllm_name_or_path = {model_path!r}
work_dir = {work_dir!r}
batch_size = {batch_size}
accumulative_counts = {accumulative_counts}
dataloader_num_workers = {num_workers}
max_epochs = {epochs}
max_steps = {max_steps}
lr = {lr!r}
warmup_ratio = {warmup_ratio!r}
save_steps = {save_steps}
save_total_limit = {save_total_limit}

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=mllm_name_or_path,
    trust_remote_code=True,
    padding_side="right",
)

model = dict(
    type=Qwen3VLForMotionAtlas,
    torch_dtype={_torch_dtype_expr(model_cfg.get("torch_dtype", "bfloat16"))},
    **{pformat(model_kwargs, sort_dicts=False)},
)

train_dataset = dict(
    type=MotionAtlasQwen3VLDataset,
    **{pformat(data_kwargs, sort_dicts=False)},
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    persistent_workers=dataloader_num_workers > 0,
    prefetch_factor=4 if dataloader_num_workers > 0 else None,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property="modality_length",
        per_device_batch_size=batch_size * accumulative_counts,
    ),
    collate_fn=dict(type=qwen3vl_motionatlas_collect, tokenizer_cfg=tokenizer),
)

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=AdamW, lr=lr, betas=(0.9, 0.999), weight_decay=0),
    clip_grad=dict(max_norm=1, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale="dynamic",
    dtype=torch.bfloat16,
)

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-5, by_epoch=True, begin=0,
         end=warmup_ratio * max_epochs, convert_to_iter_based=True),
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=True,
         begin=warmup_ratio * max_epochs, end=max_epochs, convert_to_iter_based=True),
]

{train_loop_line}
custom_hooks = []
default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=100),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, save_optimizer=False, by_epoch=False,
                    interval=save_steps, max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)
env_cfg = dict(cudnn_benchmark=False, mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0), dist_cfg=dict(backend="nccl"))
visualizer = None
log_level = "INFO"
load_from = {model_cfg.get("load_from")!r}
resume = {bool(model_cfg.get("resume", False))}
randomness = dict(seed={seed}, deterministic=False)
log_processor = dict(by_epoch=False)
'''


def write_generated_config(cfg: dict[str, Any], source_config: str | Path) -> Path:
    work_dir = Path(cfg.get("train", {}).get("work_dir", "work_dirs/qwen3vl_4b_motionatlas"))
    out_dir = work_dir / ".generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(source_config).stem + "_mmengine.py")
    out_path.write_text(build_mmengine_config_text(cfg, source_config), encoding="utf-8")
    return out_path

