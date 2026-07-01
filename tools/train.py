#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path


def _patch_dist_timeout() -> None:
    import torch.distributed as dist

    original = dist.init_process_group

    def patched(*args, **kwargs):
        if "timeout" not in kwargs:
            seconds = int(os.environ.get("DIST_TIMEOUT_SECONDS", "3600"))
            kwargs["timeout"] = timedelta(seconds=seconds)
            print(f"[MotionAtlas] distributed timeout={seconds}s", flush=True)
        return original(*args, **kwargs)

    dist.init_process_group = patched


def _maybe_generate_config(argv: list[str]) -> list[str]:
    if len(argv) < 2:
        return argv
    config_path = Path(argv[1])
    if config_path.suffix.lower() not in {".yaml", ".yml"}:
        return argv

    from projects.motionatlas.build import write_generated_config
    from projects.motionatlas.config import extract_cfg_options, load_yaml_config

    tail, overrides = extract_cfg_options(argv[2:])
    cfg = load_yaml_config(config_path, overrides=overrides)
    timeout = cfg.get("distributed", {}).get("timeout_seconds")
    if timeout and "DIST_TIMEOUT_SECONDS" not in os.environ:
        os.environ["DIST_TIMEOUT_SECONDS"] = str(timeout)
    generated = write_generated_config(cfg, config_path)
    print(f"[MotionAtlas] generated MMEngine config: {generated}", flush=True)
    return [argv[0], str(generated)] + tail


def main() -> None:
    _patch_dist_timeout()
    sys.argv = _maybe_generate_config(sys.argv)
    from xtuner.tools.train import main as xtuner_train

    xtuner_train()


if __name__ == "__main__":
    main()
