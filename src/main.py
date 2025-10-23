"""src/main.py
Top-level orchestrator.  Merely forwards to ``src.train`` so that Hydra
scopes stay cleanly separated.  *Does not* run evaluation.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

################################################################################
#                             main                                             #
################################################################################

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Ensure the output directory exists -------------------------------------
    out_root = Path(hydra.utils.to_absolute_path(cfg.results_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    overrides: List[str] = [
        f"run={cfg.run.run_id}",  # select the **same** run variant
        f"results_dir={cfg.results_dir}",
    ]
    if cfg.trial_mode:
        overrides.append("trial_mode=true")
        overrides.append("wandb.mode=disabled")
    else:
        overrides.append(f"wandb.mode={cfg.wandb.mode}")

    cmd = [sys.executable, "-u", "-m", "src.train", *overrides]
    print("[main] Launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
