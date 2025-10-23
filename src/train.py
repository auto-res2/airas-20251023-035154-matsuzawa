"""src/train.py
Full training / optimisation loop for ONE run-id. Implements vanilla
BOIL and the proposed Stability-aware BOIL (S-BOIL).  Everything is
fully logged to WandB – absolutely no local JSON summaries are
produced.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
import numpy as np
import torch
import wandb
try:
    from botorch.fit import fit_gpytorch_mll as fit_gpytorch_model
except (ImportError, AttributeError):
    try:
        from botorch.fit import fit_gpytorch_model
    except (ImportError, AttributeError):
        try:
            from botorch.optim.fit import fit_gpytorch_mll_torch as fit_gpytorch_model
        except (ImportError, AttributeError):
            from botorch.optim import fit_gpytorch_mll as fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LinearRegression

from .model import build_model
from .preprocess import get_dataloaders

################################################################################
#                        Helper : sampling search-space                        #
################################################################################

def _log_uniform(low: float, high: float, rng: np.random.Generator) -> float:
    """Sample from a base-10 log-uniform distribution."""
    return float(10 ** rng.uniform(np.log10(low), np.log10(high)))


def _sample(space: Dict, rng: np.random.Generator) -> Dict:
    """Draw *one* random sample from an Optuna-style search-space dict."""
    out: Dict = {}
    for name, spec in space.items():
        t = spec["type"].lower()
        if t == "loguniform":
            out[name] = _log_uniform(spec["low"], spec["high"], rng)
        elif t == "uniform":
            out[name] = float(rng.uniform(spec["low"], spec["high"]))
        elif t == "int":
            out[name] = int(rng.integers(spec["low"], spec["high"] + 1))
        elif t == "categorical":
            out[name] = rng.choice(spec["choices"])
        else:
            raise ValueError(f"Unknown space type: {t}")
    return out


def _encode(sample: Dict, space: Dict) -> List[float]:
    """Normalise a sample to [0,1] – required by the GP surrogate."""
    vec: List[float] = []
    for name, spec in space.items():
        val = sample[name]
        t = spec["type"].lower()
        if t == "loguniform":
            low, high = math.log10(spec["low"]), math.log10(spec["high"])
            vec.append((math.log10(val) - low) / (high - low))
        elif t in {"uniform", "int"}:
            low, high = spec["low"], spec["high"]
            vec.append((float(val) - low) / (high - low))
        elif t == "categorical":
            idx = spec["choices"].index(val)
            vec.append(idx / (len(spec["choices"]) - 1))
    return vec

################################################################################
#                      Single training episode (inner loop)                    #
################################################################################

def _train_episode(
    sample: Dict,
    cfg: DictConfig,
    device: torch.device,
    trial_mode: bool,
    wb_run: wandb.sdk.wandb_run.Run,
    step_offset: int,
) -> Tuple[float, float, Dict]:
    """Trains an MLP with the *sampled* hyper-parameters.

    Returns
    -------
    best_val : float
        Best validation accuracy achieved in the episode.
    wall_time : float
        Wall-clock time consumed (in seconds).
    best_state : Dict
        ``state_dict`` of the best model (empty dict if nothing improved).
    """
    t0 = time.perf_counter()

    # ------------------------------------------------------------------ merge
    # We create a *local* copy of the configuration into which the sampled
    # values are injected.  This leaves the original cfg untouched and safe
    # for the following episodes.
    cfg_l = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg_l.model.hidden_units = int(sample["hidden_units"])
    cfg_l.training.learning_rate = float(sample["learning_rate"])
    budget = int(sample["training_iteration_budget"])
    epochs = 1 if trial_mode else budget * cfg.training.epochs_per_iteration

    train_dl, val_dl, _ = get_dataloaders(cfg.dataset, cfg.training)

    model = build_model(cfg_l.model).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg_l.training.learning_rate)

    best_val, best_state = 0.0, {}
    global_step = step_offset

    for epoch in range(1, epochs + 1):
        # --------------------------- training loop -----------------------
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for b_idx, (x, y) in enumerate(train_dl, 1):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

            wb_run.log({"batch_loss": loss.item(), "phase": "train"}, step=global_step)
            global_step += 1
            if trial_mode and b_idx >= 2:
                break

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        # --------------------------- validation -------------------------
        model.eval()
        v_loss, v_corr, v_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * y.size(0)
                v_corr += (logits.argmax(1) == y).sum().item()
                v_tot += y.size(0)
        val_loss = v_loss / max(v_tot, 1)
        val_acc = v_corr / max(v_tot, 1)

        if val_acc > best_val:
            best_val = val_acc
            best_state = deepcopy(model.state_dict())

        wb_run.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "hyper/learning_rate": cfg_l.training.learning_rate,
                "hyper/hidden_units": cfg_l.model.hidden_units,
                "hyper/epochs": epochs,
            },
            step=global_step,
        )

        if trial_mode:
            break

    wall = time.perf_counter() - t0
    return best_val, wall, best_state

################################################################################
#                                    Main                                      #
################################################################################

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:  # noqa: C901 – function is inevitably long
    # ---------------------- trial-mode soft overrides -------------------------
    if cfg.trial_mode:
        cfg.wandb.mode = "disabled"
        cfg.experiment.total_evaluations = 1
        cfg.experiment.n_init_points = 1
        cfg.training.epochs_per_iteration = 1
        cfg.optuna.n_trials = 0

    # ----------------------------- output dirs --------------------------------
    root_dir = Path(hydra.utils.to_absolute_path(cfg.results_dir))
    run_dir = root_dir / cfg.run.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save *this* run's resolved configuration for full reproducibility
    OmegaConf.save(cfg, run_dir / "config.yaml")

    # Also place a global WandB pointer file at <results_dir>/config.yaml so
    # that the evaluation stage can easily recover entity/project details.
    global_cfg_path = root_dir / "config.yaml"
    if not global_cfg_path.exists():
        global_wb_cfg = OmegaConf.create({
            "wandb": {"entity": cfg.wandb.entity, "project": cfg.wandb.project}
        })
        OmegaConf.save(global_wb_cfg, global_cfg_path)

    # ------------------------------ WandB -------------------------------------
    wb_run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        id=cfg.run.run_id,
        dir=str(run_dir),
        mode=cfg.wandb.mode,
        resume="allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    if wb_run.mode != "disabled":
        print(f"[WandB] run URL: {wb_run.url}")

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed=42)
    space = cfg.optuna.search_space

    # ------------------- containers for BOIL / S-BOIL -------------------------
    X, Y, C, samples = [], [], [], []  # design-matrix, targets, costs, raw-samples
    best_global, best_state = -1.0, {}

    for ev in range(cfg.experiment.total_evaluations):
        # ------------------------ suggestion phase -------------------------
        if ev < cfg.experiment.n_init_points:
            candidate = _sample(space, rng)
        else:
            # Build GP on *normalised* X, Y
            X_t = torch.tensor(X, dtype=torch.double)
            Y_t = torch.tensor(Y, dtype=torch.double).unsqueeze(-1)
            gp = SingleTaskGP(X_t, Y_t)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_model(mll)
            best_y = max(Y)

            # Cheap linear regression surrogate for cost
            cost_model = LinearRegression().fit(np.array(X), np.log(np.array(C)))

            # Generate a candidate pool
            pool = [_sample(space, rng) for _ in range(512)]
            X_pool = torch.tensor([_encode(p, space) for p in pool], dtype=torch.double)

            post = gp.posterior(X_pool)
            mu, var = post.mean.squeeze(-1), post.variance.squeeze(-1).clamp_min(1e-12)
            sigma = var.sqrt()
            z = (mu - best_y) / sigma
            norm = torch.distributions.Normal(torch.zeros_like(z), torch.ones_like(z))
            ei = (mu - best_y) * norm.cdf(z) + sigma * norm.log_prob(z).exp()
            ei = ei.clamp_min(1e-12)

            cost_pred = torch.from_numpy(np.exp(cost_model.predict(X_pool.cpu().numpy()))).double()

            acquis = cfg.experiment.acquisition.lower()
            if acquis == "boil":
                acq = torch.log(ei) - torch.log(cost_pred)
            elif acquis == "s-boil":
                betas = torch.tensor([
                    p.get("beta", cfg.experiment.beta_default) for p in pool
                ], dtype=torch.double)
                acq = torch.log(ei) - betas * var - torch.log(cost_pred)
            else:
                raise ValueError(f"Unknown acquisition: {cfg.experiment.acquisition}")

            candidate = pool[int(torch.argmax(acq).item())]

        wb_run.log({f"suggested/{k}": v for k, v in candidate.items()}, step=ev)

        # -------------------- objective (training episode) ------------------
        v_acc, w_time, best_loc = _train_episode(
            candidate, cfg, device, cfg.trial_mode, wb_run, step_offset=ev * 10_000
        )

        X.append(_encode(candidate, space))
        Y.append(v_acc)
        C.append(w_time)
        samples.append(candidate)

        # ---------------------- bookkeeping --------------------------------
        if v_acc > best_global:
            best_global = v_acc
            best_state = deepcopy(best_loc)
            wb_run.summary["best_val_accuracy"] = best_global

        wb_run.log({
            "evaluation_id": ev,
            "val_acc": v_acc,
            "wall_time": w_time,
            "best_val_acc_so_far": best_global,
        }, step=(ev + 1) * 10_000)

        # first time hitting 0.8 accuracy ⇒ record time-to-target
        if v_acc >= 0.8 and "time_to_target" not in wb_run.summary:
            wb_run.summary["time_to_target"] = float(sum(C))
            wb_run.summary["evals_to_target"] = ev + 1

        if cfg.trial_mode:
            break

    wb_run.summary["total_wall_time"] = float(sum(C))

    # ------------------------------ artefacts ---------------------------------
    if best_state:
        torch.save(best_state, run_dir / "best_model.pt")
        wb_run.save(str(run_dir / "best_model.pt"), base_path=str(run_dir))

    wandb.finish()


if __name__ == "__main__":
    main()
