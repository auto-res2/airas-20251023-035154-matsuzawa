"""src/evaluate.py
Independent evaluation / visualisation stage.  Fetches full histories
from WandB, creates per-run artefacts (JSON + plots) and an aggregated
comparison across runs.  **Not** called during training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats as sstats
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .model import build_model
from .preprocess import get_dataloaders

################################################################################
#                               utilities                                      #
################################################################################

def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False)


def _plot_learning_curve(df: pd.DataFrame, out_dir: Path, run_id: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    if "val_acc" in df.columns:
        ax.plot(df["val_acc"], label="Validation", linewidth=2)
    if "train_acc" in df.columns:
        ax.plot(df["train_acc"], label="Training", alpha=0.5)
    ax.set_xlabel("Logged step")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning curve – {run_id}")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_confusion(run_cfg: Dict, model_path: Path, out_dir: Path, run_id: str):
    if not model_path.exists():
        return None

    import torch

    cfg = OmegaConf.create(run_cfg) if not isinstance(run_cfg, Dict) else run_cfg

    # obtain ONLY test loader --------------------------------------------------
    _, _, test_dl = get_dataloaders(cfg.dataset, cfg.training)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg.model).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            preds.append(model(x).argmax(1).cpu().numpy())
            labels.append(y.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues")
    plt.title(f"Confusion matrix – {run_id}")
    plt.tight_layout()
    path = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(path)
    plt.close()
    return path

################################################################################
#                               main                                           #
################################################################################

def _load_global_wandb_cfg(results_dir: Path) -> Tuple[str, str]:
    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Could not locate global config at {cfg_path}. Each training run "
            "should have created it."
        )
    cfg = OmegaConf.load(cfg_path)
    return cfg.wandb.entity, cfg.wandb.project


def main() -> None:  # noqa: C901 – evaluation is necessarily multi-step
    parser = argparse.ArgumentParser(description="Comprehensive evaluation script")
    parser.add_argument("results_dir", type=str, help="Directory that contains run sub-folders")
    parser.add_argument("run_ids", type=str, help="JSON list, e.g. '[\"run-1\", \"run-2\"]'")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    run_ids: List[str] = json.loads(args.run_ids)

    entity, project = _load_global_wandb_cfg(results_dir)
    api = wandb.Api()

    aggregated_metrics: Dict[str, Dict] = {}
    generated_paths: List[str] = []

    # ---------------------------------------------------------------- per-run
    for run_id in run_ids:
        run_dir = results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run = api.run(f"{entity}/{project}/{run_id}")
        history_df = run.history()  # pandas DataFrame
        summary = dict(run.summary)  # final metrics
        config = dict(run.config)

        # Save metrics --------------------------------------------------------
        metrics_out = {
            "history": history_df.to_dict(orient="list"),
            "summary": summary,
            "config": config,
        }
        metrics_path = run_dir / "metrics.json"
        _save_json(metrics_path, metrics_out)
        generated_paths.append(str(metrics_path))

        # Learning curve ------------------------------------------------------
        lc_path = _plot_learning_curve(history_df, run_dir, run_id)
        generated_paths.append(str(lc_path))

        # Confusion matrix (if model available) -------------------------------
        best_model_path = run_dir / "best_model.pt"
        cm_path = _plot_confusion(config, best_model_path, run_dir, run_id)
        if cm_path is not None:
            generated_paths.append(str(cm_path))

        # Register for aggregation -------------------------------------------
        aggregated_metrics[run_id] = {
            "best_val_accuracy": summary.get("best_val_accuracy", float("nan")),
            "total_wall_time": summary.get("total_wall_time", float("nan")),
            "time_to_target": summary.get("time_to_target", float("nan")),
            "evals_to_target": summary.get("evals_to_target", float("nan")),
        }

    # --------------------------------------------------------- aggregated step
    cmp_dir = results_dir / "comparison"
    cmp_dir.mkdir(parents=True, exist_ok=True)

    # Save aggregated metrics JSON -------------------------------------------
    agg_metrics_path = cmp_dir / "aggregated_metrics.json"
    _save_json(agg_metrics_path, aggregated_metrics)
    generated_paths.append(str(agg_metrics_path))

    # Improvement-rate vs. baseline (take first run as baseline) -------------
    baseline_id = run_ids[0]
    baseline_acc = aggregated_metrics[baseline_id]["best_val_accuracy"]

    improvement_rates = {
        rid: (m["best_val_accuracy"] - baseline_acc) / baseline_acc if not np.isnan(baseline_acc) else np.nan
        for rid, m in aggregated_metrics.items()
    }

    # Bar chart of best validation accuracy ----------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=list(aggregated_metrics.keys()), y=[m["best_val_accuracy"] for m in aggregated_metrics.values()], ax=ax)
    ax.set_ylabel("Best validation accuracy")
    ax.set_xlabel("Run ID")
    for p, acc in zip(ax.patches, [m["best_val_accuracy"] for m in aggregated_metrics.values()]):
        ax.annotate(f"{acc:.3f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    bar_path = cmp_dir / "comparison_best_val_accuracy_bar_chart.pdf"
    plt.savefig(bar_path)
    plt.close()
    generated_paths.append(str(bar_path))

    # Box plot of improvement rates -----------------------------------------
    if len(run_ids) > 1 and not np.isnan(baseline_acc):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=list(improvement_rates.keys()), y=list(improvement_rates.values()), ax=ax, palette="viridis")
        ax.set_ylabel("Improvement rate vs. baseline")
        ax.set_xlabel("Run ID")
        for p, imp in zip(ax.patches, list(improvement_rates.values())):
            ax.annotate(f"{imp:.2%}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                        ha="center", va="bottom", fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        imp_path = cmp_dir / "comparison_improvement_rate_bar_chart.pdf"
        plt.savefig(imp_path)
        plt.close()
        generated_paths.append(str(imp_path))

    # Simple statistical significance: t-test on best accuracies -------------
    if len(run_ids) > 1:
        accs = np.array([aggregated_metrics[r]["best_val_accuracy"] for r in run_ids])
        # pair-wise t-test against baseline
        pvals = {
            rid: float(sstats.ttest_ind([aggregated_metrics[baseline_id]["best_val_accuracy"]],
                                        [aggregated_metrics[rid]["best_val_accuracy"]],
                                        equal_var=False).pvalue)
            for rid in run_ids[1:]
        }
        pval_path = cmp_dir / "comparison_pvalues.json"
        _save_json(pval_path, pvals)
        generated_paths.append(str(pval_path))

    # ------------------------------- stdout ----------------------------------
    print("\n".join(generated_paths))


if __name__ == "__main__":
    main()
