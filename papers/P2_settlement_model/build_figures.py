"""
Build core Paper 2 figures from finalized experiment metrics (E007-E013).

Run from repo root:
    py papers/P2_settlement_model/build_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent
PAPER_DIR = REPO_ROOT / "papers" / "P2_settlement_model"
FIG_DIR = PAPER_DIR / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Consolidated metrics from finalized experiment result files.
    metrics = pd.DataFrame(
        [
            ("E007", 0.659, 0.656, 0.318, 0.314, -0.095),
            ("E008", 0.685, 0.695, 0.345, 0.379, -0.153),
            ("E009", 0.664, 0.643, 0.337, 0.312, -0.266),
            ("E010", 0.711, 0.699, 0.384, 0.380, -0.142),
            ("E011", 0.725, 0.716, 0.447, 0.408, -0.169),
            ("E012", 0.730, 0.724, 0.420, 0.413, -0.160),
            ("E013", 0.768, 0.742, 0.507, 0.458, -0.229),
        ],
        columns=["experiment", "xgb_auc", "rf_auc", "xgb_tss", "rf_tss", "rho_taut"],
    )
    metrics.to_csv(PAPER_DIR / "tables_experiment_progression.csv", index=False)

    x = np.arange(len(metrics))
    labels = metrics["experiment"].tolist()

    # Figure 3: AUC/TSS progression.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("E007-E013 Progression: Performance Under Bias-Correction Pipeline", fontsize=12)

    w = 0.37
    axes[0].bar(x - w / 2, metrics["xgb_auc"], width=w, label="XGBoost", color="#E53935", alpha=0.85)
    axes[0].bar(x + w / 2, metrics["rf_auc"], width=w, label="RandomForest", color="#1E88E5", alpha=0.85)
    axes[0].axhline(0.75, color="green", linestyle="--", linewidth=1.2, label="MVR AUC (0.75)")
    axes[0].set_title("Spatial AUC by Experiment")
    axes[0].set_ylabel("AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.55, 0.82)
    axes[0].legend(fontsize=8)

    axes[1].bar(x - w / 2, metrics["xgb_tss"], width=w, label="XGBoost", color="#43A047", alpha=0.85)
    axes[1].bar(x + w / 2, metrics["rf_tss"], width=w, label="RandomForest", color="#6D4C41", alpha=0.85)
    axes[1].axhline(0.40, color="purple", linestyle="--", linewidth=1.2, label="Secondary target TSS (0.40)")
    axes[1].set_title("TSS by Experiment")
    axes[1].set_ylabel("TSS")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylim(0.20, 0.56)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_auc_tss_progression.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Figure 5: Challenge 1 rho progression.
    fig2, ax2 = plt.subplots(figsize=(8.5, 4.5))
    colors = ["#2E7D32" if v <= 0 else "#F9A825" for v in metrics["rho_taut"]]
    ax2.bar(x, metrics["rho_taut"], color=colors, alpha=0.9)
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.axhline(0.3, color="red", linestyle="--", linewidth=1.2, label="Tautology risk threshold (rho=0.3)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Spearman rho (suitability vs volcano distance)")
    ax2.set_title("Challenge 1 Tautology Test Across Experiments")
    ax2.set_ylim(-0.33, 0.33)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig5_tautology_rho_progression.png", dpi=200, bbox_inches="tight")
    plt.close()

    # Copy key E013 visual outputs into paper figure folder for manuscript linkage.
    e013_figs = {
        REPO_ROOT / "experiments" / "E013_settlement_model_v7" / "results" / "model_cv_results.png":
            FIG_DIR / "fig4_e013_cv_by_fold.png",
        REPO_ROOT / "experiments" / "E013_settlement_model_v7" / "results" / "sweep_heatmap.png":
            FIG_DIR / "fig2_hybrid_sweep_heatmap.png",
    }
    for src, dst in e013_figs.items():
        if src.exists():
            dst.write_bytes(src.read_bytes())

    print("Saved figures:")
    print(f"  {FIG_DIR / 'fig2_hybrid_sweep_heatmap.png'}")
    print(f"  {FIG_DIR / 'fig3_auc_tss_progression.png'}")
    print(f"  {FIG_DIR / 'fig4_e013_cv_by_fold.png'}")
    print(f"  {FIG_DIR / 'fig5_tautology_rho_progression.png'}")
    print(f"Saved table: {PAPER_DIR / 'tables_experiment_progression.csv'}")


if __name__ == "__main__":
    main()
