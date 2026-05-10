import os
import matplotlib.pyplot as plt

from src.config import OUTPUT_DIR


def plot_growth(backtests, title="Portfolio Growth Comparison"):
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, bt in backtests.items():
        ax.plot(bt["values"], label=name, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Test period index")
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "portfolio_growth.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"Saved: {path}")


def plot_metrics(results_df):
    cols = ["Sharpe", "Sortino", "MaxDD", "CVaR_5pct", "Final Wealth"]
    plot_df = results_df[cols].copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", ax=ax)

    ax.set_title("Strategy Comparison: Risk and Return Metrics")
    ax.grid(alpha=0.25)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "metrics_comparison.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"Saved: {path}")


def plot_regimes(regime_df):
    if regime_df is None or regime_df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    regime_df.plot(ax=ax, linewidth=2)

    ax.set_title("HMM Regime Probabilities")
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability")
    ax.grid(alpha=0.25)

    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "regime_probabilities.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)

    print(f"Saved: {path}")


def validation_dashboard(results_df, stress_df, ablation_df, hypothesis_df, backtests):
    from matplotlib.gridspec import GridSpec

    final_name = "Final RL: Regime + Risk"
    rl_stats = results_df.loc[final_name]

    fig = plt.figure(figsize=(20, 14), constrained_layout=True)

    try:
        gs = GridSpec(5, 4, figure=fig)

        fig.suptitle(
            "Portfolio Strategy Validation Dashboard",
            fontsize=20,
            fontweight="bold"
        )

        kpis = [
            ("Final RL Sharpe", rl_stats["Sharpe"]),
            ("Final RL MaxDD", rl_stats["MaxDD"]),
            ("Final RL CVaR", rl_stats["CVaR_5pct"]),
            ("Final Wealth", rl_stats["Final Wealth"]),
        ]

        for i, (label, val) in enumerate(kpis):
            ax = fig.add_subplot(gs[0, i])
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"{label}\n\n{val:.3f}",
                ha="center",
                va="center",
                fontsize=14,
                bbox=dict(
                    boxstyle="round,pad=0.6",
                    facecolor="#F8F9FA",
                    edgecolor="#2E86DE",
                    linewidth=2
                )
            )

        ax1 = fig.add_subplot(gs[1, 0:2])
        for name, bt in backtests.items():
            ax1.plot(bt["values"], label=name, linewidth=2)
        ax1.set_title("Portfolio Growth")
        ax1.grid(alpha=0.25)
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[1, 2:4])
        results_df[["Sharpe", "Sortino", "Calmar"]].plot(kind="bar", ax=ax2)
        ax2.set_title("Risk-Adjusted Metrics")
        ax2.grid(alpha=0.2)
        ax2.tick_params(axis="x", rotation=30)

        ax3 = fig.add_subplot(gs[2, 0:2])
        if stress_df is not None and not stress_df.empty:
            stress_df["Stress Sharpe"].plot(kind="bar", ax=ax3)
            ax3.set_title("Stress Period Sharpe")
            ax3.grid(alpha=0.2)
            ax3.tick_params(axis="x", rotation=30)
        else:
            ax3.text(0.5, 0.5, "Not enough stress observations", ha="center", va="center")
            ax3.set_title("Stress Period Sharpe")

        ax4 = fig.add_subplot(gs[2, 2:4])
        ablation_df["Sharpe"].plot(kind="bar", ax=ax4)
        ax4.set_title("Ablation Study: Sharpe")
        ax4.grid(alpha=0.2)
        ax4.tick_params(axis="x", rotation=30)

        ax5 = fig.add_subplot(gs[3, :])
        ax5.axis("off")

        hyp_text = "\n".join([
            f"{r['Hypothesis']}: {r['Pass/Fail']} — {r['Evidence']}"
            for _, r in hypothesis_df.iterrows()
        ])

        ax5.text(
            0.01,
            0.85,
            hyp_text,
            fontsize=12,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#F8F9FA",
                edgecolor="black"
            )
        )
        ax5.set_title("Hypothesis Experiment Summary")

        ax6 = fig.add_subplot(gs[4, :])
        ax6.axis("off")

        best = results_df["Sharpe"].idxmax()

        txt = (
            f"Best Sharpe Strategy: {best}\n"
            f"Final RL Wealth: {rl_stats['Final Wealth']:.2f}x\n"
            f"Final RL Max Drawdown: {rl_stats['MaxDD']:.2%}\n"
            "Interpretation: The final model is evaluated against RL ablations and traditional baselines. "
            "Because the dataset is yearly, conclusions should be framed as annual tactical allocation, not daily trading."
        )

        ax6.text(
            0.01,
            0.75,
            txt,
            fontsize=13,
            bbox=dict(
                boxstyle="round,pad=0.6",
                facecolor="#F8F9FA",
                edgecolor="#27AE60",
                linewidth=2
            )
        )
        ax6.set_title("Executive Summary")

        path = os.path.join(OUTPUT_DIR, "validation_dashboard.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")

        plt.show()
        print(f"Saved: {path}")

    finally:
        plt.close(fig)
