#!/usr/bin/env python3
"""
ARIMA Diagnostic & Comparison Plots

Generates:
  1. IV Forecast fan chart (point forecast + 95% CI over time)
  2. Standard vs ARIMA equity curve overlay
  3. ARIMA signal analysis (entries taken/skipped, early exits)
  4. Model diagnostics: residual ACF, fitted vs actual
  5. Comparative bar chart: Sharpe / Alpha improvement from ARIMA
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
DOCS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images")

DARK_BG   = "#0d1117"
DARK_FACE = "#161b22"
COLORS    = ["#58a6ff", "#f78166", "#3fb950", "#d2a8ff", "#ff7b72", "#79c0ff"]


def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def dark_style(ax):
    ax.set_facecolor(DARK_FACE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("white")
    ax.grid(True, alpha=0.1, color="#30363d")


# ==========================================================================
# 1. IV Forecast Fan Chart
# ==========================================================================
def plot_iv_forecast_fan(strategy_name, forecast_df, equity_df):
    """Plot IV with ARIMA forecast bands overlaid."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor=DARK_BG,
                                     gridspec_kw={"height_ratios": [2, 1]})
    dark_style(ax1)
    dark_style(ax2)

    # Top: IV series with forecast overlay
    if "IV" in equity_df.columns:
        ax1.plot(equity_df["Day"], equity_df["IV"], color="#58a6ff",
                 linewidth=0.8, alpha=0.7, label="Actual IV")

    if not forecast_df.empty:
        days = forecast_df["Day"]
        ax1.plot(days, forecast_df["IV_Forecast"], color="#f78166",
                 linewidth=1.2, label="ARIMA Forecast", zorder=5)

        if "IV_Lower95" in forecast_df.columns:
            ax1.fill_between(days, forecast_df["IV_Lower95"],
                            forecast_df["IV_Upper95"],
                            color="#f78166", alpha=0.15, label="95% CI")

        # Direction arrows
        for _, row in forecast_df.iterrows():
            if row["IV_Direction"] == 1:
                ax1.annotate("", xy=(row["Day"], row["IV_Forecast"]),
                            xytext=(row["Day"], row["IV_Forecast"] - 0.3),
                            arrowprops=dict(arrowstyle="->", color="#3fb950", lw=1.5))
            elif row["IV_Direction"] == -1:
                ax1.annotate("", xy=(row["Day"], row["IV_Forecast"]),
                            xytext=(row["Day"], row["IV_Forecast"] + 0.3),
                            arrowprops=dict(arrowstyle="->", color="#ff7b72", lw=1.5))

    ax1.set_title(f"{strategy_name} — ARIMA IV Forecast", fontsize=12, pad=10)
    ax1.set_ylabel("Implied Volatility (VIX-like)")
    ax1.legend(loc="upper right", fontsize=8, facecolor=DARK_FACE,
               edgecolor="#30363d", labelcolor="white")

    # Bottom: Confidence over time
    if not forecast_df.empty and "IV_Confidence" in forecast_df.columns:
        colors = ["#3fb950" if d == 1 else "#ff7b72" if d == -1 else "#8b949e"
                  for d in forecast_df["IV_Direction"]]
        ax2.bar(forecast_df["Day"], forecast_df["IV_Confidence"],
                color=colors, alpha=0.7, width=3)
        ax2.set_ylabel("Forecast Confidence")
        ax2.set_ylim(0, 1)

    ax2.set_xlabel("Trading Day")

    plt.tight_layout()
    return fig


# ==========================================================================
# 2. Standard vs ARIMA Equity Curves
# ==========================================================================
def plot_equity_comparison(strategy_name, std_equity, arima_equity):
    """Overlay standard and ARIMA-enhanced equity curves."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), facecolor=DARK_BG,
                                     gridspec_kw={"height_ratios": [3, 1]})
    dark_style(ax1)
    dark_style(ax2)

    if "Benchmark" in std_equity.columns:
        ax1.plot(std_equity["Day"], std_equity["Benchmark"], color="#8b949e",
                 linewidth=1, alpha=0.5, linestyle="--", label="S&P 500")

    ax1.plot(std_equity["Day"], std_equity["Equity"], color="#58a6ff",
             linewidth=1.2, label="Standard", alpha=0.8)
    ax1.plot(arima_equity["Day"], arima_equity["Equity"], color="#3fb950",
             linewidth=1.5, label="ARIMA-Enhanced", zorder=5)

    ax1.set_title(f"{strategy_name} — Standard vs ARIMA-Enhanced", fontsize=12, pad=10)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left", fontsize=9, facecolor=DARK_FACE,
               edgecolor="#30363d", labelcolor="white")

    # Relative improvement
    min_len = min(len(std_equity), len(arima_equity))
    if min_len > 0:
        diff = (arima_equity["Equity"].values[:min_len] -
                std_equity["Equity"].values[:min_len])
        days = std_equity["Day"].values[:min_len]
        ax2.fill_between(days, 0, diff,
                        where=diff >= 0, color="#3fb950", alpha=0.4, label="ARIMA better")
        ax2.fill_between(days, 0, diff,
                        where=diff < 0, color="#ff7b72", alpha=0.4, label="Standard better")
        ax2.axhline(0, color="#30363d", linewidth=0.5)
        ax2.set_ylabel("ARIMA Edge ($)")
        ax2.legend(loc="upper right", fontsize=7, facecolor=DARK_FACE,
                   edgecolor="#30363d", labelcolor="white")

    ax2.set_xlabel("Trading Day")
    plt.tight_layout()
    return fig


# ==========================================================================
# 3. ARIMA Trade Analysis
# ==========================================================================
def plot_arima_trade_analysis(strategy_name, trades_df):
    """Analyse ARIMA signal impact on trades."""
    fig = plt.figure(figsize=(14, 8), facecolor=DARK_BG)
    gs = GridSpec(2, 2, figure=fig)

    # 1. PnL distribution by ARIMA direction
    ax1 = fig.add_subplot(gs[0, 0])
    dark_style(ax1)
    for direction, label, color in [(-1, "Bearish IV", "#ff7b72"),
                                      (0, "Neutral", "#8b949e"),
                                      (1, "Bullish IV", "#3fb950")]:
        mask = trades_df["ARIMA_Direction"] == direction
        if mask.any():
            ax1.hist(trades_df.loc[mask, "PnL"], bins=20, alpha=0.6,
                    color=color, label=f"{label} ({mask.sum()})")
    ax1.set_title("PnL by ARIMA IV Direction", fontsize=10)
    ax1.set_xlabel("Trade PnL ($)")
    ax1.legend(fontsize=7, facecolor=DARK_FACE, edgecolor="#30363d", labelcolor="white")

    # 2. Confidence vs PnL scatter
    ax2 = fig.add_subplot(gs[0, 1])
    dark_style(ax2)
    if "ARIMA_Confidence" in trades_df.columns and "PnL" in trades_df.columns:
        colors = ["#3fb950" if p > 0 else "#ff7b72" for p in trades_df["PnL"]]
        ax2.scatter(trades_df["ARIMA_Confidence"], trades_df["PnL"],
                   c=colors, s=30, alpha=0.7)
        ax2.axhline(0, color="#30363d", linewidth=0.5)
        ax2.set_title("ARIMA Confidence vs Trade PnL", fontsize=10)
        ax2.set_xlabel("Forecast Confidence")
        ax2.set_ylabel("PnL ($)")

    # 3. Early exit analysis
    ax3 = fig.add_subplot(gs[1, 0])
    dark_style(ax3)
    if "EarlyExit" in trades_df.columns:
        early = trades_df[trades_df["EarlyExit"] == "YES"]
        normal = trades_df[trades_df["EarlyExit"] == "NO"]
        data = []
        labels = []
        if len(normal) > 0:
            data.append(normal["PnL"].values)
            labels.append(f"Normal Exit ({len(normal)})")
        if len(early) > 0:
            data.append(early["PnL"].values)
            labels.append(f"ARIMA Early ({len(early)})")
        if data:
            bp = ax3.boxplot(data, labels=labels, patch_artist=True,
                           boxprops=dict(facecolor=DARK_FACE, color="#58a6ff"),
                           medianprops=dict(color="#f78166"),
                           whiskerprops=dict(color="#58a6ff"),
                           capprops=dict(color="#58a6ff"),
                           flierprops=dict(markeredgecolor="#58a6ff", markersize=3))
    ax3.set_title("PnL: Normal vs Early Exit", fontsize=10)
    ax3.set_ylabel("PnL ($)")

    # 4. Predicted vs Actual regime
    ax4 = fig.add_subplot(gs[1, 1])
    dark_style(ax4)
    if "PredictedRegime" in trades_df.columns and "Regime" in trades_df.columns:
        correct = (trades_df["PredictedRegime"] == trades_df["Regime"]).sum()
        total = len(trades_df)
        incorrect = total - correct
        if total > 0:
            ax4.bar(["Correct", "Incorrect"], [correct, incorrect],
                   color=["#3fb950", "#ff7b72"], alpha=0.7)
            pct = 100.0 * correct / total if total > 0 else 0
            ax4.set_title(f"Regime Prediction Accuracy: {pct:.0f}%", fontsize=10)
    ax4.set_ylabel("Trade Count")

    fig.suptitle(f"{strategy_name} — ARIMA Signal Analysis",
                 fontsize=13, color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ==========================================================================
# 4. ARIMA Model Diagnostics
# ==========================================================================
def plot_arima_diagnostics(strategy_name, diag_df):
    """Plot fitted vs actual and residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), facecolor=DARK_BG)
    for ax in axes.flat:
        dark_style(ax)

    has_residuals = "Residual" in diag_df.columns and diag_df["Residual"].notna().any()
    fitted_mask = diag_df["Fitted"].notna()
    forecast_mask = diag_df["Forecast_5d"].notna()

    # 1. Fitted vs Actual
    ax = axes[0, 0]
    actual = diag_df.loc[fitted_mask & diag_df["IV"].notna()]
    if len(actual) > 0:
        ax.plot(actual["t"], actual["IV"], color="#58a6ff", linewidth=0.8,
                alpha=0.7, label="Actual")
        ax.plot(actual["t"], actual["Fitted"], color="#f78166", linewidth=0.8,
                label="Fitted")
    # Forecast
    fc = diag_df.loc[forecast_mask]
    if len(fc) > 0:
        ax.plot(fc["t"], fc["Forecast_5d"], color="#3fb950", linewidth=1.2,
                linestyle="--", label="Forecast")
        if "Lower_95" in fc.columns:
            ax.fill_between(fc["t"], fc["Lower_95"], fc["Upper_95"],
                           color="#3fb950", alpha=0.15)
    ax.set_title("Fitted vs Actual + Forecast", fontsize=10)
    ax.legend(fontsize=7, facecolor=DARK_FACE, edgecolor="#30363d", labelcolor="white")

    # 2. Residuals
    ax = axes[0, 1]
    if has_residuals:
        res = diag_df.loc[diag_df["Residual"].notna()]
        ax.plot(res["t"], res["Residual"], color="#d2a8ff", linewidth=0.6, alpha=0.7)
        ax.axhline(0, color="#30363d", linewidth=0.5)
    ax.set_title("Residuals", fontsize=10)

    # 3. Residual histogram
    ax = axes[1, 0]
    if has_residuals:
        res_vals = diag_df.loc[diag_df["Residual"].notna(), "Residual"]
        ax.hist(res_vals, bins=40, color="#d2a8ff", alpha=0.6, edgecolor="none")
    ax.set_title("Residual Distribution", fontsize=10)
    ax.set_xlabel("Residual")

    # 4. Residual ACF (manual)
    ax = axes[1, 1]
    if has_residuals:
        res_vals = diag_df.loc[diag_df["Residual"].notna(), "Residual"].values
        n = len(res_vals)
        mean = np.mean(res_vals)
        var = np.sum((res_vals - mean) ** 2)
        max_lag = min(30, n - 1)
        acf = []
        for k in range(max_lag + 1):
            c = np.sum((res_vals[k:] - mean) * (res_vals[:n-k] - mean)) / var
            acf.append(c)
        ax.bar(range(len(acf)), acf, color="#79c0ff", alpha=0.7)
        # Significance bounds
        ci = 1.96 / np.sqrt(n)
        ax.axhline(ci, color="#f78166", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(-ci, color="#f78166", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axhline(0, color="#30363d", linewidth=0.5)
    ax.set_title("Residual ACF", fontsize=10)
    ax.set_xlabel("Lag")

    fig.suptitle(f"{strategy_name} — ARIMA Model Diagnostics",
                 fontsize=13, color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ==========================================================================
# 5. Composite Comparison: All Strategies Sharpe/Alpha improvement
# ==========================================================================
def plot_comparative_improvement():
    """Bar chart comparing standard vs ARIMA across all strategies."""
    std_files  = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_equity.csv")))
    arima_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_arima_equity.csv")))

    # Filter out ARIMA files from standard
    std_files = [f for f in std_files if "_arima_" not in f]

    if not std_files or not arima_files:
        return None

    strategies = []
    std_returns = []
    arima_returns = []

    for sf, af in zip(std_files, arima_files):
        name = os.path.basename(sf).split("_equity.csv")[0].split("_", 1)[-1].replace("_", " ")
        sd = pd.read_csv(sf)
        ad = pd.read_csv(af)
        if len(sd) > 0 and len(ad) > 0:
            s_ret = (sd["Equity"].iloc[-1] / sd["Equity"].iloc[0] - 1) * 100
            a_ret = (ad["Equity"].iloc[-1] / ad["Equity"].iloc[0] - 1) * 100
            strategies.append(name)
            std_returns.append(s_ret)
            arima_returns.append(a_ret)

    if not strategies:
        return None

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=DARK_BG)
    dark_style(ax)

    x = np.arange(len(strategies))
    width = 0.35

    bars1 = ax.bar(x - width/2, std_returns, width, color="#58a6ff", alpha=0.8,
                   label="Standard")
    bars2 = ax.bar(x + width/2, arima_returns, width, color="#3fb950", alpha=0.8,
                   label="ARIMA-Enhanced")

    ax.set_xlabel("Strategy")
    ax.set_ylabel("Total Return (%)")
    ax.set_title("Standard vs ARIMA-Enhanced: Total Return Comparison",
                 fontsize=13, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=25, ha="right", fontsize=8)
    ax.legend(facecolor=DARK_FACE, edgecolor="#30363d", labelcolor="white")
    ax.axhline(0, color="#30363d", linewidth=0.5)

    # Value labels
    for bar in list(bars1) + list(bars2):
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f"{height:.1f}%", ha="center", va=va, fontsize=7, color="#c9d1d9")

    plt.tight_layout()
    return fig


# ==========================================================================
# Main
# ==========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  ARIMA Diagnostic & Comparison Plots")
    print("=" * 60)
    ensure_dirs()

    # Find all strategies with ARIMA data
    forecast_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_arima_forecasts.csv")))
    if not forecast_files:
        print("\nNo ARIMA data found. Run ./build/backtest first.")
        sys.exit(1)

    print(f"\nFound {len(forecast_files)} strategies with ARIMA data.\n")

    for fpath in forecast_files:
        base = os.path.basename(fpath).replace("_arima_forecasts.csv", "")
        strategy_name = base.split("_", 1)[-1].replace("_", " ")
        print(f"  Processing: {strategy_name}")

        forecast_df = pd.read_csv(fpath)

        # Load equity curves
        std_equity_file = os.path.join(OUTPUT_DIR, base + "_equity.csv")
        arima_equity_file = os.path.join(OUTPUT_DIR, base + "_arima_equity.csv")
        arima_trades_file = os.path.join(OUTPUT_DIR, base + "_arima_trades.csv")
        arima_diag_file = os.path.join(OUTPUT_DIR, base + "_arima_iv_diag.csv")

        std_equity = pd.read_csv(std_equity_file) if os.path.exists(std_equity_file) else pd.DataFrame()
        arima_equity = pd.read_csv(arima_equity_file) if os.path.exists(arima_equity_file) else pd.DataFrame()
        arima_trades = pd.read_csv(arima_trades_file) if os.path.exists(arima_trades_file) else pd.DataFrame()
        arima_diag = pd.read_csv(arima_diag_file) if os.path.exists(arima_diag_file) else pd.DataFrame()

        # 1. IV Forecast fan chart
        if not forecast_df.empty and not arima_equity.empty:
            fig = plot_iv_forecast_fan(strategy_name, forecast_df, arima_equity)
            out = os.path.join(DOCS_DIR, f"{base}_arima_forecast.png")
            fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved: {out}")

        # 2. Equity comparison
        if not std_equity.empty and not arima_equity.empty:
            fig = plot_equity_comparison(strategy_name, std_equity, arima_equity)
            out = os.path.join(DOCS_DIR, f"{base}_arima_equity_comparison.png")
            fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved: {out}")

        # 3. Trade analysis
        if not arima_trades.empty and len(arima_trades) > 0:
            fig = plot_arima_trade_analysis(strategy_name, arima_trades)
            out = os.path.join(DOCS_DIR, f"{base}_arima_trade_analysis.png")
            fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved: {out}")

        # 4. Model diagnostics
        if not arima_diag.empty:
            fig = plot_arima_diagnostics(strategy_name, arima_diag)
            out = os.path.join(DOCS_DIR, f"{base}_arima_diagnostics.png")
            fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            print(f"    Saved: {out}")

    # 5. Comparative improvement chart
    print("\n  Generating comparative improvement chart...")
    fig = plot_comparative_improvement()
    if fig:
        out = os.path.join(DOCS_DIR, "arima_comparison.png")
        fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"    Saved: {out}")

    print(f"\nDone! All ARIMA plots saved to {DOCS_DIR}/")
