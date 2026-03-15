#!/usr/bin/env python3
"""
3D Visualization Suite for the Options Backtesting Engine.

Generates:
  1. 3D Volatility Regime scatter: IV × PnL × S&P 500 (colour-coded by regime)
  2. PnL surface plots: Spot × IV → PnL for each strategy
  3. Equity curve comparison with drawdown overlay
  4. Regime distribution pie chart

Requirements:  pip install matplotlib numpy pandas
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import ListedColormap

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")

REGIME_COLOURS = {
    "Low":       "#2ecc71",
    "Medium":    "#3498db",
    "High":      "#e74c3c",
    "Crush":     "#9b59b6",
    "Expansion": "#f39c12",
}


def ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)


# ---- 1. 3D Volatility Regime Scatter: IV × PnL × S&P 500 ----------------
def plot_3d_regime_scatter():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_3d_surface.csv")))
    if not files:
        print("  No 3D surface CSVs found – skipping.")
        return

    for fpath in files:
        df = pd.read_csv(fpath)
        if df.empty:
            continue

        name = os.path.basename(fpath).replace("_3d_surface.csv", "")
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        for regime, colour in REGIME_COLOURS.items():
            mask = df["Regime"] == regime
            if mask.sum() == 0:
                continue
            ax.scatter(
                df.loc[mask, "IV"],
                df.loc[mask, "SP500"],
                df.loc[mask, "PnL"],
                c=colour, label=regime, alpha=0.6, s=12, edgecolors="none"
            )

        ax.set_xlabel("Implied Volatility (VIX)", fontsize=11, labelpad=10)
        ax.set_ylabel("S&P 500 Level", fontsize=11, labelpad=10)
        ax.set_zlabel("Cumulative PnL ($)", fontsize=11, labelpad=10)
        ax.set_title(f"3D Volatility Regime Map – {name.replace('_', ' ')}",
                     fontsize=13, pad=20)
        ax.legend(loc="upper left", fontsize=9)
        ax.view_init(elev=25, azim=135)

        out = os.path.join(PLOT_DIR, f"{name}_3d_regime.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---- 2. PnL Surface (Spot × IV → PnL) -----------------------------------
def plot_pnl_surfaces():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_pnl_grid.csv")))
    if not files:
        print("  No PnL grid CSVs found – skipping.")
        return

    for fpath in files:
        df = pd.read_csv(fpath)
        if df.empty:
            continue

        name = os.path.basename(fpath).replace("_pnl_grid.csv", "")
        spots = np.sort(df["Spot"].unique())
        ivs   = np.sort(df["IV"].unique())
        X, Y  = np.meshgrid(spots, ivs)
        Z     = df.pivot_table(index="IV", columns="Spot", values="PnL").values

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(X, Y * 100, Z, cmap="RdYlGn",
                               edgecolor="none", alpha=0.85)
        fig.colorbar(surf, ax=ax, shrink=0.5, label="PnL ($)")

        ax.set_xlabel("S&P 500 Spot", fontsize=11, labelpad=10)
        ax.set_ylabel("Implied Volatility (%)", fontsize=11, labelpad=10)
        ax.set_zlabel("Strategy PnL ($)", fontsize=11, labelpad=10)
        ax.set_title(f"PnL Surface – {name.replace('_', ' ')}", fontsize=13, pad=20)
        ax.view_init(elev=30, azim=225)

        out = os.path.join(PLOT_DIR, f"{name}_pnl_surface.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


# ---- 3. Equity Curves with Drawdown Overlay ------------------------------
def plot_equity_curves():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_equity.csv")))
    if not files:
        print("  No equity CSVs found – skipping.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={"hspace": 0.05})
    benchmark_plotted = False

    for fpath in files:
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        name = os.path.basename(fpath).replace("_equity.csv", "").split("_", 1)[-1]
        label = name.replace("_", " ")

        ax1.plot(df["Day"], df["Equity"], linewidth=1.2, label=label)
        ax2.fill_between(df["Day"], -df["Drawdown"] * 100, alpha=0.3, label=label)

        if not benchmark_plotted:
            ax1.plot(df["Day"], df["Benchmark"], linewidth=1.5, color="black",
                     linestyle="--", label="S&P 500 (Buy & Hold)")
            benchmark_plotted = True

    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.set_title("Strategy Equity Curves vs S&P 500 Benchmark", fontsize=13)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Trading Day", fontsize=11)
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.grid(True, alpha=0.3)

    out = os.path.join(PLOT_DIR, "equity_curves_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- 4. Regime Distribution from Trade Logs ------------------------------
def plot_regime_distribution():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_trades.csv")))
    all_trades = []
    for f in files:
        df = pd.read_csv(f)
        if not df.empty:
            all_trades.append(df)

    if not all_trades:
        print("  No trade CSVs found – skipping.")
        return

    combined = pd.concat(all_trades, ignore_index=True)
    regime_counts = combined["Regime"].value_counts()

    colours = [REGIME_COLOURS.get(r, "#95a5a6") for r in regime_counts.index]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.pie(regime_counts.values, labels=regime_counts.index, colors=colours,
            autopct="%1.1f%%", startangle=140, textprops={"fontsize": 10})
    ax1.set_title("Trade Distribution by Vol Regime", fontsize=12)

    # PnL by regime
    regime_pnl = combined.groupby("Regime")["PnL"].sum()
    bars = ax2.bar(regime_pnl.index, regime_pnl.values,
                   color=[REGIME_COLOURS.get(r, "#95a5a6") for r in regime_pnl.index])
    ax2.set_title("Aggregate PnL by Vol Regime", fontsize=12)
    ax2.set_ylabel("Total PnL ($)", fontsize=11)
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")

    out = os.path.join(PLOT_DIR, "regime_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ---- 5. IV Time-Series with Regime Shading --------------------------------
def plot_iv_timeseries():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_equity.csv")))
    if not files:
        return

    df = pd.read_csv(files[0])
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["Day"], df["IV"], color="#2c3e50", linewidth=0.8, label="Implied Vol (VIX)")

    # Shade regime periods
    prev_regime = df["Regime"].iloc[0]
    start = df["Day"].iloc[0]
    for i in range(1, len(df)):
        if df["Regime"].iloc[i] != prev_regime or i == len(df) - 1:
            ax.axvspan(start, df["Day"].iloc[i],
                       color=REGIME_COLOURS.get(prev_regime, "#95a5a6"),
                       alpha=0.15)
            start = df["Day"].iloc[i]
            prev_regime = df["Regime"].iloc[i]

    ax.set_xlabel("Trading Day", fontsize=11)
    ax.set_ylabel("Implied Volatility (VIX)", fontsize=11)
    ax.set_title("Implied Volatility with Regime Classification", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    out = os.path.join(PLOT_DIR, "iv_timeseries_regimes.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ==========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Options Backtester – 3D Visualization Suite")
    print("=" * 60)
    ensure_dirs()

    print("\n[1/5] Generating 3D Volatility Regime Scatter plots...")
    plot_3d_regime_scatter()

    print("\n[2/5] Generating PnL Surface plots...")
    plot_pnl_surfaces()

    print("\n[3/5] Generating Equity Curve comparisons...")
    plot_equity_curves()

    print("\n[4/5] Generating Regime Distribution charts...")
    plot_regime_distribution()

    print("\n[5/5] Generating IV Time-Series with Regime shading...")
    plot_iv_timeseries()

    print(f"\nAll plots saved to: {PLOT_DIR}/")
    print("Done.")
