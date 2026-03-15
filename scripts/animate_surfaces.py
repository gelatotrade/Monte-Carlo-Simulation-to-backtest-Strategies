#!/usr/bin/env python3
"""
Animated 3D PnL Surface Visualization — Time-Evolving Volatility Regime Surfaces

Generates animated GIFs showing how each strategy's PnL surface morphs in real-time
as the underlying price and implied volatility move minute-by-minute.

The animation shows:
  - The 3D PnL surface (Spot × IV → PnL) reshaping with theta decay
  - A live cursor (red sphere) tracking the current spot & IV position
  - Regime colour transitions as the market environment changes
  - DTE countdown and live Greeks in the title

Requirements:  pip install matplotlib numpy pandas pillow
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import Normalize
import io

try:
    from PIL import Image
except ImportError:
    print("ERROR: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
PLOT_DIR   = os.path.join(OUTPUT_DIR, "plots")
DOCS_DIR   = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "images")

REGIME_COLOURS = {
    "Low":       "#2ecc71",
    "Medium":    "#3498db",
    "High":      "#e74c3c",
    "Crush":     "#9b59b6",
    "Expansion": "#f39c12",
}


def ensure_dirs():
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(DOCS_DIR, exist_ok=True)


def render_frame_to_image(fig):
    """Render matplotlib figure to PIL Image without saving to disk."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    return img


def animate_strategy(fpath):
    """Generate animated GIF for a single strategy's evolving surface."""
    df = pd.read_csv(fpath)
    if df.empty:
        return None

    name = os.path.basename(fpath).replace("_evolving.csv", "")
    strategy_label = name.split("_", 1)[-1].replace("_", " ")
    frames = sorted(df["Frame"].unique())

    if len(frames) < 2:
        print(f"  Skipping {strategy_label}: only {len(frames)} frame(s)")
        return None

    print(f"  Animating: {strategy_label} ({len(frames)} frames)...")
    images = []

    # Pre-compute global PnL range for consistent colour scale
    global_pnl_min = df["PnL"].quantile(0.02)
    global_pnl_max = df["PnL"].quantile(0.98)
    pnl_norm = Normalize(vmin=global_pnl_min, vmax=global_pnl_max)

    for idx, frame_id in enumerate(frames):
        frame_df = df[df["Frame"] == frame_id]
        if frame_df.empty:
            continue

        live_spot = frame_df["LiveSpot"].iloc[0]
        live_iv   = frame_df["LiveIV"].iloc[0]
        dte_days  = frame_df["DTE_days"].iloc[0]
        regime    = frame_df["Regime"].iloc[0]
        minute    = frame_df["Minute"].iloc[0]

        spots = np.sort(frame_df["Spot"].unique())
        ivs   = np.sort(frame_df["IV"].unique())

        if len(spots) < 3 or len(ivs) < 3:
            continue

        pivot = frame_df.pivot_table(index="IV", columns="Spot", values="PnL",
                                      aggfunc="first")
        X, Y = np.meshgrid(pivot.columns.values, pivot.index.values * 100)
        Z = pivot.values

        # Get live Greeks at closest grid point
        dist = ((frame_df["Spot"] - live_spot)**2 +
                (frame_df["IV"] - live_iv)**2)
        closest = frame_df.loc[dist.idxmin()]

        fig = plt.figure(figsize=(12, 8), facecolor="#0d1117")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0d1117")

        # Surface with RdYlGn colourmap
        surf = ax.plot_surface(X, Y, Z, cmap="RdYlGn", norm=pnl_norm,
                               edgecolor="none", alpha=0.82,
                               rstride=1, cstride=1)

        # Live position cursor (red sphere)
        live_pnl = closest["PnL"]
        ax.scatter([live_spot], [live_iv * 100], [live_pnl],
                   color="#ff4757", s=120, zorder=10, edgecolors="white",
                   linewidths=1.5, depthshade=False)

        # Vertical drop line from cursor to surface base
        ax.plot([live_spot, live_spot], [live_iv * 100, live_iv * 100],
                [Z.min(), live_pnl], color="#ff4757", linewidth=1.0,
                alpha=0.5, linestyle="--")

        # Crosshair lines on the surface
        spot_idx = np.argmin(np.abs(pivot.columns.values - live_spot))
        iv_idx   = np.argmin(np.abs(pivot.index.values - live_iv))
        if spot_idx < Z.shape[1]:
            ax.plot(np.full_like(Y[:, 0], pivot.columns.values[spot_idx]),
                    Y[:, 0], Z[:, spot_idx],
                    color="#ff4757", linewidth=1.5, alpha=0.6)
        if iv_idx < Z.shape[0]:
            ax.plot(X[0, :], np.full_like(X[0, :], pivot.index.values[iv_idx] * 100),
                    Z[iv_idx, :],
                    color="#ff4757", linewidth=1.5, alpha=0.6)

        # Zero-PnL plane (breakeven)
        x_range = [X.min(), X.max()]
        y_range = [Y.min(), Y.max()]
        xx_plane, yy_plane = np.meshgrid(x_range, y_range)
        ax.plot_surface(xx_plane, yy_plane, np.zeros_like(xx_plane),
                        color="white", alpha=0.08)

        # Styling
        regime_colour = REGIME_COLOURS.get(regime, "#95a5a6")
        hours = minute // 60
        mins  = minute % 60

        title = (f"{strategy_label}\n"
                 f"DTE: {dte_days:.1f}d  |  Time: {hours:02d}:{mins:02d}  |  "
                 f"Regime: {regime}  |  "
                 f"S&P: {live_spot:.0f}  |  IV: {live_iv*100:.1f}%\n"
                 f"\u0394={closest['Delta']:.3f}  "
                 f"\u0393={closest['Gamma']:.5f}  "
                 f"\u03BD={closest['Vega']:.3f}  "
                 f"\u0398={closest['Theta']:.3f}  "
                 f"PnL=${live_pnl:.2f}")

        ax.set_title(title, fontsize=10, color="white", pad=15,
                     fontfamily="monospace")
        ax.set_xlabel("S&P 500 Spot", fontsize=9, color="#aaa", labelpad=8)
        ax.set_ylabel("Implied Vol (%)", fontsize=9, color="#aaa", labelpad=8)
        ax.set_zlabel("PnL ($)", fontsize=9, color="#aaa", labelpad=8)

        # Dark theme for axes
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333")
        ax.yaxis.pane.set_edgecolor("#333")
        ax.zaxis.pane.set_edgecolor("#333")
        ax.tick_params(colors="#888", labelsize=7)
        ax.grid(True, alpha=0.15)

        # Regime indicator bar at top
        fig.patches.append(plt.Rectangle(
            (0.02, 0.96), 0.08, 0.025, transform=fig.transFigure,
            facecolor=regime_colour, edgecolor="none", zorder=100))
        fig.text(0.11, 0.968, f" {regime}", fontsize=9, color=regime_colour,
                 transform=fig.transFigure, fontweight="bold", va="center")

        # Frame counter
        fig.text(0.95, 0.02, f"Frame {idx+1}/{len(frames)}",
                 fontsize=8, color="#555", transform=fig.transFigure,
                 ha="right")

        # Rotate view slightly per frame for cinematic effect
        base_azim = 225
        azim_wobble = 15 * np.sin(2 * np.pi * idx / len(frames))
        ax.view_init(elev=28, azim=base_azim + azim_wobble)

        img = render_frame_to_image(fig)
        images.append(img)
        plt.close(fig)

    if not images:
        return None

    # Save animated GIF
    gif_path = os.path.join(PLOT_DIR, f"{name}_animated.gif")
    docs_gif_path = os.path.join(DOCS_DIR, f"{name}_animated.gif")

    # Ensure all frames have the same size (resize to first frame's size)
    base_size = images[0].size
    resized = []
    for img in images:
        if img.size != base_size:
            img = img.resize(base_size, Image.LANCZOS)
        resized.append(img)

    resized[0].save(
        gif_path,
        save_all=True,
        append_images=resized[1:],
        duration=150,       # ms per frame
        loop=0,             # infinite loop
        optimize=True
    )

    # Copy to docs
    resized[0].save(
        docs_gif_path,
        save_all=True,
        append_images=resized[1:],
        duration=150,
        loop=0,
        optimize=True
    )

    file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"    Saved: {gif_path} ({file_size_mb:.1f} MB, {len(resized)} frames)")
    return gif_path


def create_composite_snapshot():
    """Create a static composite showing 4 strategies at different time points."""
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_evolving.csv")))
    if len(files) < 2:
        return

    fig, axes = plt.subplots(2, 3, figsize=(20, 12),
                              subplot_kw={"projection": "3d"},
                              facecolor="#0d1117")

    for idx, fpath in enumerate(files[:6]):
        if idx >= 6:
            break
        ax = axes[idx // 3][idx % 3]
        ax.set_facecolor("#0d1117")

        df = pd.read_csv(fpath)
        name = os.path.basename(fpath).replace("_evolving.csv", "")
        strategy_label = name.split("_", 1)[-1].replace("_", " ")

        # Show 3 time slices: early, mid, late
        frames = sorted(df["Frame"].unique())
        mid_frame = frames[len(frames) // 2]
        frame_df = df[df["Frame"] == mid_frame]

        if frame_df.empty:
            continue

        spots = np.sort(frame_df["Spot"].unique())
        ivs   = np.sort(frame_df["IV"].unique())
        pivot = frame_df.pivot_table(index="IV", columns="Spot", values="PnL",
                                      aggfunc="first")
        X, Y = np.meshgrid(pivot.columns.values, pivot.index.values * 100)
        Z = pivot.values

        live_spot = frame_df["LiveSpot"].iloc[0]
        live_iv = frame_df["LiveIV"].iloc[0]
        dte = frame_df["DTE_days"].iloc[0]
        regime = frame_df["Regime"].iloc[0]

        ax.plot_surface(X, Y, Z, cmap="RdYlGn", edgecolor="none", alpha=0.8)

        dist = ((frame_df["Spot"] - live_spot)**2 +
                (frame_df["IV"] - live_iv)**2)
        closest = frame_df.loc[dist.idxmin()]
        ax.scatter([live_spot], [live_iv * 100], [closest["PnL"]],
                   color="#ff4757", s=80, edgecolors="white", linewidths=1,
                   depthshade=False, zorder=10)

        ax.set_title(f"{strategy_label}\nDTE: {dte:.0f}d | {regime}",
                     fontsize=9, color="white", pad=5)
        ax.set_xlabel("Spot", fontsize=7, color="#888", labelpad=4)
        ax.set_ylabel("IV%", fontsize=7, color="#888", labelpad=4)
        ax.set_zlabel("PnL", fontsize=7, color="#888", labelpad=4)
        ax.tick_params(colors="#666", labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#333")
        ax.yaxis.pane.set_edgecolor("#333")
        ax.zaxis.pane.set_edgecolor("#333")
        ax.view_init(elev=25, azim=225)

    fig.suptitle("Live PnL Surfaces — Mid-Trade Snapshot (All Strategies)",
                 fontsize=14, color="white", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out = os.path.join(DOCS_DIR, "evolving_surfaces_composite.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved composite: {out}")


# ==========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  Animated 3D PnL Surface Generator")
    print("  Time-Evolving Volatility Regime Surfaces")
    print("=" * 60)
    ensure_dirs()

    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*_evolving.csv")))
    if not files:
        print("\nNo *_evolving.csv files found. Run ./build/backtest first.")
        sys.exit(1)

    print(f"\nFound {len(files)} strategy surface files.\n")

    gif_paths = []
    for fpath in files:
        result = animate_strategy(fpath)
        if result:
            gif_paths.append(result)

    print("\nGenerating composite snapshot...")
    create_composite_snapshot()

    print(f"\nDone! Generated {len(gif_paths)} animated GIFs.")
    print(f"Output: {PLOT_DIR}/")
    print(f"Docs:   {DOCS_DIR}/")
