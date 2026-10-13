"""Render the GitHub cover entirely from the analytic synthetic AFM surface.

Usage: python examples/render_github_preview.py
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, Circle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.synthetic_afm import synthetic_surface
from afm_analysis import fit_plane


def render(destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#e9f3ff"})
    fig = plt.figure(figsize=(16, 9), dpi=160, facecolor="#080d1a")
    backdrop = fig.add_axes([0, 0, 1, 1], zorder=0)
    backdrop.set(xlim=(0, 1), ylim=(0, 1))
    backdrop.axis("off")
    rng = np.random.default_rng(202604)
    stars = rng.random((100, 2))
    backdrop.scatter(stars[:, 0], stars[:, 1], s=rng.uniform(0.4, 2.5, 100),
                     c="#7b9ac0", alpha=0.20)
    for radius in (0.18, 0.24, 0.32, 0.42):
        backdrop.add_patch(Circle((0.74, 0.61), radius, fill=False,
                                 edgecolor="#17304c", linewidth=0.7, alpha=0.65))
    fig.text(0.052, 0.924, "AFRUZ  /  SCIENTIFIC SOFTWARE", color="#69ddd5",
             fontsize=11, weight="bold")
    fig.text(0.052, 0.823, "AFM Studio", fontsize=45, weight="bold")
    fig.text(0.055, 0.758, "Explore the surface.", fontsize=22, color="#adc4e4")
    fig.text(0.055, 0.685, "Calibrated topography, roughness,\nparticle metrology and reproducible analysis.",
             fontsize=12, color="#9bb0cd", linespacing=1.6)
    backdrop.add_patch(FancyBboxPatch((0.052, 0.597), 0.238, 0.047,
        boxstyle="round,pad=0.007,rounding_size=0.014", facecolor="#132638", edgecolor="#24595d"))
    fig.text(0.065, 0.611, "v0.4.0   /   ORBIT WORKSPACE", fontsize=10, color="#83eee0", weight="bold")

    cmap = LinearSegmentedColormap.from_list("orbit_height", [
        "#142b52", "#375ab5", "#8661db", "#dc70cb", "#fbbaae", "#fff0ce"])
    raw = synthetic_surface(192)
    z, _ = fit_plane(raw, robust=True)
    z *= 1e9
    span_um = 4.0
    axis = (np.arange(z.shape[0]) + 0.5) * span_um / z.shape[0]
    x, y = np.meshgrid(axis, axis)
    norm = Normalize(vmin=z.min(), vmax=z.max())
    surface = fig.add_axes([0.385, 0.25, 0.60, 0.68], projection="3d", zorder=2)
    surface.set_facecolor("none")
    surface.plot_surface(x, y, z, cmap=cmap, norm=norm, rcount=140, ccount=140,
                          linewidth=0, antialiased=True, shade=True)
    floor = z.min() - 8
    surface.contour(x, y, z, levels=np.linspace(4, 50, 8), zdir="z", offset=floor,
                     cmap=cmap, alpha=0.75, linewidths=0.7)
    surface.set_zlim(floor, 65)
    surface.set(xlim=(0, span_um), ylim=(0, span_um))
    surface.view_init(elev=36, azim=-55)
    surface.set_box_aspect((1, 1, 0.46))
    surface.set_axis_off()
    fig.text(0.725, 0.293, "SYNTHETIC TOPOGRAPHY", fontsize=10, color="#9eb3d6", ha="center")
    fig.text(0.725, 0.263, "4 × 4 µm   ·   analytic Gaussian particles", fontsize=10,
             color="#7e93b3", ha="center")

    backdrop.add_patch(FancyBboxPatch((0.052, 0.17), 0.31, 0.356,
        boxstyle="round,pad=0.008,rounding_size=0.013", facecolor="#101b2d", edgecolor="#25354e"))
    fig.text(0.068, 0.493, "SURFACE HEIGHT", fontsize=10, color="#a9bfdc", weight="bold")
    map_ax = fig.add_axes([0.068, 0.204, 0.142, 0.255])
    map_ax.imshow(z, origin="lower", extent=(0, 4, 0, 4), cmap=cmap, norm=norm)
    map_ax.axis("off")
    map_ax.plot([0.22, 1.22], [0.28, 0.28], color="white", lw=2)
    map_ax.text(0.72, 0.43, "1 µm", color="white", ha="center", fontsize=8)
    cax = fig.add_axes([0.218, 0.242, 0.007, 0.16])
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    bar.outline.set_visible(False)
    bar.set_ticks([0, 25, 50])
    bar.ax.tick_params(labelsize=8, colors="#a9bfdc", length=0)
    bar.ax.set_title("nm", fontsize=8, color="#a9bfdc", pad=8)
    fig.text(0.267, 0.411, "01", fontsize=18, weight="bold", color="#75e3df")
    fig.text(0.267, 0.381, "IMPORT", fontsize=9, color="#d1ddf0")
    fig.text(0.267, 0.325, "02", fontsize=18, weight="bold", color="#a18cf6")
    fig.text(0.267, 0.295, "ANALYZE", fontsize=9, color="#d1ddf0")
    fig.text(0.267, 0.239, "03", fontsize=18, weight="bold", color="#e8a4d3")
    fig.text(0.267, 0.209, "EXPORT", fontsize=9, color="#d1ddf0")

    backdrop.plot([0.052, 0.948], [0.123, 0.123], color="#25354e", lw=0.8)
    fig.text(0.052, 0.073, "NID IMPORT   /   ROBUST LEVELING   /   PHYSICAL PSD   /   BATCH EXPORTS",
             fontsize=9, color="#9eb3d6")
    fig.text(0.948, 0.073, "SIMULATED DATA · NO EXPERIMENTAL SCAN", ha="right",
             fontsize=8, color="#69ddd5")
    fig.savefig(destination, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return destination


if __name__ == "__main__":
    target = Path(__file__).resolve().parents[1] / "docs" / "screenshots" / "github-cover.png"
    print(render(target))
