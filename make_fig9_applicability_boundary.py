import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "figure.dpi": 200,
        }
    )

    x = np.linspace(0, 1, 500)

    # Blue: LI/KNN (high -> low), Red: Diffusion (low -> high)
    # Smooth monotone curves with a crossover near the middle.
    blue = 0.92 - 0.75 * (x**1.2)
    red = 0.20 + 0.75 * (x**1.1)

    blue = np.clip(blue, 0, 1)
    red = np.clip(red, 0, 1)

    idx = int(np.argmin(np.abs(blue - red)))
    x_cross = float(x[idx])

    band_half_width = 0.06
    x0 = max(0.0, x_cross - band_half_width)
    x1 = min(1.0, x_cross + band_half_width)

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    ax.axvspan(x0, x1, color="#B0B0B0", alpha=0.25, label="Crossover region")
    ax.plot(x, blue, color="#1f77b4", lw=2.6, label="Traditional interpolation (LI, KNN)")
    ax.plot(x, red, color="#d62728", lw=2.6, label="Diffusion-based models")

    points = [
        (0.12, "NASA IGBT"),
        (0.25, "NASA Battery"),
        (0.86, "XJTU Bearing"),
    ]

    for xp, name in points:
        yb = float(np.interp(xp, x, blue))
        yr = float(np.interp(xp, x, red))
        if yb >= yr:
            ax.scatter([xp], [yb], s=38, color="#1f77b4", zorder=5)
            ax.annotate(
                f"{name}\n(LI/KNN better)",
                (xp, yb),
                xytext=(10, 12),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
            )
        else:
            ax.scatter([xp], [yr], s=38, color="#d62728", zorder=5)
            ax.annotate(
                f"{name}\n(Diffusion better)",
                (xp, yr),
                xytext=(10, 12),
                textcoords="offset points",
                ha="left",
                va="bottom",
                fontsize=10,
            )

    ax.annotate(
        "Medium complexity\n(to study)",
        ((x0 + x1) / 2, 0.08),
        xytext=(0, -8),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Data complexity (low → high)")
    ax.set_ylabel("Method suitability / performance (higher is better)")

    ax.text(
        0.02,
        0.5,
        "Simple data\nLI/KNN sufficient",
        ha="left",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.9),
    )
    ax.text(
        0.98,
        0.5,
        "Complex data\nDiffusion advantage",
        ha="right",
        va="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC", alpha=0.9),
    )

    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.35)
    ax.legend(frameon=True, framealpha=0.95, loc="lower center", ncol=1)

    fig.tight_layout()

    out_dir = Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fig9_applicability_boundary.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    print(str(out_path))


if __name__ == "__main__":
    main()

