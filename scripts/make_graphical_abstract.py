#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Graphical abstract — figures/graphical_abstract_N.png (auto-increment).
MDPI Graphical Abstract: python make_graphical_abstract.py --mdpi-out PATH  → 1100×560 px @ 300 dpi.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures"

# 天平图：优先 figures 下多种文件名/格式（WebP 需 Pillow 带 libwebp，否则请改用 PNG）
_BALANCE_CANDIDATES = [
    OUT_DIR / "天平.png",
    OUT_DIR / "天平.PNG",
    OUT_DIR / "天平.webp",
    OUT_DIR / "天平.jpg",
    OUT_DIR / "balance.png",
    OUT_DIR / "balance.webp",
    OUT_DIR / "tianping.webp",
    OUT_DIR / "tianping.png",
    ROOT / "天平.webp",
    ROOT / "天平.png",
]


def _load_balance_rgba(path: Path):
    """Return (H,W,4) uint8 RGBA or None."""
    path = Path(path)
    if not path.is_file():
        return None
    # 1) Pillow
    try:
        from PIL import Image

        im = Image.open(path)
        im = im.convert("RGBA")
        return np.asarray(im)
    except Exception:
        pass
    # 2) imageio（常能读 WebP）
    try:
        import imageio.v3 as iio

        arr = np.asarray(iio.imread(path))
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
        elif arr.shape[-1] == 3:
            h, w = arr.shape[:2]
            a = np.full((h, w, 1), 255, dtype=arr.dtype)
            arr = np.concatenate([arr, a], axis=-1)
        return arr
    except Exception:
        pass
    # 3) matplotlib
    try:
        import matplotlib.image as mpimg

        arr = mpimg.imread(str(path))
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.shape[-1] == 3:
            h, w = arr.shape[:2]
            a = np.full((h, w, 1), 255, dtype=arr.dtype)
            arr = np.concatenate([arr, a], axis=-1)
        return arr
    except Exception:
        pass
    return None

target_width_px = 1100
target_height_px = 560
if len(sys.argv) >= 3 and sys.argv[1] == "--mdpi-out":
    out_path = Path(sys.argv[2]).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dpi = 300
    fig_width = target_width_px / dpi
    fig_height = target_height_px / dpi
else:
    n = 1
    while (OUT_DIR / f"graphical_abstract_{n}.png").exists():
        n += 1
    out_path = OUT_DIR / f"graphical_abstract_{n}.png"
    dpi = 100
    fig_width = target_width_px / dpi
    fig_height = target_height_px / dpi

np.random.seed(42)

# Five-step: enlarge boxes by +80% (×1.8); gap between boxes ×(1+1.5)=2.5
box_width_base = 0.26 * (1.0 - 0.6)
box_height_base = 0.032
gap_base = 0.008
box_width = box_width_base * 1.8
box_height = box_height_base * 1.8
gap = gap_base * 2.5
box_left = 0.5 - box_width / 2

title_cx_left = 0.2
title_cx_right = 0.8
method_y = 0.108
chk_fs = 28
green_edge = "#27AE60"
# 与 Simple/Complex 同字号与线宽；无填充，仅绿色描边
method_bbox = dict(boxstyle="round,pad=0.28", facecolor="none", edgecolor=green_edge, linewidth=2)

fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi, facecolor="white")
ax = fig.add_subplot(111)
ax.axis("off")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

orange = "#E67E22"
blue_pt = "#1F618D"

# --- Left ---
ax.fill_betweenx([0.22, 0.78], 0, 0.4, color="#D6EAF8", alpha=0.35, zorder=0)
left_x = np.linspace(0.05, 0.35, 200)
y_simple = 0.68 - 0.42 * (left_x - 0.05) / 0.3 + 0.018 * np.random.randn(200)
y_simple = np.clip(y_simple, 0.30, 0.72)
ax.plot(left_x, y_simple, color=blue_pt, linewidth=3.0, solid_capstyle="round", antialiased=True, zorder=3)
ax.scatter([0.1, 0.18, 0.28], [0.62, 0.48, 0.34], color=orange, s=62, zorder=5, edgecolors="white", linewidths=1.2)

ax.text(
    title_cx_left,
    0.92,
    "Simple Data",
    fontsize=14,
    ha="center",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#1F618D", linewidth=2),
    zorder=6,
)
ax.text(title_cx_left, 0.865, "(e.g., NASA Battery)", fontsize=9, ha="center", style="italic", color="#2C3E50", zorder=6)

# --- Right ---
ax.fill_betweenx([0.22, 0.78], 0.6, 1.0, color="#FEF9E7", alpha=0.4, zorder=0)
right_x = np.linspace(0.65, 0.95, 300)
y1 = 0.52 + 0.18 * np.sin(25 * right_x) + 0.025 * np.random.randn(300)
y2 = 0.44 + 0.18 * np.sin(35 * right_x + 1) + 0.018 * np.random.randn(300)
y3 = 0.58 + 0.14 * np.sin(45 * right_x + 2) + 0.028 * np.random.randn(300)
y1, y2, y3 = np.clip(y1, 0.28, 0.74), np.clip(y2, 0.28, 0.74), np.clip(y3, 0.28, 0.74)
ax.plot(right_x, y1, color=orange, linewidth=2.5, antialiased=True, zorder=3)
ax.plot(right_x, y2, color=orange, linewidth=2.5, antialiased=True, zorder=3)
ax.plot(right_x, y3, color=orange, linewidth=2.5, antialiased=True, zorder=3)
ax.scatter(
    [0.70, 0.80, 0.90],
    [
        y1[int(np.argmin(np.abs(right_x - 0.70)))],
        y2[int(np.argmin(np.abs(right_x - 0.80)))],
        y3[int(np.argmin(np.abs(right_x - 0.90)))],
    ],
    color=blue_pt,
    s=58,
    zorder=5,
    edgecolors="white",
    linewidths=1.2,
)

ax.text(
    title_cx_right,
    0.92,
    "Complex Data",
    fontsize=14,
    ha="center",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=orange, linewidth=2),
    zorder=6,
)
ax.text(title_cx_right, 0.865, "(e.g., XJTU Bearing)", fontsize=9, ha="center", style="italic", color="#2C3E50", zorder=6)

# --- Transition: top arrow L→R, bottom arrow R→L; slogan in between ---
y_arrow_hi, y_arrow_lo = 0.818, 0.668
ax.annotate(
    "",
    xy=(0.58, y_arrow_hi),
    xytext=(0.42, y_arrow_hi),
    arrowprops=dict(arrowstyle="->", color="#5D6D7E", lw=2.8, shrinkA=0, shrinkB=0),
    zorder=4,
)
ax.annotate(
    "",
    xy=(0.42, y_arrow_lo),
    xytext=(0.58, y_arrow_lo),
    arrowprops=dict(arrowstyle="->", color="#5D6D7E", lw=2.8, shrinkA=0, shrinkB=0),
    zorder=4,
)
ax.text(
    0.5,
    (y_arrow_hi + y_arrow_lo) / 2,
    "Data Characteristics\nDrive Method Choice",
    fontsize=10.5,
    ha="center",
    va="center",
    weight="bold",
    bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor="#7F8C8D", linewidth=1.5),
    zorder=6,
)

# --- Method selection stack ---
steps = [
    "1. Channels",
    r"2. Nonlinearity ($R^2$)",
    "3. Sample size",
    "4. Missing pattern",
    "5. Downstream",
]
title_y = 0.638
step_start = 0.578
step_fs = max(9.0, min(10.8, 9.2 * (box_height / 0.032)))

ax.text(0.5, title_y, "Method Selection", fontsize=10, ha="center", weight="bold", color="#2C3E50", zorder=6)

for i, label in enumerate(steps):
    y_top = step_start - i * (box_height + gap)
    y_bot = y_top - box_height
    ax.add_patch(
        Rectangle(
            (box_left, y_bot),
            box_width,
            box_height,
            facecolor="#F8F9F9",
            edgecolor="#5D6D7E",
            linewidth=1.05,
            alpha=0.95,
            zorder=4,
        )
    )
    ax.text(
        0.5,
        (y_top + y_bot) / 2,
        label,
        fontsize=step_fs,
        va="center",
        ha="center",
        weight="bold",
        color="#1C2833",
        zorder=5,
    )

for i in range(len(steps) - 1):
    y_bot_i = step_start - i * (box_height + gap) - box_height
    y_top_next = step_start - (i + 1) * (box_height + gap)
    ax.annotate(
        "",
        xy=(0.5, y_top_next + 0.002),
        xytext=(0.5, y_bot_i - 0.002),
        arrowprops=dict(arrowstyle="-|>", color=orange, lw=2.0, mutation_scale=11, shrinkA=0, shrinkB=0),
        zorder=5,
    )

# --- 天平图：imshow+extent 固定显示区域（避免 OffsetImage 缩放过小看不见）---
_balance_arr = None
_balance_used = None
for _p in _BALANCE_CANDIDATES:
    _balance_arr = _load_balance_rgba(_p)
    if _balance_arr is not None:
        _balance_used = _p
        break

_q_y = 0.168
if _balance_arr is not None:
    xc, yc, w, h = 0.5, 0.118, 0.15, 0.072
    ih, iw = _balance_arr.shape[0], _balance_arr.shape[1]
    aspect_img = iw / max(ih, 1)
    aspect_box = w / h
    if aspect_img > aspect_box:
        h = w / aspect_img
    else:
        w = h * aspect_img
    x0, x1 = xc - w / 2, xc + w / 2
    y0, y1 = yc - h / 2, yc + h / 2
    ax.imshow(
        _balance_arr,
        extent=[x0, x1, y0, y1],
        zorder=5,
        aspect="auto",
        interpolation="antialiased",
    )
    _q_y = y1 + 0.038
    print("Balance image loaded:", _balance_used)
else:
    print(
        "未加载天平图。请将图片放到以下任一路径并重跑：\n"
        + "\n".join(f"  - {p}" for p in _BALANCE_CANDIDATES[:4])
        + "\n若仅有 .webp 且报错，请将图另存为 PNG（figures/天平.png），或: pip install imageio pillow"
    )

ax.text(0.5, _q_y, "?", fontsize=24, ha="center", va="center", color=orange, weight="bold", zorder=7)

# --- LI/KNN & Diffusion: same style as Simple/Complex (14pt, white fill, green border) ---
t_li = ax.text(
    title_cx_left,
    method_y,
    "LI / KNN",
    fontsize=14,
    ha="center",
    va="center",
    weight="bold",
    bbox=method_bbox,
    zorder=6,
)
t_df = ax.text(
    title_cx_right,
    method_y,
    "Diffusion Model",
    fontsize=14,
    ha="center",
    va="center",
    weight="bold",
    bbox=method_bbox,
    zorder=6,
)
fig.canvas.draw()
r_li = t_li.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
r_df = t_df.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transData.inverted())
ax.text(
    r_li.x1 + 0.028,
    method_y,
    "✓",
    fontsize=chk_fs,
    color="#1E8449",
    weight="bold",
    ha="center",
    va="center",
    zorder=7,
)
ax.text(
    r_df.x1 + 0.028,
    method_y,
    "✓",
    fontsize=chk_fs,
    color="#1E8449",
    weight="bold",
    ha="center",
    va="center",
    zorder=7,
)

# --- Bottom ---
ax.axhline(y=0.075, xmin=0.02, xmax=0.98, color="#BDC3C7", linewidth=1.0, linestyle="--", zorder=2)
ax.text(
    0.5,
    0.038,
    "On simple data, LI/KNN suffice; on complex data, diffusion excels.",
    fontsize=10.5,
    ha="center",
    va="center",
    style="italic",
    color="#2C3E50",
    bbox=dict(boxstyle="round,pad=0.35", facecolor="#F2F4F4", edgecolor="none", alpha=0.95),
    zorder=6,
)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(out_path, dpi=dpi, facecolor="white", pad_inches=0.02, format="png")
plt.close(fig)

if len(sys.argv) >= 3 and sys.argv[1] == "--mdpi-out":
    print("Saved MDPI Graphical Abstract (1100x560 @300dpi):", out_path)
else:
    print("Saved:", out_path, f"(#{n})")
