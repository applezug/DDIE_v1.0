#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate all paper figures for DDI-E / method-selection draft.
Style aligned with Degradation_Data_Imputation_v1.1 (generate_paper_figs_academic.py).
Output: figures/ in project root.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle

# Project root (parent of scripts/)
ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
# DDIE03: NASA Battery + NASA IGBT (paper experiments)
RESULTS_DIR_DDIE03 = ROOT / "results_DDIE03"
USE_DDIE03 = True  # Paper uses DDIE03 group only

# ---- Style (match v1.1 academic) ----
COLOR_MAIN_BOX = "#4c78a8"
FILL_MAIN_BOX = "#e6f2ff"
COLOR_UNCERTAIN = "#f58518"
FILL_UNCERTAIN = "#fff2e6"
COLOR_GT = "#1f77b4"
COLOR_MISSING = "#7f7f7f"
COLOR_IMPUTED = "#d62728"
FILL_CI = "#ff9896"
PALETTE_LI = "#ff7f0e"
PALETTE_KNN = "#2ca02c"
PALETTE_DDIE = "#d62728"
PALETTE_EXTRA = "#9467bd"


def setup_style():
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Arial Unicode MS"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["font.size"] = 10
    matplotlib.rcParams["axes.linewidth"] = 1.0
    matplotlib.rcParams["axes.labelsize"] = 10
    matplotlib.rcParams["axes.titlesize"] = 11
    matplotlib.rcParams["legend.fontsize"] = 9


def set_subtitle(ax, text):
    ax.set_title(text, y=-0.28, fontsize=11)


def format_axes_line(ax, xlabel="Time step", ylabel="Value"):
    ax.grid(True, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def parse_metric(s: str) -> tuple[float, float]:
    if not s or s == "-":
        return np.nan, np.nan
    s = s.strip()
    parts = re.split(r"\s*[±]\s*", s, maxsplit=1)
    mean = float(parts[0].strip())
    std = float(parts[1].strip()) if len(parts) > 1 else 0.0
    return mean, std


def load_imputation_results(dataset: str) -> dict:
    path = RESULTS_DIR / dataset / "imputation_results.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for mr_str, methods in raw.items():
        if mr_str.startswith("_"):
            continue
        mr = float(mr_str)
        out[mr] = {}
        for method, metrics in methods.items():
            out[mr][method] = {
                k: parse_metric(v) if isinstance(v, str) else (v, 0.0)
                for k, v in metrics.items()
            }
    return out


def load_imputation_results_ddie03(dataset: str, filename: str | None = None) -> dict:
    """Load from results_DDIE03 (NASA Battery / NASA IGBT)."""
    name = filename or "imputation_results_DDIE03.json"
    path = RESULTS_DIR_DDIE03 / dataset / name
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for mr_str, methods in raw.items():
        if mr_str.startswith("_"):
            continue
        mr = float(mr_str)
        out[mr] = {}
        for method, metrics in methods.items():
            out[mr][method] = {
                k: parse_metric(v) if isinstance(v, str) else (v, 0.0)
                for k, v in metrics.items()
            }
    return out


def load_loocv_aggregate_ddie03() -> dict:
    """Load DDIE03 LOOCV aggregate (10%, 50%, 90% etc.) for NASA Battery."""
    path = RESULTS_DIR_DDIE03 / "loocv_nasa_battery" / "imputation_results_loocv_DDIE03.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    agg = raw.get("aggregate", {})
    out = {}
    for mr_str, methods in agg.items():
        if mr_str.startswith("_"):
            continue
        mr = float(mr_str)
        out[mr] = {}
        for method, metrics in methods.items():
            out[mr][method] = {
                k: parse_metric(v) if isinstance(v, str) else (v, 0.0)
                for k, v in metrics.items()
            }
    return out


# =============================================================================
# Figure 1 统一编号 — 与 docs/Figure1_工作记录与代码对照.md 一致，改图时请按 ID 定位
#
#   FIG1-01  主框架（分块 + 整图坐标二次叠画）     fig1_framework()
#            → figures/fig1_ddie_framework.png
#
#   FIG1-02  分块拼接 + 连线（无叠画）            fig1_framework_redesign_blocks()
#            CLI: fig1_redesign_blocks
#            → figures/fig1_FIG1-02_blocks_NNN.png（序号递增，不覆盖）
#
#   FIG1-03  三栏规范示意（笔记 redesign 1.md）    fig1_framework_redesign_threecolumn()
#            CLI: fig1_redesign（默认）
#            → figures/fig1_FIG1-03_threecolumn_NNN.png
#
#   FIG1-04  定稿/投稿用框架（手工或外部矢量导出） 固定文件名
#            figures/ 或 assets/ 下 fig1_ddie_framework_redesign.png
#            MDPI 批次示例：journal_MDPI_v001/10_fig1_framework_redesign_MDPI_v001.png
#            （批次内序号 NN 随导出顺序变化，以 manifest 为准）
#
#   FIG1-00  单分块调试图                        fig1_framework_block_preview()
#            → figures/fig1_block_B*.png
# =============================================================================


def _fig1_next_serial(stem: str) -> Path:
    """下一序号文件，如 stem='fig1_FIG1-02_blocks' → fig1_FIG1-02_blocks_001.png。"""
    n = 1
    while True:
        p = FIG_DIR / f"{stem}_{n:03d}.png"
        if not p.exists():
            return p
        n += 1


# ---------- FIG1-00 / FIG1-02 共用：分块布局 (x0,y0,w,h 为整图 0–1 坐标) ----------
def _to_fig(bbox, x, y):
    """Map local (x,y) in [0,1] to figure coordinates."""
    x0, y0, w, h = bbox
    return (x0 + x * w, y0 + y * h)

FIG1_LAYOUT = {
    "B0": (0, 0.85, 1, 0.14),
    "B1": (0.02, 0.58, 0.14, 0.20),
    "B2": (0.02, 0.46, 0.14, 0.06),
    "B3": (0.04, 0.26, 0.12, 0.20),
    "B4": (0.20, 0.66, 0.22, 0.20),
    "B5": (0.20, 0.18, 0.24, 0.48),
    "B6": (0.46, 0.66, 0.22, 0.20),
    "B8": (0.72, 0.48, 0.12, 0.22),
    "B9": (0.72, 0.32, 0.12, 0.12),
    "B10": (0.72, 0.14, 0.12, 0.14),
    "B11": (0.68, 0.02, 0.30, 0.10),
    "B12": (0.25, 0, 0.5, 0.04),
}

def _fig1_block_B0(ax, bbox):
    """顶部总标题与阶段划分。"""
    def f(x, y): return _to_fig(bbox, x, y)
    ax.text(*f(0.5, 0.85), "DDI-E: Conditional diffusion for electrical time series imputation", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(*f(0.35, 0.95), "Training", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(*f(0.72, 0.95), "Inference / Sampling", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.plot([f(0.42, 0)[0], f(0.42, 1)[0]], [f(0.42, 0)[1], f(0.42, 1)[1]], color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(*f(0.09, 0.36), "Input &\ncondition", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(*f(0.32, 0.36), "Core model &\ndiffusion", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(*f(0.78, 0.36), "Output &\napplication", ha="center", va="center", fontsize=9, fontweight="bold")

def _fig1_block_B1(ax, bbox):
    """观测序列 (Observed sequence)：框 + 波形 + X_obs + 标题（与框有间距）。"""
    def f(x, y): return _to_fig(bbox, x, y)
    # 布局：顶部留白给标题，底部留白给 X_obs，中间为框
    title_y = 0.94   # 标题在此，与框顶留 gap
    box_ytop = 0.86  # 框顶
    box_ybot = 0.18  # 框底
    label_y = 0.06   # X_obs 在框下方，与框底留 gap
    box_xleft, box_xright = 0.06, 0.94

    # 圆角矩形框（#f5f5f5, 边框 #888）
    box_w = f(box_xright, 0)[0] - f(box_xleft, 0)[0]
    box_h = f(0, box_ytop)[1] - f(0, box_ybot)[1]
    box = FancyBboxPatch(f(box_xleft, box_ybot), box_w, box_h, boxstyle="round,pad=0.015", ec="#888", fc="#f5f5f5")
    ax.add_patch(box)

    # 框内示意波形（正弦+谐波，归一化到框内高度）
    t = np.linspace(0, 1, 50)
    wave = 0.5 + 0.35 * np.sin(2 * np.pi * t) + 0.1 * np.sin(6 * np.pi * t)
    wnorm = (wave - wave.min()) / (wave.max() - wave.min() + 1e-8)
    # 波形 x: 框内 10%~90%，y: 框内 15%~85%
    wx = box_xleft + 0.10 * (box_xright - box_xleft) + 0.80 * (box_xright - box_xleft) * t
    wy = box_ybot + 0.15 * (box_ytop - box_ybot) + 0.70 * (box_ytop - box_ybot) * wnorm
    ax.plot([f(x, y)[0] for x, y in zip(wx, wy)], [f(x, y)[1] for x, y in zip(wx, wy)], color=COLOR_GT, linewidth=1.4)

    # 框下方符号
    ax.text(*f(0.5, label_y), r"$X_{obs}$", ha="center", va="top", fontsize=10)
    # 框上方标题（与框顶保持间距）
    ax.text(*f(0.5, title_y), "Observed sequence", ha="center", va="bottom", fontsize=9, style="italic")

def _fig1_block_B2(ax, bbox):
    """缺失掩码 M：横条 + 分段着色 + 标签。"""
    def f(x, y): return _to_fig(bbox, x, y)
    ax.add_patch(Rectangle(f(0, 0.15), f(1, 0)[0] - f(0, 0.15)[0], f(0, 0.85)[1] - f(0, 0.15)[1], fc="#e8e8e8", ec="#888"))
    for i in range(8):
        x0 = 0.05 + i * 0.11
        if i % 3 != 0:
            ax.add_patch(Rectangle(f(x0, 0.2), f(0.09, 0)[0] - f(x0, 0.2)[0], f(0, 0.7)[1] - f(0, 0.2)[1], fc=COLOR_MAIN_BOX, ec="none", alpha=0.7))
    ax.text(*f(0.5, 0.02), "Missing mask M", ha="center", va="top", fontsize=8)

def _fig1_block_B3(ax, bbox):
    """条件构造：参考 B5，两路输入 3D 薄条 + ⊕ concat + 块到块箭头 → Condition construction。"""
    def f(x, y): return _to_fig(bbox, x, y)
    xm, xw = 0.06, 0.88
    depth = 0.018

    # 外框（主色）
    box = FancyBboxPatch(f(0, 0), f(1, 1)[0] - f(0, 0)[0], f(1, 1)[1] - f(0, 0)[1], boxstyle="round,pad=0.03", ec=COLOR_MAIN_BOX, fc=FILL_MAIN_BOX)
    ax.add_patch(box)

    # ----- 上排：X_obs 与 Mask，中间 ⊕，虚线框包住；concat 紧挨箭头左侧、与箭头平行 -----
    inp_h = 0.22
    inp_y_top = 0.78
    inp_y_bot = inp_y_top - inp_h
    w1, w2 = 0.36 * xw, 0.36 * xw
    x2 = xm + w1 + 0.22 * xw
    # 左：X_obs
    _draw_3d_slab(ax, f, xm, inp_y_bot, w1, inp_h, depth=depth, fc_top="white", fc_side="#a8a8a8", fc_front="#b8b8b8")
    ax.text(*f(xm + w1 / 2, inp_y_bot + inp_h / 2), r"$X_{obs}$", ha="center", va="center", fontsize=9)
    # 中间：⊕（调大）
    cx, cy = xm + w1 + 0.11 * xw, inp_y_bot + inp_h / 2
    ax.text(*f(cx, cy), r"$\oplus$", ha="center", va="center", fontsize=18, fontweight="bold")
    # 右：Mask
    _draw_3d_slab(ax, f, x2, inp_y_bot, w2, inp_h, depth=depth, fc_top="#f0f0f0", fc_side="#a0a0a0", fc_front="#b0b0b0")
    ax.text(*f(x2 + w2 / 2, inp_y_bot + inp_h / 2), "Mask", ha="center", va="center", fontsize=9)
    # 虚线框包住 X_obs 与 Mask 两模块
    pad = 0.02
    dash_x0 = xm - pad
    dash_y0 = inp_y_bot - depth - pad
    dash_w = (x2 + w2 + depth) - xm + 2 * pad
    dash_h = inp_y_top - (inp_y_bot - depth) + 2 * pad
    rect = FancyBboxPatch(f(dash_x0, dash_y0), f(dash_x0 + dash_w, dash_y0)[0] - f(dash_x0, dash_y0)[0], f(dash_x0, dash_y0 + dash_h)[1] - f(dash_x0, dash_y0)[1], boxstyle="round,pad=0.005", ec="#666", fc="none", linestyle="--", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(*f(0.5, inp_y_top + 0.08), "Condition construction", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # ----- 下块：concat 的结果 = 2 通道条件（供去噪网络用） -----
    cond_h = 0.36
    cond_y_bot = 0.08
    cond_top_y = cond_y_bot + cond_h
    _draw_3d_slab(ax, f, xm, cond_y_bot, xw, cond_h, depth=depth, fc_top="white", fc_side="#a8a8a8", fc_front="#b8b8b8")
    # 标题：即“条件构造”的产物
    ax.text(*f(0.5, cond_y_bot + cond_h / 2 + 0.06), "Condition (2 ch)", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(*f(0.5, cond_y_bot + cond_h / 2 - 0.02), "output of concat", ha="center", va="center", fontsize=7, style="italic")
    ax.text(*f(0.5, cond_y_bot + cond_h / 2 - 0.10), "to denoising network", ha="center", va="center", fontsize=7)
    # 箭头：上排底面中心 → 下块顶面
    ax.annotate("", xy=(f(0.5, cond_top_y)[0], f(0.5, cond_top_y)[1]), xytext=(f(0.5, inp_y_bot - depth)[0], f(0.5, inp_y_bot - depth)[1]), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    # concat：与箭头平行，紧挨箭头左侧（稍有空隙）
    arrow_mid_y = (inp_y_bot - depth + cond_top_y) / 2
    concat_x = 0.5 - 0.035
    ax.text(*f(concat_x, arrow_mid_y), "concat", ha="center", va="center", fontsize=8, rotation=90)

def _fig1_block_B4(ax, bbox):
    """前向扩散：X_0 -> X_t -> X_t -> X_T 四格 + 箭头 + Add noise。"""
    def f(x, y): return _to_fig(bbox, x, y)
    rs = np.random.RandomState(42)
    for i, (lx, label) in enumerate([(0.02, r"$X_0$"), (0.26, r"$X_t$"), (0.50, r"$X_t$"), (0.74, r"$X_T$")]):
        bw = 0.22
        bx = FancyBboxPatch(f(lx, 0.12), f(lx + bw, 0.12)[0] - f(lx, 0.12)[0], f(lx, 0.88)[1] - f(lx, 0.12)[1], boxstyle="round,pad=0.01", ec="#666", fc="white")
        ax.add_patch(bx)
        if i == 0:
            ti = np.linspace(0, 1, 20)
            ax.plot([f(lx + 0.05 + 0.12 * tii, 0.2 + 0.55 * np.sin(3 * tii))[0] for tii in ti], [f(lx + 0.05, 0.2 + 0.55 * np.sin(3 * tii))[1] for tii in ti], color=COLOR_GT, lw=1)
        elif i == 3:
            sx = lx + 0.08 + 0.12 * rs.rand(15)
            sy = 0.25 + 0.5 * rs.rand(15)
            ax.scatter([f(xx, yy)[0] for xx, yy in zip(sx, sy)], [f(xx, yy)[1] for xx, yy in zip(sx, sy)], s=6, color=COLOR_GT, alpha=0.8)
        else:
            ti = np.linspace(0, 1, 20)
            yc = 0.25 + 0.5 * (rs.randn(20).cumsum() / 20 + 0.5)
            ax.plot([f(lx + 0.05 + 0.12 * tii, 0.2 + 0.55 * yc[j])[0] for j, tii in enumerate(ti)], [f(lx + 0.05, 0.2 + 0.55 * yc[j])[1] for j in range(20)], color=COLOR_GT, lw=0.8, alpha=0.8)
        ax.text(*f(lx + 0.11, 0.04), label, ha="center", va="top", fontsize=8)
    ax.annotate("", xy=f(0.24, 0.75), xytext=f(0.02, 0.75), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=f(0.48, 0.75), xytext=f(0.26, 0.75), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=f(0.72, 0.75), xytext=f(0.50, 0.75), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(*f(0.5, 0.92), "Add noise", fontsize=7, ha="center", va="bottom")

def _draw_3d_slab(ax, f, x0, y0, w, h, depth=0.018, fc_top="white", fc_side="#b0b0b0", fc_front="#c8c8c8", ec="#555"):
    """在局部坐标 (x0,y0) 画宽 w 高 h 的 3D 薄条：顶面 + 右侧面 + 前面。"""
    # 顶面矩形 (x0, y0)-(x0+w, y0+h)
    pts_top = [f(x0, y0), f(x0 + w, y0), f(x0 + w, y0 + h), f(x0, y0 + h)]
    ax.add_patch(Polygon(pts_top, fc=fc_top, ec=ec, lw=0.6))
    # 前面（底面可见条）：(x0, y0)-(x0+w, y0)-(x0+w+depth, y0-depth)-(x0+depth, y0-depth)
    pts_front = [f(x0, y0), f(x0 + w, y0), f(x0 + w + depth, y0 - depth), f(x0 + depth, y0 - depth)]
    ax.add_patch(Polygon(pts_front, fc=fc_front, ec=ec, lw=0.5))
    # 右侧面：(x0+w, y0)-(x0+w, y0+h)-(x0+w+depth, y0+h-depth)-(x0+w+depth, y0-depth)
    pts_right = [f(x0 + w, y0), f(x0 + w, y0 + h), f(x0 + w + depth, y0 + h - depth), f(x0 + w + depth, y0 - depth)]
    ax.add_patch(Polygon(pts_right, fc=fc_side, ec=ec, lw=0.5))


def _fig1_block_B5(ax, bbox):
    """去噪网络：3D 薄条分层、每层有标识、箭头块到块、输入为 concat 关系。"""
    def f(x, y): return _to_fig(bbox, x, y)
    xm, xw = 0.06, 0.88
    depth = 0.016  # 3D 厚度

    # 外框
    box = FancyBboxPatch(f(0, 0), f(1, 1)[0] - f(0, 0)[0], f(1, 1)[1] - f(0, 0)[1], boxstyle="round,pad=0.02", ec=COLOR_MAIN_BOX, fc=FILL_MAIN_BOX)
    ax.add_patch(box)
    ax.text(*f(0.5, 0.95), r"Time-series conditional Transformer $\varepsilon_\theta$", ha="center", va="center", fontsize=9, fontweight="bold")

    # ----- 输入区：X_t,t 与 Mask 为 concat 关系（一个整体块内左右两格 + 箭头合并） -----
    inp_h = 0.068
    inp_y_top = 0.84
    inp_y_bot = inp_y_top - inp_h
    # 左格：X_t, t
    w1 = 0.36 * xw
    _draw_3d_slab(ax, f, xm, inp_y_bot, w1, inp_h, depth=depth, fc_top="white")
    ax.text(*f(xm + w1 / 2, inp_y_bot + inp_h / 2), r"$X_t$, $t$", ha="center", va="center", fontsize=8)
    # 中间：concat 符号 + 箭头指向“合并”
    ax.text(*f(xm + w1 + 0.08 * xw, inp_y_bot + inp_h / 2), r"$\oplus$", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(*f(xm + w1 + 0.08 * xw, inp_y_bot - 0.02), "concat", ha="center", va="top", fontsize=6)
    # 右格：Mask
    w2 = 0.36 * xw
    x2 = xm + w1 + 0.18 * xw
    _draw_3d_slab(ax, f, x2, inp_y_bot, w2, inp_h, depth=depth, fc_top="#f0f0f0")
    ax.text(*f(x2 + w2 / 2, inp_y_bot + inp_h / 2), "Mask", ha="center", va="center", fontsize=8)
    # 合并后的“输入块”标签（整块视为 Input 2ch）
    ax.text(*f(xm + xw / 2, inp_y_top + 0.02), "Input (2 ch)", ha="center", va="bottom", fontsize=7, style="italic")
    # 箭头：从输入块底面中心 指到 Enc1 顶面
    enc1_top_y = 0.658
    ax.annotate("", xy=(f(0.5, enc1_top_y)[0], f(0.5, enc1_top_y)[1]), xytext=(f(0.5, inp_y_bot - depth)[0], f(0.5, inp_y_bot - depth)[1]), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    # ----- Encoder：4 层 3D 薄条，每层有标识 -----
    ax.text(*f(xm, 0.70), "Encoder", ha="left", va="center", fontsize=7, fontweight="bold")
    layer_h = 0.028
    gap = 0.006
    enc_bottoms = []
    for i in range(4):
        y_top = 0.658 - i * (layer_h + gap)
        y_bot = y_top - layer_h
        enc_bottoms.append(y_bot - depth)
        _draw_3d_slab(ax, f, xm, y_bot, xw, layer_h, depth=depth, fc_top="white", fc_side="#a8a8a8", fc_front="#b8b8b8")
        ax.text(*f(xm + xw / 2, y_bot + layer_h / 2), "Enc %d" % (i + 1), ha="center", va="center", fontsize=7)
    enc_last_bottom = enc_bottoms[-1]
    dec1_top_y = 0.428
    # 箭头：Enc 4 底面 → Dec 1 顶面
    ax.annotate("", xy=(f(0.5, dec1_top_y)[0], f(0.5, dec1_top_y)[1]), xytext=(f(0.5, enc_last_bottom)[0], f(0.5, enc_last_bottom)[1]), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    # ----- Decoder：2 层 3D 薄条，每层有标识 -----
    ax.text(*f(xm, 0.44), "Decoder", ha="left", va="center", fontsize=7, fontweight="bold")
    dec_bottoms = []
    for i in range(2):
        y_top = dec1_top_y - 0.002 - i * (layer_h + gap)
        y_bot = y_top - layer_h
        dec_bottoms.append(y_bot - depth)
        _draw_3d_slab(ax, f, xm, y_bot, xw, layer_h, depth=depth, fc_top="#fafafa", fc_side="#a0a0a0", fc_front="#b0b0b0")
        ax.text(*f(xm + xw / 2, y_bot + layer_h / 2), "Dec %d" % (i + 1), ha="center", va="center", fontsize=7)
    dec_last_bottom = dec_bottoms[-1]
    out_top_y = 0.22
    # 箭头：Dec 2 底面 → Output 顶面
    ax.annotate("", xy=(f(0.5, out_top_y)[0], f(0.5, out_top_y)[1]), xytext=(f(0.5, dec_last_bottom)[0], f(0.5, dec_last_bottom)[1]), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    # ----- 输出区：3D 薄条块 -----
    out_h = 0.10
    out_y_bot = 0.12
    _draw_3d_slab(ax, f, xm, out_y_bot, xw, out_h, depth=depth, fc_top="white", fc_side="#a8a8a8", fc_front="#b8b8b8")
    ax.text(*f(0.5, out_y_bot + out_h / 2), r"Output: $\hat{\varepsilon}$ or $\hat{X}_0$", ha="center", va="center", fontsize=8)
    ax.text(*f(xm + xw, 0.04), r"$L_{freq}$ (opt.)", ha="right", va="center", fontsize=6, style="italic", color="#666")

def _fig1_block_B6(ax, bbox):
    """反向扩散：X_T -> X_0 四格 + Denoise。"""
    def f(x, y): return _to_fig(bbox, x, y)
    rs = np.random.RandomState(1)
    for i, (lx, label) in enumerate([(0.02, r"$X_T$"), (0.26, r"$X_t$"), (0.50, r"$X_t$"), (0.74, r"$X_0$")]):
        bw = 0.22
        bx = FancyBboxPatch(f(lx, 0.12), f(lx + bw, 0.12)[0] - f(lx, 0.12)[0], f(lx, 0.88)[1] - f(lx, 0.12)[1], boxstyle="round,pad=0.01", ec="#666", fc="white")
        ax.add_patch(bx)
        if i == 0:
            sx = lx + 0.08 + 0.12 * rs.rand(15)
            sy = 0.25 + 0.5 * rs.rand(15)
            ax.scatter([f(xx, yy)[0] for xx, yy in zip(sx, sy)], [f(xx, yy)[1] for xx, yy in zip(sx, sy)], s=6, color=COLOR_GT, alpha=0.85)
        elif i == 3:
            # 反向末端 X_0：去噪后的清晰时序（与训练端 X_0 示意一致，用强调色表示插补输出）
            ti = np.linspace(0, 1, 20)
            ax.plot(
                [f(lx + 0.05 + 0.12 * tii, 0.2 + 0.55 * np.sin(3 * tii))[0] for tii in ti],
                [f(lx + 0.05, 0.2 + 0.55 * np.sin(3 * tii))[1] for tii in ti],
                color=PALETTE_LI,
                lw=1.1,
            )
        else:
            ti = np.linspace(0, 1, 20)
            yc = 0.25 + 0.5 * (rs.randn(20).cumsum() / 20 + 0.5)
            ax.plot([f(lx + 0.05 + 0.12 * tii, 0.2 + 0.55 * yc[j])[0] for j, tii in enumerate(ti)], [f(lx + 0.05, 0.2 + 0.55 * yc[j])[1] for j in range(20)], color=COLOR_GT, lw=0.8, alpha=0.8)
        ax.text(*f(lx + 0.11, 0.04), label, ha="center", va="top", fontsize=8)
    ax.annotate("", xy=f(0.24, 0.75), xytext=f(0.02, 0.75), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=f(0.48, 0.75), xytext=f(0.26, 0.75), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=f(0.72, 0.75), xytext=f(0.50, 0.75), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(*f(0.5, 0.92), "Denoise", fontsize=7, ha="center", va="bottom")

def _fig1_block_B8(ax, bbox):
    """(A) 插补输出：曲线+95% CI + 标题在上方有间距。"""
    def f(x, y): return _to_fig(bbox, x, y)
    box = FancyBboxPatch(f(0.05, 0.12), f(0.95, 0)[0] - f(0.05, 0.12)[0], f(0, 0.88)[1] - f(0, 0.12)[1], boxstyle="round,pad=0.02", ec="#888", fc="#fafafa")
    ax.add_patch(box)
    tt = np.linspace(0, 1, 30)
    mu = 0.5 + 0.3 * np.sin(2 * np.pi * tt)
    xs = [f(0.08 + 0.84 * ti, 0.2 + 0.5 * (mu[i] - 0.15))[0] for i, ti in enumerate(tt)]
    ys = [f(0.08 + 0.84 * ti, 0.2 + 0.5 * (mu[i] - 0.15))[1] for i, ti in enumerate(tt)]
    ax.fill_between([f(0.08 + 0.84 * ti, 0)[0] for i, ti in enumerate(tt)], ys, [f(0.08 + 0.84 * ti, 0.2 + 0.5 * (mu[i] + 0.15))[1] for i, ti in enumerate(tt)], color=FILL_CI, alpha=0.4)
    ax.plot([f(0.08 + 0.84 * ti, 0.2 + 0.5 * mu[i])[0] for i, ti in enumerate(tt)], [f(0.08 + 0.84 * ti, 0.2 + 0.5 * mu[i])[1] for i, ti in enumerate(tt)], color=COLOR_IMPUTED, lw=1.2)
    ax.text(*f(0.5, 0.04), r"$\hat{X}$", ha="center", va="top", fontsize=10)
    ax.text(*f(0.5, 0.98), "(A) Imputation", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(*f(0.5, 0.90), "95% CI", ha="center", va="bottom", fontsize=7, style="italic")

def _fig1_block_B9(ax, bbox):
    """Downstream 框。"""
    def f(x, y): return _to_fig(bbox, x, y)
    ax.add_patch(FancyBboxPatch(f(0, 0), f(1, 1)[0] - f(0, 0)[0], f(1, 1)[1] - f(0, 0)[1], boxstyle="round,pad=0.03", ec="#666", fc="#f0f0f0"))
    ax.text(*f(0.5, 0.6), "Downstream", ha="center", va="center", fontsize=8)
    ax.text(*f(0.5, 0.35), "RUL / Forecast", ha="center", va="center", fontsize=7)

def _fig1_block_B10(ax, bbox):
    """(B) Generation 框。"""
    def f(x, y): return _to_fig(bbox, x, y)
    ax.add_patch(FancyBboxPatch(f(0, 0), f(1, 1)[0] - f(0, 0)[0], f(1, 1)[1] - f(0, 0)[1], boxstyle="round,pad=0.03", ec="#888", fc="#f5f5f5"))
    ax.plot([f(0.1 + 0.8 * ti, 0.35 + 0.35 * np.sin(4 * ti))[0] for ti in np.linspace(0, 1, 25)], [f(0.1 + 0.8 * ti, 0.35 + 0.35 * np.sin(4 * ti))[1] for ti in np.linspace(0, 1, 25)], color=COLOR_GT, lw=0.8, alpha=0.7)
    ax.text(*f(0.5, 0.08), "(B) Generation", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(*f(0.5, 0.45), "From noise", ha="center", va="center", fontsize=7, style="italic")

def _fig1_block_B11(ax, bbox):
    """图例：贴边。"""
    def f(x, y): return _to_fig(bbox, x, y)
    ax.add_patch(FancyBboxPatch(f(0, 0), f(1, 1)[0] - f(0, 0)[0], f(1, 1)[1] - f(0, 0)[1], boxstyle="round,pad=0.05", ec="#ccc", fc="#fafafa"))
    ax.plot([f(0.08, 0.72), f(0.28, 0.72)], [f(0.08, 0.72)[1], f(0.28, 0.72)[1]], color="#333", lw=2)
    ax.text(*f(0.35, 0.72), "Data flow", fontsize=7, va="center", ha="left")
    ax.plot([f(0.08, 0.42), f(0.28, 0.42)], [f(0.08, 0.42)[1], f(0.28, 0.42)[1]], color=COLOR_IMPUTED, lw=1.5, linestyle="--")
    ax.text(*f(0.35, 0.42), "Conditional control", fontsize=7, va="center", ha="left")
    ax.add_patch(Rectangle(f(0.08, 0.12), f(0.28, 0)[0] - f(0.08, 0.12)[0], f(0, 0.28)[1] - f(0, 0.12)[1], fc=FILL_CI, alpha=0.5, ec="none"))
    ax.text(*f(0.35, 0.20), "Uncertainty (95% CI)", fontsize=7, va="center", ha="left")

def _fig1_block_B12(ax, bbox):
    """底部子图标题。"""
    def f(x, y): return _to_fig(bbox, x, y)
    ax.text(*f(0.5, 0.5), "(a) DDI-E framework", ha="center", va="center", fontsize=11)

FIG1_BLOCK_DRAWERS = {
    "B0": _fig1_block_B0, "B1": _fig1_block_B1, "B2": _fig1_block_B2, "B3": _fig1_block_B3,
    "B4": _fig1_block_B4, "B5": _fig1_block_B5, "B6": _fig1_block_B6,
    "B8": _fig1_block_B8, "B9": _fig1_block_B9, "B10": _fig1_block_B10,
    "B11": _fig1_block_B11, "B12": _fig1_block_B12,
}

def fig1_framework_block_preview(block_id: str):
    """只画指定分块到独立图，便于逐步调整。block_id 如 'B0','B1',...,'B12'。"""
    if block_id not in FIG1_BLOCK_DRAWERS:
        raise ValueError("block_id must be one of %s" % list(FIG1_BLOCK_DRAWERS.keys()))
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    FIG1_BLOCK_DRAWERS[block_id](ax, (0, 0, 1, 1))
    out = FIG_DIR / ("fig1_block_%s.png" % block_id)  # FIG1-00 预览
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved FIG1-00 block preview:", out)

# ---------- FIG1-01：主框架（含整图二次绘制，可能与分块区域重叠） ----------
def fig1_framework():
    setup_style()
    fig, ax = plt.subplots(figsize=(14, 8.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 1) 按布局绘制各分块
    for bid, bbox in FIG1_LAYOUT.items():
        if bid in FIG1_BLOCK_DRAWERS:
            FIG1_BLOCK_DRAWERS[bid](ax, bbox)

    # 2) 块间连线（B7 及箭头）
    ax.annotate("", xy=(0.10, 0.44), xytext=(0.09, 0.58), arrowprops=dict(arrowstyle="->", color="gray", lw=1))
    ax.annotate("", xy=(0.10, 0.44), xytext=(0.09, 0.52), arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    # ----- Center: Forward diffusion (y 0.68-0.82), net (0.18-0.64), reverse (0.68-0.82) -----
    y_fwd, y_arrow = 0.68, 0.84  # label above arrow, no overlap with boxes
    for i, (xx, label) in enumerate([(0.22, r"$X_0$"), (0.28, r"$X_t$"), (0.34, r"$X_t$"), (0.40, r"$X_T$")]):
        bx = FancyBboxPatch((xx, 0.68), 0.055, 0.12, boxstyle="round,pad=0.005", ec="#666", fc="white")
        ax.add_patch(bx)
        if i == 0:
            ax.plot(xx + 0.01 + 0.035 * np.linspace(0, 1, 20), 0.71 + 0.08 * np.sin(3 * np.linspace(0, 1, 20)), color=COLOR_GT, lw=1)
        elif i == 3:
            ax.scatter(xx + 0.02 + 0.03 * np.random.RandomState(42).rand(15), 0.71 + 0.08 * np.random.RandomState(42).rand(15), s=4, color=COLOR_GT, alpha=0.8)
        else:
            ax.plot(xx + 0.01 + 0.035 * np.linspace(0, 1, 20), 0.71 + 0.06 * np.random.RandomState(i).randn(20).cumsum() / 20 + 0.04, color=COLOR_GT, lw=0.8, alpha=0.8)
        ax.text(xx + 0.0275, 0.675, label, ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(0.275, 0.74), xytext=(0.22, 0.74), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(0.335, 0.74), xytext=(0.28, 0.74), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(0.40, 0.74), xytext=(0.355, 0.74), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(0.31, y_arrow, "Add noise", fontsize=7, ha="center", va="bottom", rotation=0)

    net_box = FancyBboxPatch((0.20, 0.18), 0.24, 0.48, boxstyle="round,pad=0.02", ec=COLOR_MAIN_BOX, fc=FILL_MAIN_BOX)
    ax.add_patch(net_box)
    ax.text(0.32, 0.635, r"Time-series conditional Transformer $\varepsilon_\theta$", ha="center", va="center", fontsize=9, fontweight="bold")
    ax.text(0.32, 0.59, r"Input: $X_t$, $t$ (step embed)", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((0.23, 0.46), 0.18, 0.10, boxstyle="round,pad=0.01", ec="#666", fc="white"))
    ax.text(0.32, 0.51, "Encoder–Decoder", ha="center", va="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((0.23, 0.38), 0.18, 0.06, boxstyle="round,pad=0.01", ec="#666", fc="white"))
    ax.text(0.32, 0.41, "AdaLN, Mask as 2nd ch", ha="center", va="center", fontsize=7)
    ax.text(0.32, 0.33, r"Optional: $L_{freq}$", ha="center", va="center", fontsize=7, style="italic")
    ax.text(0.32, 0.24, r"Output: $\hat{\varepsilon}$ or $\hat{X}_0$", ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(0.32, 0.66), xytext=(0.29, 0.68), arrowprops=dict(arrowstyle="->", color="#333", lw=1))
    ax.annotate("", xy=(0.44, 0.53), xytext=(0.44, 0.68), arrowprops=dict(arrowstyle="->", color="#333", lw=1))

    # Reverse diffusion
    for i, (xx, label) in enumerate([(0.48, r"$X_T$"), (0.54, r"$X_t$"), (0.60, r"$X_t$"), (0.66, r"$X_0$")]):
        bx = FancyBboxPatch((xx, 0.68), 0.055, 0.12, boxstyle="round,pad=0.005", ec="#666", fc="white")
        ax.add_patch(bx)
        if i == 0:
            ax.scatter(xx + 0.02 + 0.03 * np.random.RandomState(1).rand(15), 0.71 + 0.08 * np.random.RandomState(1).rand(15), s=4, color=COLOR_GT, alpha=0.8)
        elif i == 3:
            ax.plot(xx + 0.01 + 0.035 * np.linspace(0, 1, 20), 0.71 + 0.08 * np.sin(3 * np.linspace(0, 1, 20)), color=PALETTE_LI, lw=1)
        else:
            ax.plot(xx + 0.01 + 0.035 * np.linspace(0, 1, 20), 0.71 + 0.06 * np.random.RandomState(i + 10).randn(20).cumsum() / 20 + 0.04, color=COLOR_GT, lw=0.8, alpha=0.8)
        ax.text(xx + 0.0275, 0.675, label, ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(0.535, 0.74), xytext=(0.48, 0.74), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(0.595, 0.74), xytext=(0.54, 0.74), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.annotate("", xy=(0.66, 0.74), xytext=(0.615, 0.74), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(0.57, y_arrow, "Denoise", fontsize=7, ha="center", va="bottom", rotation=0)
    # Conditional control: route around net_box (no crossing). Path: (0.17,0.35)->(0.17,0.70)->(0.46,0.70)->(0.46,0.66)
    ax.plot([0.17, 0.17, 0.46], [0.35, 0.70, 0.70], color=COLOR_IMPUTED, ls="--", lw=1)
    ax.annotate("", xy=(0.46, 0.66), xytext=(0.46, 0.70), arrowprops=dict(arrowstyle="->", color=COLOR_IMPUTED, lw=1))
    ax.text(0.30, 0.71, "Obs. replacement", fontsize=7, color=COLOR_IMPUTED, ha="center", va="bottom", rotation=0)
    ax.annotate("", xy=(0.44, 0.68), xytext=(0.44, 0.66), arrowprops=dict(arrowstyle="->", color="#333", lw=1))

    # ----- Right: (A) Imputation box 0.48-0.68; labels above with gap; Downstream 0.34-0.46; (B) 0.18-0.30 -----
    imp_y0, imp_y1 = 0.48, 0.68
    imp_box = FancyBboxPatch((0.72, imp_y0), 0.12, imp_y1 - imp_y0, boxstyle="round,pad=0.01", ec="#888", fc="#fafafa")
    ax.add_patch(imp_box)
    tt = np.linspace(0, 1, 30)
    mu = 0.5 + 0.3 * np.sin(2 * np.pi * tt)
    ax.fill_between(0.73 + 0.10 * tt, 0.54 + 0.06 * (mu - 0.15), 0.54 + 0.06 * (mu + 0.15), color=FILL_CI, alpha=0.4)
    ax.plot(0.73 + 0.10 * tt, 0.54 + 0.06 * mu, color=COLOR_IMPUTED, lw=1.2)
    ax.text(0.78, 0.465, r"$\hat{X}$", ha="center", va="top", fontsize=9)
    # Data flow from reverse diffusion X_0 to (A): horizontal then down, no crossing
    ax.plot([0.66, 0.71, 0.71], [0.74, 0.74, 0.68], color="#333", lw=1.2)
    ax.annotate("", xy=(0.72, 0.68), xytext=(0.71, 0.68), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(0.78, 0.72, "(A) Imputation", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.text(0.78, 0.70, "95% CI", ha="center", va="bottom", fontsize=7, style="italic")

    ax.add_patch(FancyBboxPatch((0.72, 0.32), 0.12, 0.12, boxstyle="round,pad=0.01", ec="#666", fc="#f0f0f0"))
    ax.text(0.78, 0.41, "Downstream", ha="center", va="center", fontsize=8)
    ax.text(0.78, 0.36, "RUL / Forecast", ha="center", va="center", fontsize=7)
    ax.annotate("", xy=(0.78, 0.44), xytext=(0.78, 0.48), arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))
    ax.text(0.78, 0.28, "Improved accuracy", ha="center", va="center", fontsize=7, style="italic")
    ax.annotate("", xy=(0.78, 0.32), xytext=(0.78, 0.32), arrowprops=dict(arrowstyle="->", color="#333", lw=1))

    ax.add_patch(FancyBboxPatch((0.72, 0.14), 0.12, 0.14, boxstyle="round,pad=0.01", ec="#888", fc="#f5f5f5"))
    ax.plot(0.73 + 0.10 * np.linspace(0, 1, 25), 0.19 + 0.06 * np.sin(4 * np.linspace(0, 1, 25)), color=COLOR_GT, lw=0.8, alpha=0.7)
    ax.text(0.78, 0.13, "(B) Generation", ha="center", va="top", fontsize=8, fontweight="bold")
    ax.text(0.78, 0.20, "From noise", ha="center", va="center", fontsize=7, style="italic")
    # Arrow from reverse X_0 to (B): route right then down to top of (B), no overlap with (A)
    ax.plot([0.66, 0.69, 0.69], [0.74, 0.74, 0.29], color="gray", lw=1)
    ax.annotate("", xy=(0.78, 0.28), xytext=(0.69, 0.29), arrowprops=dict(arrowstyle="->", color="gray", lw=1))

    # ----- Legend: bottom-right, aligned to right edge, in a frame -----
    leg_x0, leg_x1 = 0.68, 0.98
    leg_y0, leg_y1 = 0.02, 0.12
    ax.add_patch(FancyBboxPatch((leg_x0, leg_y0), leg_x1 - leg_x0, leg_y1 - leg_y0, boxstyle="round,pad=0.02", ec="#ccc", fc="#fafafa"))
    ax.plot([leg_x0 + 0.04, leg_x0 + 0.10], [0.095, 0.095], color="#333", lw=2)
    ax.text(leg_x0 + 0.14, 0.095, "Data flow", fontsize=7, va="center", ha="left")
    ax.plot([leg_x0 + 0.04, leg_x0 + 0.10], [0.065, 0.065], color=COLOR_IMPUTED, lw=1.5, linestyle="--")
    ax.text(leg_x0 + 0.14, 0.065, "Conditional control", fontsize=7, va="center", ha="left")
    ax.add_patch(Rectangle((leg_x0 + 0.04, 0.032), 0.06, 0.022, fc=FILL_CI, alpha=0.5, ec="none"))
    ax.text(leg_x0 + 0.14, 0.043, "Uncertainty (95% CI)", fontsize=7, va="center", ha="left")

    ax.text(0.5, 0.01, "(a) DDI-E framework", ha="center", va="center", fontsize=11)
    ax.text(0.5, 0.93, "DDI-E: Conditional diffusion for electrical time series imputation", ha="center", va="center", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_ddie_framework.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved FIG1-01:", FIG_DIR / "fig1_ddie_framework.png")


# ---------- FIG1-02：分块拼接 + 连线（推荐替代 FIG1-01 叠画问题） ----------
def fig1_framework_redesign_blocks(out_path: Path | str | None = None) -> Path:
    """
    FIG1-02：FIG1_LAYOUT 各分块 + 块间连线，无二次叠画。
    默认输出 fig1_FIG1-02_blocks_NNN.png（序号递增）。
    """
    setup_style()
    if out_path is None:
        out_path = _fig1_next_serial("fig1_FIG1-02_blocks")
    else:
        out_path = Path(out_path)

    def T(bid, x, y):
        return _to_fig(FIG1_LAYOUT[bid], x, y)

    fig, ax = plt.subplots(figsize=(15, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for bid, bbox in FIG1_LAYOUT.items():
        if bid in FIG1_BLOCK_DRAWERS:
            FIG1_BLOCK_DRAWERS[bid](ax, bbox)

    # 左列：观测 / 掩码 → 条件构造
    p_b1 = T("B1", 0.5, 0.14)
    p_b2 = T("B2", 0.5, 0)
    p_b3_in = T("B3", 0.5, 0.99)
    ax.annotate("", xy=(p_b3_in[0], p_b3_in[1] - 0.008), xytext=(p_b1[0], p_b1[1]), arrowprops=dict(arrowstyle="->", color="gray", lw=1.1))
    ax.annotate("", xy=(p_b3_in[0], p_b3_in[1] - 0.008), xytext=(p_b2[0], p_b2[1]), arrowprops=dict(arrowstyle="->", color="gray", lw=1.1))

    # B7：条件控制虚线 → 网络顶部右侧（绕开 B5 主体）
    p_cond = T("B3", 0.92, 0.45)
    ax.plot(
        [p_cond[0], 0.17, 0.17, 0.445, 0.445],
        [p_cond[1], p_cond[1], 0.695, 0.695, T("B5", 0.95, 0.92)[1]],
        color=COLOR_IMPUTED,
        ls="--",
        lw=1.05,
    )
    ax.annotate(
        "",
        xy=(0.445, T("B5", 0.92, 0.88)[1]),
        xytext=(0.445, 0.695),
        arrowprops=dict(arrowstyle="->", color=COLOR_IMPUTED, lw=1),
    )
    ax.text(0.30, 0.705, "Obs. replacement", fontsize=7, color=COLOR_IMPUTED, ha="center", va="bottom")

    # 前向扩散块底 → 网络顶；网络顶 → 反向扩散块底
    p_b4_bot = T("B4", 0.5, 0.02)
    p_b5_top = T("B5", 0.5, 0.98)
    p_b6_bot = T("B6", 0.5, 0.02)
    ax.annotate("", xy=(p_b5_top[0], p_b5_top[1] + 0.004), xytext=(p_b4_bot[0], p_b4_bot[1]), arrowprops=dict(arrowstyle="->", color="#333", lw=1.15))
    ax.annotate("", xy=(p_b6_bot[0], p_b6_bot[1]), xytext=(p_b5_top[0], p_b5_top[1] + 0.004), arrowprops=dict(arrowstyle="->", color="#333", lw=1.15))

    # 反向链末端 X_0 → (A) 插补
    p_x0 = T("B6", 0.86, 0.5)
    p_b8_in = T("B8", 0.02, 0.52)
    ax.plot([p_x0[0] + 0.02, 0.71, 0.71], [p_x0[1], p_x0[1], p_b8_in[1]], color="#333", lw=1.15)
    ax.annotate("", xy=(p_b8_in[0], p_b8_in[1]), xytext=(0.71, p_b8_in[1]), arrowprops=dict(arrowstyle="->", color="#333", lw=1.15))

    # (A) → Downstream
    p_b8_bot = T("B8", 0.5, 0.06)
    p_b9_top = T("B9", 0.5, 0.95)
    ax.annotate("", xy=(p_b9_top[0], p_b9_top[1]), xytext=(p_b8_bot[0], p_b8_bot[1]), arrowprops=dict(arrowstyle="->", color="#333", lw=1.1))

    # 反向 → (B) Generation（灰色支路，绕开 B8）
    ax.plot(
        [T("B6", 0.72, 0.5)[0], 0.695, 0.695],
        [T("B6", 0.72, 0.5)[1], T("B6", 0.72, 0.5)[1], T("B10", 0.5, 0.92)[1]],
        color="gray",
        lw=1,
    )
    ax.annotate(
        "",
        xy=(T("B10", 0.5, 0.88)[0], T("B10", 0.5, 0.88)[1]),
        xytext=(0.695, T("B10", 0.5, 0.92)[1]),
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "DDI-E FIG1-02 blocks composite"},
    )
    plt.close(fig)
    print("Saved FIG1-02 (no overwrite):", out_path.resolve())
    return out_path


# ---------- FIG1-03：三栏规范示意（Figure1-DDI-E-framework-redesign 1.md） ----------
def fig1_framework_redesign_threecolumn(out_path: Path | str | None = None) -> Path:
    """
    FIG1-03：三栏式 + 脚注；默认 fig1_FIG1-03_threecolumn_NNN.png。
    """
    setup_style()
    np.random.seed(42)
    if out_path is None:
        out_path = _fig1_next_serial("fig1_FIG1-03_threecolumn")
    else:
        out_path = Path(out_path)

    # 略放大高度以容纳脚注
    fig_w, fig_h, dpi = 11.0, 6.35, 300
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi, facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    title_fs, label_fs, small_fs, foot_fs = 13, 10, 8.5, 7.5
    c_in, c_enc, c_dec, c_out = "#1565C0", "#1976D2", "#388E3C", "#C62828"
    fc_enc, fc_dec = "#BBDEFB", "#C8E6C9"
    lw_main, lw_thick = 1.2, 2.0

    # ----- 左栏 x≈0.05–0.30 -----
    xl0, xl1 = 0.06, 0.29
    ax.text(0.175, 0.96, "Input", ha="center", va="top", fontsize=title_fs, fontweight="bold", color=c_in)
    ax.text(
        0.175,
        0.91,
        r"Observed sequence with missing values $\mathbf{X}^{\mathrm{obs}}$",
        ha="center",
        va="top",
        fontsize=small_fs,
        color="#333",
    )

    t = np.linspace(0, 1, 140)
    y_cap = 0.78 - 0.20 * t + 0.012 * np.sin(6 * np.pi * t)
    x_cap = xl0 + (xl1 - xl0) * t
    miss = (t >= 0.34) & (t <= 0.55)
    ax.plot(x_cap[~miss], y_cap[~miss], color=c_in, lw=2.3, solid_capstyle="round")
    ax.plot(x_cap[miss], y_cap[miss], color=c_in, lw=2.3, ls=":", alpha=0.9)
    idx_vis = [0, 32, 56, 85, 139]
    ax.scatter(x_cap[idx_vis], y_cap[idx_vis], c="#0D47A1", s=36, zorder=5, edgecolors="white", linewidths=0.6)

    my0, mh = 0.11, 0.028
    nseg = 22
    edges = np.linspace(xl0, xl1, nseg + 1)
    for i in range(nseg):
        tc = 0.5 * (edges[i] + edges[i + 1] - 2 * xl0) / (xl1 - xl0)
        is_miss = 0.34 <= tc <= 0.55
        fc = "#1a1a1a" if is_miss else "#f5f5f5"
        ax.add_patch(Rectangle((edges[i], my0), edges[i + 1] - edges[i], mh, facecolor=fc, edgecolor="#888", linewidth=0.4))
    ax.text(0.175, my0 - 0.012, r"Mask $\mathbf{M}$ (1: observed, 0: missing)", ha="center", va="top", fontsize=small_fs - 0.5)

    ax.annotate(
        "",
        xy=(0.22, my0 + mh + 0.01),
        xytext=(0.28, 0.84),
        arrowprops=dict(arrowstyle="->", color="#F57C00", lw=1.2),
    )
    ax.text(
        0.29,
        0.86,
        "Conditioning via\nmask only",
        ha="left",
        va="center",
        fontsize=small_fs,
        color="#E65100",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFF9C4", edgecolor="#F9A825", linewidth=0.8),
    )

    # ----- 中栏 x≈0.33–0.67 -----
    xc0, xc1 = 0.33, 0.67
    ax.add_patch(Rectangle((xc0, 0.62), xc1 - xc0, 0.26, facecolor="#E0E0E0", edgecolor="none", alpha=0.45))
    ax.text(0.5, 0.875, "Forward diffusion (training only)", ha="center", va="center", fontsize=label_fs, fontweight="bold", color="#424242")
    y_fwd = 0.745
    ax.annotate("", xy=(0.63, y_fwd), xytext=(0.37, y_fwd), arrowprops=dict(arrowstyle="->", color="#555", lw=lw_thick))
    ax.text(0.36, y_fwd + 0.028, r"$\mathbf{X}_0$", ha="center", fontsize=9)
    ax.text(0.5, y_fwd + 0.028, r"$\epsilon_t$", ha="center", fontsize=9, style="italic")
    ax.text(0.64, y_fwd + 0.028, r"$\mathbf{X}_T$", ha="center", fontsize=9)
    ax.text(
        0.5,
        0.665,
        r"$\mathbf{X}_t=\sqrt{\bar{\alpha}_t}\mathbf{X}_0+\sqrt{1-\bar{\alpha}_t}\epsilon$",
        ha="center",
        fontsize=9,
        style="italic",
        color="#333",
    )

    ax.text(0.5, 0.545, r"Conditional denoising network $\varepsilon_\theta$", ha="center", fontsize=label_fs, fontweight="bold")
    ey0, eh, ew = 0.14, 0.22, 0.12
    ex_enc, ex_dec = 0.385, 0.545
    ax.add_patch(FancyBboxPatch((ex_enc, ey0), ew, eh, boxstyle="round,pad=0.012", ec=c_enc, fc=fc_enc, linewidth=1.4))
    ax.add_patch(FancyBboxPatch((ex_dec, ey0), ew, eh, boxstyle="round,pad=0.012", ec=c_dec, fc=fc_dec, linewidth=1.4))
    ax.text(ex_enc + ew / 2, ey0 + eh * 0.62, "Transformer\nEncoder", ha="center", va="center", fontsize=small_fs, fontweight="bold")
    ax.text(ex_enc + ew / 2, ey0 + 0.04, r"AdaLN for step $t$", ha="center", fontsize=7.5, color=c_enc)
    ax.text(ex_dec + ew / 2, ey0 + eh * 0.62, "Transformer\nDecoder", ha="center", va="center", fontsize=small_fs, fontweight="bold")
    ax.text(
        ex_dec + ew / 2,
        ey0 + 0.04,
        "Cross-attn + trend–season",
        ha="center",
        fontsize=7,
        color=c_dec,
    )
    ax.annotate("", xy=(ex_dec - 0.008, ey0 + eh * 0.5), xytext=(ex_enc + ew + 0.008, ey0 + eh * 0.5), arrowprops=dict(arrowstyle="->", color="#333", lw=lw_main))

    ax.text(ex_enc + ew * 0.5, ey0 + eh + 0.035, r"$\mathbf{X}_t$, $t$, $\mathbf{M}$ (ch.)", ha="center", fontsize=8)
    ax.annotate("", xy=(ex_enc + ew * 0.5, ey0 + eh), xytext=(ex_enc + ew * 0.5, ey0 + eh + 0.028), arrowprops=dict(arrowstyle="->", color="#333", lw=1))

    ax.annotate(
        "",
        xy=(ex_enc + 0.02, ey0 + 0.06),
        xytext=(0.175, my0 + mh * 0.5),
        arrowprops=dict(arrowstyle="->", color="#BF360C", lw=1.1, connectionstyle="arc3,rad=0.15"),
    )
    ax.text(
        0.26,
        0.05,
        "Mask concatenated\nas additional channel",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#BF360C",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFECB3", edgecolor="#FF8F00", linewidth=0.6),
    )
    ax.text(
        0.5,
        0.095,
        "No operating condition labels",
        ha="center",
        va="center",
        fontsize=small_fs,
        style="italic",
        fontweight="bold",
        color="#B71C1C",
    )

    # ----- 右栏 x≈0.70–0.95 -----
    xr0, xr1 = 0.71, 0.94
    ax.text(0.825, 0.96, "Output", ha="center", va="top", fontsize=title_fs, fontweight="bold", color=c_out)
    tr = np.linspace(0, 1, 160)
    yr = 0.78 - 0.20 * tr + 0.006 * np.sin(5 * np.pi * tr)
    xr = xr0 + (xr1 - xr0) * tr
    band = 0.045
    ax.fill_between(xr, yr - band, yr + band, color="#FFCDD2", alpha=0.55, edgecolor="none")
    ax.plot(xr, yr, color=c_out, lw=2.4, solid_capstyle="round")
    ax.text(0.825, 0.88, r"Imputed $\hat{\mathbf{X}}$ with 95% CI", ha="center", va="top", fontsize=label_fs, color=c_out, fontweight="bold")

    gen_y0, gen_h = 0.08, 0.14
    ax.add_patch(
        FancyBboxPatch(
            (xr0, gen_y0),
            xr1 - xr0,
            gen_h,
            boxstyle="round,pad=0.02",
            facecolor="none",
            ec="#9E9E9E",
            linewidth=1.2,
            linestyle="--",
        )
    )
    ax.plot(xr0 + (xr1 - xr0) * tr, gen_y0 + gen_h * 0.55 + 0.04 * np.sin(4 * np.pi * tr), color="#757575", lw=1.2, ls="--")
    ax.text(0.825, gen_y0 + gen_h * 0.48, r"Unconditional $\tilde{\mathbf{X}}$ (optional)", ha="center", va="center", fontsize=7.5, color="#616161")
    ax.annotate(
        "",
        xy=(0.825, gen_y0 + gen_h + 0.02),
        xytext=(0.825, 0.72),
        arrowprops=dict(arrowstyle="->", color="#9E9E9E", lw=1.2, linestyle="--"),
    )

    # ----- 三栏粗箭头 -----
    ax.annotate("", xy=(0.325, 0.48), xytext=(0.305, 0.48), arrowprops=dict(arrowstyle="->", color="#424242", lw=lw_thick))
    ax.annotate("", xy=(0.695, 0.48), xytext=(0.675, 0.48), arrowprops=dict(arrowstyle="->", color="#424242", lw=lw_thick))

    foot = (
        "Figure 1. Schematic of the DDI-E framework. Compared to the original DDI for bearing data, DDI-E operates on "
        "single-channel electrical signals and uses only the mask as conditioning (no operating condition labels). "
        "The forward diffusion (top center) is used only during training; the reverse denoising network (bottom center) "
        "iteratively reconstructs the sequence conditioned on the mask. The output includes a 95% confidence interval "
        "obtained via multiple stochastic samplings."
    )
    foot_lines = textwrap.fill(foot, width=118).split("\n")
    fy = 0.082
    for li, line in enumerate(foot_lines):
        ax.text(0.5, fy - li * 0.022, line, ha="center", va="top", fontsize=foot_fs, style="italic", color="#333")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.12,
        facecolor="white",
        metadata={"Software": "DDI-E FIG1-03 three-column spec"},
    )
    plt.close(fig)
    print("Saved FIG1-03 (no overwrite):", out_path.resolve())
    return out_path


def fig1_framework_redesign(out_path: Path | str | None = None) -> Path:
    """CLI fig1_redesign → FIG1-03（三栏规范版）。"""
    return fig1_framework_redesign_threecolumn(out_path)


# ---------- Fig 2 & 3: NASA (Battery & IGBT) bar charts ----------
def _draw_mae_rmse_bars(ax1, ax2, data, rates, methods, colors, subtitles):
    x = np.arange(len(rates))
    width = 0.22
    bars1 = []
    bars2 = []
    for i, method in enumerate(methods):
        off = (i - 1) * width
        means_mae = [data[mr][method]["MAE"][0] for mr in rates]
        stds_mae = [data[mr][method]["MAE"][1] for mr in rates]
        b1 = ax1.bar(
            x + off,
            means_mae,
            width,
            yerr=stds_mae,
            label=method,
            color=colors[method],
            capsize=3,
            edgecolor="white",
            linewidth=0.8,
        )
        means_rmse = [data[mr][method]["RMSE"][0] for mr in rates]
        stds_rmse = [data[mr][method]["RMSE"][1] for mr in rates]
        b2 = ax2.bar(
            x + off,
            means_rmse,
            width,
            yerr=stds_rmse,
            label=method,
            color=colors[method],
            capsize=3,
            edgecolor="white",
            linewidth=0.8,
        )
        bars1.append((method, b1, stds_mae))
        bars2.append((method, b2, stds_rmse))

    def _val_txt(v: float) -> str:
        if not np.isfinite(v):
            return ""
        if abs(v) >= 1:
            return f"{v:.2f}"
        if abs(v) >= 0.1:
            return f"{v:.3f}"
        return f"{v:.4f}"

    def _add_labels(ax, bar_groups):
        ymax = 0.0
        for method, cont, stds in bar_groups:
            # Move KNN (green) labels up by ~1 character height
            extra_pts = 8 if method == "KNN" else 0
            for rect, s in zip(cont.patches, stds):
                h = float(rect.get_height())
                if not np.isfinite(h):
                    continue
                ymax = max(ymax, h + (float(s) if np.isfinite(s) else 0.0))
                ax.annotate(
                    _val_txt(h),
                    (rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3 + extra_pts),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    clip_on=False,
                )
        if ymax > 0:
            # Increase headroom above bars (~2 character heights)
            ax.set_ylim(0, ymax * 1.28)

    _add_labels(ax1, bars1)
    _add_labels(ax2, bars2)
    for ax, sub in zip([ax1, ax2], subtitles):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(r*100)}%" for r in rates], fontsize=10)
        ax.set_ylabel("MAE" if ax == ax1 else "RMSE", fontsize=10)
        # Legend: 1 row x 3 columns, centered in the headroom band (no bar overlap).
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=True,
            fontsize=9,
            borderaxespad=0.2,
            handlelength=1.6,
            columnspacing=1.4,
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)
        set_subtitle(ax, sub)
    format_axes_line(ax2, xlabel="Missing rate", ylabel="RMSE (normalized)")
    format_axes_line(ax1, xlabel="Missing rate", ylabel="MAE (normalized)")


def fig2_nasa_imputation():
    setup_style()
    data = load_imputation_results_ddie03("nasa_battery") if USE_DDIE03 else load_imputation_results("nasa_battery")
    if not data:
        print("Skip fig2: no nasa_battery results")
        return
    rates = sorted(data.keys())
    methods = ["LI", "KNN", "DDI-E"]
    colors = {"LI": PALETTE_LI, "KNN": PALETTE_KNN, "DDI-E": PALETTE_DDIE}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    _draw_mae_rmse_bars(ax1, ax2, data, rates, methods, colors, ["(a) NASA Battery: MAE", "(b) NASA Battery: RMSE"])
    unit = " (Ah)" if USE_DDIE03 else " (normalized)"
    ax1.set_ylabel(f"MAE{unit}")
    ax2.set_ylabel(f"RMSE{unit}")
    # Reserve headroom for per-axes legends (centered at top).
    fig.subplots_adjust(left=0.07, right=0.99, top=0.82, bottom=0.22, wspace=0.18)
    fig.savefig(FIG_DIR / "fig2_nasa_imputation_mae_rmse.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG_DIR / "fig2_nasa_imputation_mae_rmse.png")


def fig3_nasa_igbt_imputation():
    """DDIE03: NASA IGBT imputation MAE/RMSE (process verification; LI/KNN ~0 in raw scale)."""
    setup_style()
    data = load_imputation_results_ddie03("nasa_igbt") if USE_DDIE03 else {}
    if not USE_DDIE03:
        print("Skip fig3: DDIE v1.0 focuses on NASA Battery + NASA IGBT (DDIE03).")
        return
    if not data:
        print("Skip fig3: no nasa_igbt DDIE03 results")
        return
    # NASA IGBT: MAE/RMSE 均为 0.0000（量纲小），报告建议仅用 MAPE 表示方法差异，避免全零柱状图
    rates = sorted(data.keys())
    methods = ["LI", "KNN", "DDI-E"]
    colors = {"LI": PALETTE_LI, "KNN": PALETTE_KNN, "DDI-E": PALETTE_DDIE}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(rates))
    width = 0.22
    for i, method in enumerate(methods):
        off = (i - 1) * width
        means = [data[mr][method]["MAPE"][0] for mr in rates]
        stds = [data[mr][method]["MAPE"][1] for mr in rates]
        ax.bar(x + off, means, width, yerr=stds, label=method, color=colors[method], capsize=3, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r*100)}%" for r in rates], fontsize=10)
    ax.set_ylabel("MAPE (%)", fontsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        fontsize=9,
        borderaxespad=0.2,
        handlelength=1.6,
        columnspacing=1.4,
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)
    format_axes_line(ax, "Missing rate", "MAPE (%)")
    tops = []
    for mr in rates:
        for method in methods:
            m = data[mr][method]["MAPE"][0]
            s = data[mr][method]["MAPE"][1]
            tops.append(float(m) + float(s))
    if tops:
        ax.set_ylim(0, max(tops) * 1.28)
    fig.subplots_adjust(left=0.10, right=0.99, top=0.86, bottom=0.18)
    out_main = FIG_DIR / "fig3_nasa_igbt_imputation_mae_rmse.png"
    fig.savefig(out_main, dpi=300, bbox_inches="tight", facecolor="white")
    # Paper numbering (Figure 5): keep an alias file name to avoid manual renaming.
    out_alias = FIG_DIR / "fig5_nasa_igbt_imputation_mae_rmse.png"
    try:
        fig.savefig(out_alias, dpi=300, bbox_inches="tight", facecolor="white")
    except Exception:
        pass
    plt.close(fig)
    print("Saved:", out_main)
    print("Saved alias (Figure 5):", out_alias)


# ---------- Fig 4: Method selection (FancyBboxPatch style) ----------
def fig4_method_selection():
    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    b1 = FancyBboxPatch((0.5, 4.0), 2.5, 1.2, boxstyle="round,pad=0.02", ec="#27ae60", fc="#d5f4e6")
    b2 = FancyBboxPatch((3.5, 4.0), 2.5, 1.2, boxstyle="round,pad=0.02", ec=COLOR_MAIN_BOX, fc=FILL_MAIN_BOX)
    b3 = FancyBboxPatch((6.5, 4.0), 2.5, 1.2, boxstyle="round,pad=0.02", ec=COLOR_UNCERTAIN, fc=FILL_UNCERTAIN)
    ax.add_patch(b1)
    ax.add_patch(b2)
    ax.add_patch(b3)
    ax.text(1.75, 4.6, "Univariate, smooth,\nsmall sample", ha="center", va="center", fontsize=10)
    ax.text(4.75, 4.6, "Multivariate, nonlinear,\nmulti-condition", ha="center", va="center", fontsize=10)
    ax.text(7.75, 4.6, "Need uncertainty or\nphysical/system", ha="center", va="center", fontsize=10)

    ax.add_patch(FancyArrowPatch((1.75, 4.0), (1.75, 3.25), arrowstyle="->", mutation_scale=12, color="#27ae60"))
    ax.add_patch(FancyArrowPatch((4.75, 4.0), (4.75, 3.25), arrowstyle="->", mutation_scale=12, color=COLOR_MAIN_BOX))
    ax.add_patch(FancyArrowPatch((7.75, 4.0), (7.75, 3.25), arrowstyle="->", mutation_scale=12, color=COLOR_UNCERTAIN))

    r1 = FancyBboxPatch((0.2, 1.7), 3.1, 1.2, boxstyle="round,pad=0.02", ec="#1abc9c", fc="#e8f8f5")
    r2 = FancyBboxPatch((3.2, 1.7), 3.1, 1.2, boxstyle="round,pad=0.02", ec=COLOR_MAIN_BOX, fc=FILL_MAIN_BOX)
    r3 = FancyBboxPatch((6.2, 1.7), 3.1, 1.2, boxstyle="round,pad=0.02", ec="#f39c12", fc="#fef9e7")
    ax.add_patch(r1)
    ax.add_patch(r2)
    ax.add_patch(r3)
    ax.text(1.75, 2.3, "LI, KNN\n(simple methods)", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(4.75, 2.3, "Conditional diffusion\n(DDI / DDI-E)", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(7.75, 2.3, "PINN + MBD or\nTransformer system", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.text(5, 5.6, "Method selection framework", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_method_selection_framework.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG_DIR / "fig4_method_selection_framework.png")


# ---------- Fig 5: Applicability boundary (clean fill + scatter) ----------
def fig5_applicability_boundary():
    setup_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.fill_between([0, 4.5], 0, 10, color="#d5f4e6", alpha=0.5, label="Simple methods sufficient")
    ax.fill_between([4.5, 10], 0, 10, color=FILL_MAIN_BOX, alpha=0.5, label="Diffusion beneficial")
    ax.axvline(4.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.scatter(2, 5, s=220, c="#27ae60", marker="o", edgecolors="#1e8449", linewidths=2, zorder=5)
    ax.scatter(7.5, 6, s=220, c=COLOR_MAIN_BOX, marker="s", edgecolors="#2c5aa0", linewidths=2, zorder=5)
    ax.annotate("NASA Battery\n(simple, univariate)", (2, 5), xytext=(2, 2.2), fontsize=10, ha="center", arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.annotate("XJTU Bearing\n(complex, multi-condition)", (7.5, 6), xytext=(7.5, 3.2), fontsize=10, ha="center", arrowprops=dict(arrowstyle="->", color="gray", lw=1.2))
    ax.set_xlabel("Data complexity (simple → complex)", fontsize=11)
    ax.set_ylabel("Task / method suitability", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.3)
    set_subtitle(ax, "Applicability boundary: when to use simple vs. diffusion imputation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_applicability_boundary.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG_DIR / "fig5_applicability_boundary.png")


# ---------- Fig 6: Combined NASA Battery + NASA IGBT (DDIE03) ----------
def fig6_combined_mae():
    setup_style()
    if USE_DDIE03:
        nasa = load_imputation_results_ddie03("nasa_battery")
        second = load_imputation_results_ddie03("nasa_igbt")
        second_title = "NASA IGBT"
        metric2 = "MAPE"  # IGBT MAE≈0，用 MAPE 才有区分度
        ylabel1, ylabel2 = "MAE (Ah)", "MAPE (%)"
        out_name = "fig6_combined_nasa_battery_igbt_mae.png"
        suptitle = "Imputation: NASA Battery (MAE) vs NASA IGBT (MAPE, DDIE03)"
    else:
        print("Skip fig6: DDIE v1.0 focuses on NASA Battery + NASA IGBT (DDIE03).")
        return
    if not nasa or not second:
        print("Skip fig6: missing nasa or second-dataset results")
        return
    methods = ["LI", "KNN", "DDI-E"]
    colors = {"LI": PALETTE_LI, "KNN": PALETTE_KNN, "DDI-E": PALETTE_DDIE}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    # 左：NASA Battery MAE
    rates1 = sorted(nasa.keys())
    x1 = np.arange(len(rates1))
    width = 0.22
    for i, method in enumerate(methods):
        off = (i - 1) * width
        means = [nasa[mr][method]["MAE"][0] for mr in rates1]
        stds = [nasa[mr][method]["MAE"][1] for mr in rates1]
        ax1.bar(x1 + off, means, width, yerr=stds, label=method, color=colors[method], capsize=3, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([f"{int(r*100)}%" for r in rates1])
    ax1.set_ylabel(ylabel1)
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        fontsize=9,
        borderaxespad=0.2,
        handlelength=1.6,
        columnspacing=1.4,
    )
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.set_axisbelow(True)
    set_subtitle(ax1, "NASA Battery")
    format_axes_line(ax1, "Missing rate", ylabel1)
    tops1 = []
    for mr in rates1:
        for method in methods:
            m = nasa[mr][method]["MAE"][0]
            s = nasa[mr][method]["MAE"][1]
            tops1.append(float(m) + float(s))
    if tops1:
        ax1.set_ylim(0, max(tops1) * 1.45)
    # 右：NASA IGBT（MAPE）
    rates2 = sorted(second.keys())
    x2 = np.arange(len(rates2))
    for i, method in enumerate(methods):
        off = (i - 1) * width
        means = [second[mr][method][metric2][0] for mr in rates2]
        stds = [second[mr][method][metric2][1] for mr in rates2]
        ax2.bar(x2 + off, means, width, yerr=stds, label=method, color=colors[method], capsize=3, edgecolor="white", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([f"{int(r*100)}%" for r in rates2])
    ax2.set_ylabel(ylabel2)
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=3,
        frameon=True,
        fontsize=9,
        borderaxespad=0.2,
        handlelength=1.6,
        columnspacing=1.4,
    )
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_axisbelow(True)
    set_subtitle(ax2, second_title)
    format_axes_line(ax2, "Missing rate", ylabel2)
    tops2 = []
    for mr in rates2:
        for method in methods:
            m = second[mr][method][metric2][0]
            s = second[mr][method][metric2][1]
            tops2.append(float(m) + float(s))
    if tops2:
        ax2.set_ylim(0, max(tops2) * 1.45)
    # Remove top caption and reserve larger headroom (~4 character heights).
    fig.subplots_adjust(left=0.07, right=0.99, top=0.78, bottom=0.22, wspace=0.18)
    fig.savefig(FIG_DIR / out_name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG_DIR / out_name)


# ---------- Fig 7: Downstream tasks (DDIE03: RUL only; else RUL + load forecast) ----------
def fig7_downstream_tasks():
    setup_style()
    if USE_DDIE03:
        rul_path = RESULTS_DIR_DDIE03 / "downstream_rul_DDIE03.json"
        load_path = None
    else:
        rul_path = RESULTS_DIR / "downstream_rul.json"
        load_path = RESULTS_DIR / "downstream_load_forecast.json"
    has_rul = rul_path.exists()
    has_load = load_path and load_path.exists()
    if not has_rul and not has_load:
        print("Skip fig7: no downstream result files")
        return
    nplots = (1 if has_rul else 0) + (1 if has_load else 0)
    fig, axes = plt.subplots(1, max(nplots, 1), figsize=(4.5 * nplots, 4.5))
    if nplots == 1:
        axes = [axes]
    idx = 0
    if has_rul:
        with open(rul_path, "r", encoding="utf-8") as f:
            rul = json.load(f)
        m = rul.get("result", {}).get("metrics", {})
        if m:
            ax = axes[idx]
            phm = m.get("PHM_Score", 0)
            ax.bar(["RMSE", "|PHM Score|"], [m.get("RMSE", 0), abs(phm)], color=[COLOR_MAIN_BOX, PALETTE_EXTRA], edgecolor="white", linewidth=0.8)
            set_subtitle(ax, "Downstream RUL (NASA Battery, DDIE03)" if USE_DDIE03 else "Downstream RUL (NASA, single run)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_axisbelow(True)
            idx += 1
    if has_load:
        with open(load_path, "r", encoding="utf-8") as f:
            load = json.load(f)
        m = load.get("result", {}).get("metrics", {})
        if m:
            ax = axes[idx]
            ax.bar(["RMSE", "MAE"], [m.get("RMSE", 0), m.get("MAE", 0)], color=[PALETTE_KNN, PALETTE_LI], edgecolor="white", linewidth=0.8)
            set_subtitle(ax, "Downstream load forecast (optional)")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3, axis="y")
            ax.set_axisbelow(True)
    fig.suptitle("Downstream task performance", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_downstream_tasks.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG_DIR / "fig7_downstream_tasks.png")


# ---------- Fig 3 / Fig 8: NASA Battery LOOCV original scale (DDIE03) ----------
# 组稿09：正文 Figure 3 对应文件名 fig3_nasa_loocv_original_scale.png（与 fig8 为同一图）。
def fig8_nasa_loocv_original_scale():
    setup_style()
    if USE_DDIE03:
        agg = load_loocv_aggregate_ddie03()
        if not agg:
            print("Skip fig8: no DDIE03 LOOCV aggregate")
            return
        rates_show = [0.1, 0.5, 0.9]
        li = [agg[mr]["LI"]["MAE"][0] for mr in rates_show]
        li_std = [agg[mr]["LI"]["MAE"][1] for mr in rates_show]
        knn = [agg[mr]["KNN"]["MAE"][0] for mr in rates_show]
        knn_std = [agg[mr]["KNN"]["MAE"][1] for mr in rates_show]
        ddie = [agg[mr]["DDI-E"]["MAE"][0] for mr in rates_show]
        ddie_std = [agg[mr]["DDI-E"]["MAE"][1] for mr in rates_show]
    else:
        li, li_std = [0.0067, 0.0088, 0.0196], [0.0021, 0.0025, 0.0053]
        knn, knn_std = [0.0138, 0.0211, 0.0518], [0.0034, 0.0037, 0.0090]
        ddie, ddie_std = [0.3888, 0.3928, 0.3937], [0.0495, 0.0507, 0.0531]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(3)
    w = 0.24
    ax.bar(x - w, li, w, yerr=li_std, label="LI", color=PALETTE_LI, capsize=3, edgecolor="white", linewidth=0.8)
    ax.bar(x, knn, w, yerr=knn_std, label="KNN", color=PALETTE_KNN, capsize=3, edgecolor="white", linewidth=0.8)
    ax.bar(x + w, ddie, w, yerr=ddie_std, label="DDI-E", color=PALETTE_DDIE, capsize=3, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["10%", "50%", "90%"])
    ax.set_ylabel("MAE (Ah)")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    def _loocv_val_txt(v: float) -> str:
        if abs(v) < 0.02:
            return f"{v:.5f}"
        if abs(v) < 0.1:
            return f"{v:.4f}"
        return f"{v:.3f}"

    tops = []
    for i in range(3):
        tops.extend([li[i] + li_std[i], knn[i] + knn_std[i], ddie[i] + ddie_std[i]])
    ypad = max(tops) * 0.018
    for i in range(3):
        ax.text(x[i] - w, li[i] + li_std[i] + ypad, _loocv_val_txt(li[i]), ha="center", va="bottom", fontsize=7, color="#333")
        ax.text(x[i], knn[i] + knn_std[i] + ypad, _loocv_val_txt(knn[i]), ha="center", va="bottom", fontsize=7, color="#333")
        ax.text(x[i] + w, ddie[i] + ddie_std[i] + ypad, _loocv_val_txt(ddie[i]), ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_ylim(0, max(tops) + ypad * 4.5)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=1,
        labelspacing=0.55,
        handletextpad=0.45,
        frameon=True,
        fontsize=9,
        framealpha=0.95,
        edgecolor="#ccc",
    )
    fig.subplots_adjust(left=0.10, right=0.99, top=0.82, bottom=0.13)
    for fname in ("fig8_nasa_loocv_original_scale.png", "fig3_nasa_loocv_original_scale.png"):
        fig.savefig(FIG_DIR / fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG_DIR / "fig8_nasa_loocv_original_scale.png", "&", FIG_DIR / "fig3_nasa_loocv_original_scale.png")


# ---------- Fig 9: Imputation example (v1.1 line style + subtitle) ----------
def _linear_interp_1d(arr, mask):
    out = arr.copy()
    ok = mask > 0.5
    if ok.all() or not ok.any():
        return out
    idx = np.arange(len(arr))
    out[~ok] = np.interp(idx[~ok], idx[ok], arr[ok])
    return out


def _knn_impute_1d_simple(arr, mask, k=5):
    out = arr.copy()
    ok = mask > 0.5
    obs_idx = np.where(ok)[0]
    mis_idx = np.where(~ok)[0]
    if len(obs_idx) == 0 or len(mis_idx) == 0:
        return out
    for i in mis_idx:
        d = np.abs(obs_idx - i)
        nn = obs_idx[np.argsort(d)[:k]]
        out[i] = np.mean(arr[nn])
    return out


def fig9_imputation_example():
    setup_style()
    L = 128
    np.random.seed(42)
    t = np.linspace(0, 1, L)
    orig = 1.0 - 0.3 * t ** 0.8 + 0.02 * np.sin(4 * np.pi * t) + 0.01 * np.random.randn(L).cumsum()
    orig = np.clip(orig, 0.5, 1.0)
    mask = (np.random.RandomState(42).rand(L) > 0.3).astype(np.float32)
    li_1d = _linear_interp_1d(orig, mask)
    knn_1d = _knn_impute_1d_simple(orig, mask, k=5)
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.5), sharex=True)
    time_axis = np.arange(L)

    axes[0].plot(time_axis, orig, color=COLOR_GT, linewidth=1.5, label="Original")
    format_axes_line(axes[0], ylabel="Value")
    axes[0].xaxis.set_label_coords(0.78, -0.10)
    axes[0].legend(loc="upper right", frameon=True, fontsize=9)
    set_subtitle(axes[0], "(a) Original series (synthetic capacity-like)")

    obs_plot = orig.copy()
    obs_plot[mask < 0.5] = np.nan
    axes[1].plot(time_axis, obs_plot, color=COLOR_MISSING, linewidth=1.5, label="Observed")
    axes[1].scatter(time_axis[mask < 0.5], orig[mask < 0.5], s=12, c=COLOR_IMPUTED, alpha=0.6, label="Missing", zorder=3)
    format_axes_line(axes[1], ylabel="Value")
    axes[1].xaxis.set_label_coords(0.78, -0.10)
    axes[1].legend(loc="upper right", frameon=True, fontsize=9)
    set_subtitle(axes[1], "(b) With 30% MCAR missing")

    axes[2].plot(time_axis, li_1d, color=PALETTE_LI, linewidth=1.5, linestyle="--", label="LI")
    axes[2].plot(time_axis, orig, color=COLOR_GT, linewidth=1.0, alpha=0.5, label="Original")
    format_axes_line(axes[2], ylabel="Value")
    axes[2].xaxis.set_label_coords(0.78, -0.10)
    axes[2].legend(loc="upper right", frameon=True, fontsize=9)
    set_subtitle(axes[2], "(c) LI imputation")

    axes[3].plot(time_axis, knn_1d, color=PALETTE_KNN, linewidth=1.5, linestyle="--", label="KNN")
    axes[3].plot(time_axis, orig, color=COLOR_GT, linewidth=1.0, alpha=0.5, label="Original")
    format_axes_line(axes[3], "Time step", "Value")
    axes[3].xaxis.set_label_coords(0.78, -0.10)
    axes[3].legend(loc="upper right", frameon=True, fontsize=9)
    set_subtitle(axes[3], "(d) KNN imputation")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig9_imputation_example.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", FIG_DIR / "fig9_imputation_example.png")


def main():
    import sys
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    # 预览分块： python generate_paper_figures.py fig1_block B1  或  fig1_block all
    if len(sys.argv) >= 3 and sys.argv[1] == "fig1_block":
        arg = sys.argv[2].upper()
        if arg == "ALL":
            for bid in FIG1_BLOCK_DRAWERS:
                fig1_framework_block_preview(bid)
            print("All block previews saved under:", FIG_DIR)
        else:
            bid = arg if arg.startswith("B") else "B" + str(arg)
            fig1_framework_block_preview(bid)
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "fig1_redesign":
        fig1_framework_redesign()
        return
    if len(sys.argv) >= 2 and sys.argv[1] == "fig1_redesign_blocks":
        fig1_framework_redesign_blocks()
        return
    fig1_framework()
    fig2_nasa_imputation()
    fig3_nasa_igbt_imputation()
    fig4_method_selection()
    fig5_applicability_boundary()
    fig6_combined_mae()
    fig7_downstream_tasks()
    fig8_nasa_loocv_original_scale()
    fig9_imputation_example()
    print("All figures saved under:", FIG_DIR)


if __name__ == "__main__":
    main()
