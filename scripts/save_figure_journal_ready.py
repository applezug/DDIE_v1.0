#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将任意论文用 PNG 调整为期刊常见要求后另存（不覆盖源文件）：
  - 宽 ≥ 1100 px（不足则左右白边）
  - 300 dpi 写入 PNG
  - 若宽 < 2400 px，先 2× LANCZOS 放大再保存（小图投稿更清晰；已是高清则不再放大）

输出文件名：{源文件名去掉扩展名}_submit_{NNN}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOURNAL_DPI = 300
MIN_WIDTH_PX = 1100
UPSCALE_THRESHOLD_W = 2400  # 低于此宽度则 2×


def _to_rgb(im):
    from PIL import Image

    if im.mode in ("RGBA", "P"):
        bg = Image.new("RGB", im.size, (255, 255, 255))
        if im.mode == "RGBA":
            bg.paste(im, mask=im.split()[3])
        else:
            bg.paste(im)
        return bg
    return im.convert("RGB")


def journal_ready_png(src: Path) -> Path:
    from PIL import Image

    src = Path(src)
    if not src.is_file():
        raise FileNotFoundError(src)

    im = _to_rgb(Image.open(src))
    w, h = im.size
    if w < UPSCALE_THRESHOLD_W:
        im = im.resize((w * 2, h * 2), Image.Resampling.LANCZOS)
        w, h = im.size

    if w < MIN_WIDTH_PX:
        out_im = Image.new("RGB", (MIN_WIDTH_PX, h), (255, 255, 255))
        out_im.paste(im, ((MIN_WIDTH_PX - w) // 2, 0))
        im = out_im

    stem = src.stem
    parent = src.parent
    n = 1
    while True:
        dst = parent / f"{stem}_submit_{n:03d}.png"
        if not dst.exists():
            break
        n += 1

    im.save(dst, "PNG", dpi=(JOURNAL_DPI, JOURNAL_DPI), compress_level=6)
    return dst


def main() -> None:
    # FIG1-04 定稿图（见 docs/Figure1_工作记录与代码对照.md）
    default = ROOT / "figures" / "fig1_ddie_framework_redesign.png"
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    out = journal_ready_png(src)
    from PIL import Image

    print("Saved:", out)
    i = Image.open(out)
    print("Size:", i.size, "dpi", i.info.get("dpi"))


if __name__ == "__main__":
    main()
