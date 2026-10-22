# -*- coding: utf-8 -*-
"""V5 → V5.1 的「缺陷尺度」账本图：把 ROI 裁剪为什么有效的几何原因画出来。

这张图的每个数都能独立复算（见 core/roi.py 与 docs/preprocess_analysis_20261021.md）：

    有效边长 = 原生等效边长 × letterbox 缩放率
    缩放率   = min(画布边长 / 图高, 画布边长 / 图宽)

原生等效边长（高分辨率组 2129x1198）：
    极片损伤 37.7px / 暗斑 25.8px / 亮斑 16.8px

用法：
    ./runtime/python.exe scripts/plot_scale_story.py
产出：
    reports/train/defect_scale.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 原生等效边长（高分辨率组 2129x1198，实测）
NATIVE = {"损伤": 37.7, "暗斑": 25.8, "亮斑": 16.8}
CLS_COLOR = {"损伤": "#c2410c", "暗斑": "#1d4ed8", "亮斑": "#b45309"}

HI = (2129, 1198)          # 高分辨率原图 W,H（占 68%）
ROI_W, ROI_H = 647, 991    # ROI 裁剪后尺寸


def scale_of(canvas, img_w, img_h):
    return min(canvas / float(img_h), canvas / float(img_w))


def build_configs():
    """(名称, 缩放率, 画布算力MP, 说明)"""
    cfgs = []
    # V5：mosaic 训练时每个子图只占 imgsz/2
    cfgs.append(("V5 训练\nmosaic 格 960", scale_of(960, *HI), 3.69,
                 "mosaic 把 4 张拼进一个画布，每张只有 960"))
    cfgs.append(("V5 推理\n全图 1920", scale_of(1920, *HI), 3.69,
                "16:9 图放进方形画布，上下补黑边"))
    cfgs.append(("V5.1-A\n全图 2112", scale_of(2112, *HI), 4.46,
                 "关掉 mosaic 单图训练，但仍是全图"))
    cfgs.append(("ROI@1024", scale_of(1024, ROI_W, ROI_H), 1.05, "画布太小"))
    cfgs.append(("ROI@1216", scale_of(1216, ROI_W, ROI_H), 1.48, "折中档"))
    cfgs.append(("ROI@1536\n★本次", scale_of(1536, ROI_W, ROI_H), 2.36, "本次采用"))
    cfgs.append(("ROI@2112", scale_of(2112, ROI_W, ROI_H), 4.46, "过算力"))
    return cfgs


def main():
    cfgs = build_configs()
    names = [c[0] for c in cfgs]
    scales = np.array([c[1] for c in cfgs])
    mp = np.array([c[2] for c in cfgs])

    outdir = os.path.join(ROOT, "reports", "train")
    os.makedirs(outdir, exist_ok=True)

    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])

    # ---- 左：三类缺陷的等效边长 ----
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(cfgs))
    w = 0.26
    for i, (cl, nat) in enumerate(NATIVE.items()):
        vals = nat * scales
        bars = ax.bar(x + (i - 1) * w, vals, width=w * 0.9, color=CLS_COLOR[cl],
                      label="%s（原生 %.1fpx）" % (cl, nat), edgecolor="white", linewidth=0.5,
                      zorder=3)
        ax.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=1)
    ax.set_xticks(x, names, fontsize=7.2)
    ax.set_ylabel("等效边长 (px) @ 送进网络的图上")
    ax.set_title("缺陷在网络上「有多大」——这是小目标检测的第一性变量", fontsize=11)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=8, loc="upper left")
    ax.axvspan(5 - 0.45, 5 + 0.45, color="#16a34a", alpha=0.10, zorder=0)
    ax.text(5, ax.get_ylim()[1] * 0.97, "本次方案", ha="center", va="top",
            fontsize=8, color="#15803d", fontweight="bold")
    # 两组之间加分隔线，说明「前 3 个是全图口径，后 4 个是 ROI 口径」
    ax.axvline(2.5, color="#64748b", ls=":", lw=1.0, alpha=0.7)
    ax.text(1.0, ax.get_ylim()[1] * 0.90, "全图口径", ha="center", fontsize=8, color="#64748b")
    ax.text(4.5, ax.get_ylim()[1] * 0.90, "ROI 口径", ha="center", fontsize=8, color="#64748b")

    # ---- 右：有效缺陷尺度 vs 画布算力（帕累托） ----
    ax2 = fig.add_subplot(gs[0, 1])
    light = NATIVE["亮斑"] * scales
    colors = ["#94a3b8", "#94a3b8", "#dc2626", "#f59e0b", "#f59e0b", "#16a34a", "#f59e0b"]
    for i in range(len(cfgs)):
        ax2.scatter(mp[i], light[i], s=110, color=colors[i], edgecolors="white",
                    linewidths=1.0, zorder=5)
        ax2.annotate(names[i].replace("\n", " "), (mp[i], light[i]), fontsize=7,
                     xytext=(7, -2), textcoords="offset points")
    # 帕累托前沿：右下（低算力高尺度）
    order = np.argsort(mp)
    front = []
    best = 0.0
    for i in order:
        if light[i] > best:
            best = light[i]
            front.append(i)
    ax2.step(mp[front], light[front], where="post", ls="--", lw=1.2, color="#16a34a",
             alpha=0.7, label="帕累托前沿（同等算力下最优）")
    ax2.axhline(light[1], ls=":", lw=1.0, color="#94a3b8")
    ax2.text(mp.min() + 0.03, light[1] + 0.7, "V5 推理基线 %.1fpx" % light[1],
             ha="left", fontsize=8, color="#475569")
    ax2.set_xlabel("画布算力 (MP，≈推理耗时)")
    ax2.set_ylabel("亮斑等效边长 (px)")
    ax2.set_ylim(4, light.max() * 1.12)
    ax2.set_xlim(mp.min() - 0.15, mp.max() + 0.35)
    ax2.set_title("算力 vs 小目标像素：ROI@1536 用 64% 算力换 1.7 倍像素", fontsize=11)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")

    fig.suptitle("V5 → V5.1：把「看不见的亮斑」变成「看得见的亮斑」", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(outdir, "defect_scale.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(p, ROOT))
    for n, s, m, _ in cfgs:
        print("%-24s 缩放 %.3f | 亮斑 %.1fpx | %.2f MP"
              % (n.replace("\n", " "), s, NATIVE["亮斑"] * s, m))


if __name__ == "__main__":
    sys.exit(main())
