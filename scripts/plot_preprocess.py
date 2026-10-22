# -*- coding: utf-8 -*-
"""
生成预处理分析的图表（只读，输出到 reports/preprocess/）
用法：runtime\\python.exe scripts/plot_preprocess.py
"""
import os
import sys
import json
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import dataset_split as ds  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CLS = {0: "damage", 1: "dark", 2: "light"}
CN = {"damage": "极片损伤", "dark": "暗斑", "light": "亮斑"}
COLOR = {"damage": "#c2410c", "dark": "#1d4ed8", "light": "#b45309"}
OUT = os.path.join(ROOT, "reports", "preprocess")


def collect_sides():
    """按类 + 分辨率组收集等效边长（原生 / @1920方形 / @1920+mosaic）"""
    rows = ds.load_manifest("reports/splits/v5_split_manifest.csv")
    data = {c: {"hi": {"native": [], "at1920": [], "mos": []},
                "lo": {"native": [], "at1920": [], "mos": []}} for c in CN}
    for r in rows:
        if not r["label_json"]:
            continue
        W, H = (int(x) for x in r["size"].split("x"))
        g = "hi" if W > 1500 else "lo"
        try:
            d = json.load(open(r["label_json"], encoding="utf-8"))
        except Exception:
            continue
        s = min(1920.0 / W, 1920.0 / H)
        for sp in d.get("shapes", []):
            try:
                c = int(sp["label"])
            except (ValueError, KeyError):
                continue
            n = CLS.get(c)
            pts = sp.get("points") or []
            if not n or len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            side = (max(1e-6, (max(xs) - min(xs)) * (max(ys) - min(ys)))) ** 0.5
            data[n][g]["native"].append(side)
            data[n][g]["at1920"].append(side * s)
            data[n][g]["mos"].append(side * s / 2.0)
    return data


def main():
    os.makedirs(OUT, exist_ok=True)
    data = collect_sides()

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
    fig.suptitle("电芯 CT 数据集 · 预处理体检（全部为实测）", fontsize=14, y=0.985)

    # ---- (1) 画布利用率 ----
    ax = axes[0][0]
    cfgs = ["1920×1920\n(现状)", "1920×1088\n(矩形)", "2112×1184\n(原生1:1)"]
    hi_fill = [56.3, 99.3, 99.6]
    lo_fill = [55.3, 97.6, 98.7]
    x = np.arange(len(cfgs))
    ax.bar(x - 0.19, hi_fill, 0.36, label="2129×1198 (3118 张)", color="#2563eb")
    ax.bar(x + 0.19, lo_fill, 0.36, label="1052×582 (1474 张)", color="#93c5fd")
    for i, (a, b) in enumerate(zip(hi_fill, lo_fill)):
        ax.text(i - 0.19, a + 1.5, "%.0f%%" % a, ha="center", fontsize=9)
        ax.text(i + 0.19, b + 1.5, "%.0f%%" % b, ha="center", fontsize=9)
    ax.set_xticks(x, cfgs, fontsize=9)
    ax.set_ylim(0, 118)
    ax.set_ylabel("画布有效内容占比 (%)")
    ax.set_title("① 方形画布浪费 44% 算力（图全部 16:9）", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    # ---- (2) 缺陷等效边长分布 ----
    ax = axes[0][1]
    bins = np.arange(0, 90, 2)
    for c in ("damage", "dark", "light"):
        v = data[c]["hi"]["at1920"]
        ax.hist(v, bins=bins, alpha=0.55, label="%s (p50=%.1fpx)" % (CN[c], np.median(v)),
                color=COLOR[c])
    ax.axvline(8, color="#dc2626", ls="--", lw=1.4)
    ax.text(8.6, ax.get_ylim()[1] * 0.92, "stride 8 (1 个 feature cell)", color="#dc2626", fontsize=8)
    ax.set_xlabel("缺陷等效边长 @1920 (px)")
    ax.set_ylabel("框数量")
    ax.set_title("② 三类缺陷尺寸差 2 倍以上，亮斑最小", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    # ---- (3) mosaic 后的尺寸坍塌 ----
    ax = axes[1][0]
    labels, below8, below16 = [], [], []
    for c in ("damage", "dark", "light"):
        v = np.array(data[c]["hi"]["at1920"] + data[c]["lo"]["at1920"])
        m = v / 2.0
        labels.append(CN[c])
        below8.append(100 * np.mean(m < 8))
        below16.append(100 * np.mean(m < 16))
    x = np.arange(len(labels))
    ax.bar(x - 0.19, below8, 0.36, label="< 8 px（P3 一个格子）", color="#dc2626")
    ax.bar(x + 0.19, below16, 0.36, label="< 16 px（P3 两个格子）", color="#f59e0b")
    for i, (a, b) in enumerate(zip(below8, below16)):
        ax.text(i - 0.19, a + 1.5, "%.1f%%" % a, ha="center", fontsize=9)
        ax.text(i + 0.19, b + 1.5, "%.1f%%" % b, ha="center", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 112)
    ax.set_ylabel("占该类框的比例 (%)")
    ax.set_title("③ mosaic 把一半亮斑压到 8px 以下", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # ---- (4) SNR 三档 ----
    ax = axes[1][1]
    ex = json.load(open(os.path.join(OUT, "preprocess_analysis_extra.json"), encoding="utf-8"))["snr"]
    names = ["damage", "dark", "light"]
    orig = [ex[n]["snr_orig_p50"] for n in names]
    lb = [ex[n]["snr_letterbox_p50"] for n in names]
    mos = [ex[n]["snr_mosaic_p50"] for n in names]
    x = np.arange(len(names))
    w = 0.26
    ax.bar(x - w, orig, w, label="原图", color="#059669")
    ax.bar(x, lb, w, label="@1920", color="#2563eb")
    ax.bar(x + w, mos, w, label="@1920+mosaic", color="#dc2626")
    for i in range(len(names)):
        for off, val in ((-w, orig[i]), (0, lb[i]), (w, mos[i])):
            ax.text(i + off, val + 0.35, "%.1f" % val, ha="center", fontsize=8)
    ax.axhline(1.0, color="#6b7280", ls=":", lw=1)
    ax.set_xticks(x, [CN[n] for n in names])
    ax.set_ylabel("可检出指数 SNR (p50)")
    ax.set_title("④ mosaic 让 SNR 近乎腰斩（对比度仅 3 灰度级）", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout(rect=[0, 0, 1, 0.975])
    p = os.path.join(OUT, "preprocess_diagnosis.png")
    fig.savefig(p, dpi=160)
    print("写出", p)

    # 第二张：单类框尺寸箱线图（原生 vs 两种画布）
    fig2, ax = plt.subplots(figsize=(9.2, 4.6))
    pos, ticks, vals, cols = [], [], [], []
    i = 0
    for c in ("damage", "dark", "light"):
        for gk, gn in (("native", "原生 2129"), ("at1920", "@1920方形"), ("mos", "@1920+mosaic")):
            pos.append(i)
            vals.append(data[c]["hi"][gk])
            ticks.append("%s\n%s" % (CN[c], gn))
            cols.append(COLOR[c])
            i += 1
        i += 0.6
    bp = ax.boxplot(vals, positions=pos, widths=0.62, patch_artist=True, showfliers=False)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col)
        patch.set_alpha(0.55)
    ax.axhline(8, color="#dc2626", ls="--", lw=1.2)
    ax.axhline(16, color="#f59e0b", ls=":", lw=1.2)
    ax.set_xticks(pos, ticks, fontsize=8)
    ax.set_ylabel("缺陷等效边长 (px)")
    ax.set_title("三类缺陷：原生分辨率 / 1920 方形 / mosaic 后再缩一半", fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    fig2.tight_layout()
    p2 = os.path.join(OUT, "preprocess_boxsize.png")
    fig2.savefig(p2, dpi=160)
    print("写出", p2)


if __name__ == "__main__":
    main()
