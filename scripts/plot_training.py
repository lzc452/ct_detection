# -*- coding: utf-8 -*-
"""训练过程可视化与摘要：从 results.csv 生成曲线图 + 结构化 summary.json。

用法（可一次对比多个 run）：

    ./runtime/python.exe scripts/plot_training.py ^
        --run V51_ROI_1536=runs/detect/CT_pro/V51_ROI_1536 ^
        --run V51_PREPROC_1021=runs/detect/CT_pro/V51_PREPROC_1021 ^
        --run V5_GROUPED_1021_80ep=runs/detect/CT_pro/V5_GROUPED_1021_80ep ^
        --outdir reports/train

产出：
    reports/train/curves_metrics.png   指标曲线（P/R/mAP50/mAP50-95）
    reports/train/curves_loss.png      损失曲线（train vs val，看是否过拟合）
    reports/train/curves_lr.png        学习率曲线
    reports/train/curves_overfit.png   泛化间隙（val-box − train-box）
    reports/train/summary.json         每个 run 的 best epoch / 最优指标 / 过拟合间隙
"""
import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 一条曲线 = (csv 列名, 图例名, 颜色)
METRIC_COLS = [
    ("metrics/mAP50-95(B)", "mAP50-95", "#b91c1c"),
    ("metrics/mAP50(B)", "mAP50", "#dc2626"),
    ("metrics/precision(B)", "Precision", "#1d4ed8"),
    ("metrics/recall(B)", "Recall", "#0891b2"),
]
LOSS_COLS = [
    ("train/box_loss", "val/box_loss", "box", "#b91c1c"),
    ("train/cls_loss", "val/cls_loss", "cls", "#1d4ed8"),
    ("train/dfl_loss", "val/dfl_loss", "dfl", "#0891b2"),
]
PALETTE = ["#b91c1c", "#1d4ed8", "#0891b2", "#b45309", "#059669", "#7c3aed", "#db2777"]


def load_results(run_dir):
    p = os.path.join(run_dir, "results.csv")
    if not os.path.isfile(p):
        raise SystemExit("找不到 %s" % p)
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()})
    if not rows:
        raise SystemExit("%s 为空" % p)
    data = {}
    for k in rows[0].keys():
        vals = []
        for r in rows:
            try:
                vals.append(float(r[k]))
            except (TypeError, ValueError):
                vals.append(np.nan)
        data[k] = np.array(vals)
    return data


def summarize(name, run_dir, d):
    ep = d["epoch"]
    out = {
        "name": name,
        "run_dir": os.path.relpath(run_dir, ROOT).replace("\\", "/"),
        "epochs_done": int(ep[-1]) if len(ep) else 0,
        "wall_time_s": float(d["time"][-1]) if "time" in d and len(d["time"]) else None,
        "sec_per_epoch": None,
        "best": {},
        "final": {},
        "overfit": {},
    }
    if out["epochs_done"] > 1 and out["wall_time_s"]:
        out["sec_per_epoch"] = round(out["wall_time_s"] / out["epochs_done"], 1)

    key = "metrics/mAP50-95(B)"
    if key in d:
        i = int(np.nanargmax(d[key]))
        out["best"] = {
            "epoch": int(ep[i]),
            "mAP50-95": round(float(d[key][i]), 4),
            "mAP50": round(float(d["metrics/mAP50(B)"][i]), 4),
            "precision": round(float(d["metrics/precision(B)"][i]), 4),
            "recall": round(float(d["metrics/recall(B)"][i]), 4),
        }
        out["final"] = {
            "epoch": int(ep[-1]),
            "mAP50-95": round(float(d[key][-1]), 4),
            "mAP50": round(float(d["metrics/mAP50(B)"][-1]), 4),
        }
        # 末 10 轮（或更少）的均值/标准差，用于判断是否还在涨
        k = min(10, len(ep))
        tail = d[key][-k:]
        out["tail"] = {
            "n": int(k),
            "mean_mAP50_95": round(float(np.nanmean(tail)), 4),
            "std_mAP50_95": round(float(np.nanstd(tail)), 4),
            "max_mAP50_95": round(float(np.nanmax(tail)), 4),
        }

    # 过拟合间隙：同一 epoch 上 val_loss 与 train_loss 的比值和差值
    if "val/box_loss" in d and "train/box_loss" in d:
        gap = d["val/box_loss"] - d["train/box_loss"]
        ratio = d["val/box_loss"] / np.maximum(d["train/box_loss"], 1e-9)
        out["overfit"] = {
            "box_gap_final": round(float(gap[-1]), 4),
            "box_gap_min": round(float(np.nanmin(gap)), 4),
            "box_ratio_final": round(float(ratio[-1]), 3),
            "box_ratio_min": round(float(np.nanmin(ratio)), 3),
        }
    return out


def _style(ax, title, ylabel, xlabel="epoch"):
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)


def plot_metrics(runs, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("训练指标曲线（同一考卷：分组切分 + 冻结 test）", fontsize=13, y=0.98)
    for j, (col, label, _c) in enumerate(METRIC_COLS):
        ax = axes[j // 2][j % 2]
        for i, (name, d) in enumerate(runs):
            if col not in d:
                continue
            ax.plot(d["epoch"], d[col], lw=1.6, color=PALETTE[i % len(PALETTE)],
                    label="%s (best %.4f)" % (name, float(np.nanmax(d[col]))))
            k = int(np.nanargmax(d[col]))
            ax.scatter([d["epoch"][k]], [d[col][k]], s=26, color=PALETTE[i % len(PALETTE)],
                       zorder=5, edgecolors="white", linewidths=0.6)
        _style(ax, label, label)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out_path, ROOT))


def plot_loss(runs, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    fig.suptitle("损失曲线 · 实线=train / 虚线=val（间隙大=过拟合）", fontsize=12, y=1.0)
    for j, (tr, va, label, _c) in enumerate(LOSS_COLS):
        ax = axes[j]
        for i, (name, d) in enumerate(runs):
            if tr not in d:
                continue
            c = PALETTE[i % len(PALETTE)]
            ax.plot(d["epoch"], d[tr], lw=1.4, color=c, label="%s train" % name)
            if va in d:
                ax.plot(d["epoch"], d[va], lw=1.4, ls="--", color=c, label="%s val" % name)
        _style(ax, "%s_loss" % label, label)
        ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out_path, ROOT))


def plot_lr(runs, out_path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for i, (name, d) in enumerate(runs):
        col = "lr/pg0" if "lr/pg0" in d else None
        if not col:
            continue
        ax.plot(d["epoch"], d[col], lw=1.5, color=PALETTE[i % len(PALETTE)], label=name)
    _style(ax, "学习率调度（cos_lr）", "lr")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out_path, ROOT))


def plot_overfit(runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for i, (name, d) in enumerate(runs):
        if "val/box_loss" not in d:
            continue
        c = PALETTE[i % len(PALETTE)]
        gap = d["val/box_loss"] - d["train/box_loss"]
        ratio = d["val/box_loss"] / np.maximum(d["train/box_loss"], 1e-9)
        axes[0].plot(d["epoch"], gap, lw=1.5, color=c, label=name)
        axes[1].plot(d["epoch"], ratio, lw=1.5, color=c, label=name)
    _style(axes[0], "泛化间隙 val_box − train_box（越小越好）", "abs gap")
    _style(axes[1], "泛化比 val_box / train_box（→1 越好）", "ratio")
    axes[1].axhline(1.0, color="#64748b", lw=0.9, ls=":")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out_path, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--outdir", default="reports/train")
    args = ap.parse_args()

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)

    runs, summaries = [], []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit("--run 需写成 NAME=DIR，收到 %r" % spec)
        name, d = spec.split("=", 1)
        rd = d if os.path.isabs(d) else os.path.join(ROOT, d)
        data = load_results(rd)
        runs.append((name, data))
        s = summarize(name, rd, data)
        summaries.append(s)
        print("[%s] %d ep / %s s/ep · best mAP50-95 %.4f @ep%d · final %.4f"
              % (name, s["epochs_done"], s["sec_per_epoch"],
                 s["best"].get("mAP50-95", float("nan")), s["best"].get("epoch", -1),
                 s["final"].get("mAP50-95", float("nan"))))

    plot_metrics(runs, os.path.join(outdir, "curves_metrics.png"))
    plot_loss(runs, os.path.join(outdir, "curves_loss.png"))
    plot_lr(runs, os.path.join(outdir, "curves_lr.png"))
    plot_overfit(runs, os.path.join(outdir, "curves_overfit.png"))

    sp = os.path.join(outdir, "summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump({"runs": summaries}, f, ensure_ascii=False, indent=2)
    print("  -> %s" % os.path.relpath(sp, ROOT))


if __name__ == "__main__":
    sys.exit(main())
