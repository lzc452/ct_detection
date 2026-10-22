# -*- coding: utf-8 -*-
"""评测结果可视化：把 reports/eval/*.json 画成可比对的图。

关键前提：**不同推理口径（全图 vs ROI、不同画布尺寸）的 mAP 绝对值不可直接比较**。
本脚本会把口径写进图例与柱状图的分组，并在图上明确标注，避免误读。

用法：

    # 自动收录 reports/eval 下所有报告
    ./runtime/python.exe scripts/plot_eval.py

    # 只画指定几个
    ./runtime/python.exe scripts/plot_eval.py ^
        --report V5_80ep=reports/eval/V5_80ep_v5test.json ^
        --report V51_ROI=reports/eval/V51_ROI_1536_v5test.json

产出（reports/eval/figs/）：
    eval_map.png            mAP50 / mAP50-95（按口径分组）
    eval_perclass_ap.png    三类 AP50 与 AP50-95
    eval_workpoint.png      工作点 P/R 散点（气泡=FP千张）
    eval_fp_miss.png        FP千张 与 组级漏检率
    eval_table.md           自适应汇总表
"""
import argparse
import glob
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
EVAL_DIR = os.path.join(ROOT, "reports", "eval")
FIG_DIR = os.path.join(EVAL_DIR, "figs")

CLS = ["damage", "dark", "light"]
CN = {"damage": "极片损伤", "dark": "暗斑", "light": "亮斑"}
CLS_COLOR = {"damage": "#c2410c", "dark": "#1d4ed8", "light": "#b45309"}

# 口径 → (显示名, 颜色, 填充样式)
SPEC_STYLE = {}


def _fmt_sz(sz):
    """imgsz 可能是 int，也可能是 predict 口径的 [h, w]；统一成可读字符串。"""
    if isinstance(sz, (list, tuple)):
        if len(sz) == 2 and sz[0] == sz[1]:
            return str(sz[0])
        return "x".join(str(v) for v in sz)
    return str(sz)


def spec_key(d):
    """统一口径字符串。兼容 eval_report 报告（imgsz_predict/layer1）与 sweep 结果（imgsz）。"""
    if "_spec" in d:
        return d["_spec"]
    roi = bool(d.get("roi"))
    sz = d.get("imgsz_predict") or d.get("imgsz_layer1") or d.get("imgsz") or "?"
    return ("ROI@%s" if roi else "全图@%s") % _fmt_sz(sz)


def spec_of(d):
    roi = bool(d.get("roi"))
    return spec_key(d), roi


def load_reports(pairs):
    out = []
    for name, path in pairs:
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        if not os.path.isfile(path):
            print("[跳过] 不存在: %s" % path)
            continue
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if "layer1_coco" not in d:
            print("[跳过] 仅工作点报告不参加包含mAP的总图: %s" % path)
            continue
        spec, roi = spec_of(d)
        d["_name"] = name
        d["_spec"] = spec
        d["_roi"] = roi
        d["_path"] = os.path.relpath(path, ROOT).replace("\\", "/")
        out.append(d)
    return out


def discover():
    pairs = []
    for p in sorted(glob.glob(os.path.join(EVAL_DIR, "*.json"))):
        if os.path.basename(p).startswith(("seen_unseen", "sweep_")):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("schema", "").startswith("ct-eval-report") and "layer1_coco" in d:
            pairs.append((os.path.splitext(os.path.basename(p))[0].replace("_v5test", ""), p))
    return pairs


def assign_styles(reports, sweeps=()):
    specs = []
    for r in reports:
        if r["_spec"] not in specs:
            specs.append(r["_spec"])
    for _n, d in sweeps:
        s = spec_key(d)
        if s not in specs:
            specs.append(s)
    palette = ["#b91c1c", "#1d4ed8", "#b45309", "#0891b2", "#7c3aed", "#059669"]
    for i, s in enumerate(specs):
        SPEC_STYLE[s] = (palette[i % len(palette)], "//" if s.startswith("ROI") else "")
    return specs


def load_sweeps(pairs):
    out = []
    for name, path in pairs:
        if not os.path.isabs(path):
            path = os.path.join(ROOT, path)
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            out.append((name, json.load(f)))
    return out


def _order(reports, specs):
    # 按口径排序，同口径内按 mAP50-95 降序
    def k(r):
        return (specs.index(r["_spec"]), -(r["layer1_coco"]["mAP50_95"]))
    return sorted(reports, key=k)


def plot_map(reports, specs, out):
    rs = _order(reports, specs)
    n = len(rs)
    x = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.15 * n + 4), 4.6), sharey=False)
    for ax, (key, title) in zip(axes, [("mAP50", "mAP50"), ("mAP50_95", "mAP50-95")]):
        vals = [r["layer1_coco"][key] for r in rs]
        cols = [SPEC_STYLE[r["_spec"]][0] for r in rs]
        hatch = [SPEC_STYLE[r["_spec"]][1] for r in rs]
        bars = ax.bar(x, vals, color=cols, hatch=hatch, edgecolor="white", linewidth=0.8, width=0.66)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, "%.3f" % v,
                    ha="center", fontsize=8)
        ax.set_xticks(x, [r["_name"] for r in rs], rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
    # 口径图例
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SPEC_STYLE[s][0],
                             hatch=SPEC_STYLE[s][1], edgecolor="white") for s in specs]
    axes[0].legend(handles, specs, fontsize=8, title="推理口径", title_fontsize=8, loc="upper left")
    fig.suptitle("检测精度对照（冻结 test 676 张 · 工作点无关的 COCO 指标）\n"
                 "【注意】斜纹=ROI 口径；不同口径的绝对值不可直接比较，只能在同口径内排序",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out, ROOT))


def plot_perclass(reports, specs, out):
    rs = _order(reports, specs)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    w = 0.8 / max(1, len(rs))
    x = np.arange(len(CLS))
    for ax, (key, title) in zip(axes, [("ap50", "AP50（宽松定位下的召回上界）"),
                                       ("ap", "AP50-95（严苛定位，反映框回归质量）")]):
        for i, r in enumerate(rs):
            vals = [r["layer1_coco"]["per_class"][c][key] for c in CLS]
            off = (i - (len(rs) - 1) / 2) * w
            ax.bar(x + off, vals, width=w * 0.92, color=SPEC_STYLE[r["_spec"]][0],
                   hatch=SPEC_STYLE[r["_spec"]][1], edgecolor="white", linewidth=0.6,
                   alpha=0.55 + 0.45 * (i + 1) / len(rs),
                   label="%s (%s)" % (r["_name"], r["_spec"]))
        ax.set_xticks(x, [CN[c] for c in CLS])
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.0)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("分类别精度：ROI 主要救的是「极片损伤」与小目标，不是已经很好的暗斑", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out, ROOT))


def plot_workpoint(reports, specs, out):
    """工作点下的分类别 P / R 分组柱状图。

    比 P-R 散点更可读：散点在高分区会糊成一团，柱子能一眼看出「哪一类、差多少」。
    """
    rs = _order(reports, specs)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    w = 0.8 / max(1, len(rs))
    x = np.arange(len(CLS))
    for ax, (key, title) in zip(axes, [("recall", "分类别召回率 R（漏检的直接来源）"),
                                       ("precision", "分类别精确率 P（误报的直接来源）")]):
        for i, r in enumerate(rs):
            pc = r["layer2_3_workpoint"]["per_class"]
            vals = [pc[c][key] for c in CLS]
            off = (i - (len(rs) - 1) / 2) * w
            ax.bar(x + off, vals, width=w * 0.92, color=SPEC_STYLE[r["_spec"]][0],
                   hatch=SPEC_STYLE[r["_spec"]][1], edgecolor="white", linewidth=0.6,
                   alpha=0.55 + 0.45 * (i + 1) / len(rs),
                   label="%s (%s)" % (r["_name"], r["_spec"]))
        ax.set_xticks(x, [CN[c] for c in CLS])
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        for b in ax.containers:
            try:
                ax.bar_label(b, fmt="%.2f", fontsize=5.6, padding=1)
            except Exception:
                pass
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("工作点 conf=0.25 / IoU=0.5 的分类别 P、R", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out, ROOT))


def plot_sweep(sweeps, out):
    """conf 扫描曲线：选工作点的依据。sweeps = [(name, dict), ...]"""
    if not sweeps:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    # (1) 全局 P/R/F1
    ax = axes[0]
    for i, (name, d) in enumerate(sweeps):
        c = [p["conf"] for p in d["points"]]
        col = SPEC_STYLE.get(spec_key(d), ("#475569", ""))[0]
        ax.plot(c, [p["precision"] for p in d["points"]], lw=1.4, color=col,
                label="%s P" % name)
        ax.plot(c, [p["recall"] for p in d["points"]], lw=1.4, ls="--", color=col,
                label="%s R" % name)
        ax.plot(c, [p["f1"] for p in d["points"]], lw=1.2, ls=":", color=col,
                label="%s F1" % name)
    ax.set_xlabel("conf")
    ax.set_title("全局 P / R / F1 随 conf 变化", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, ncol=2)

    # (2) 代价曲线：FP千张 与 组级漏检
    ax = axes[1]
    ax2 = ax.twinx()
    for i, (name, d) in enumerate(sweeps):
        c = [p["conf"] for p in d["points"]]
        col = SPEC_STYLE.get(spec_key(d), ("#475569", ""))[0]
        ax.plot(c, [p["fp_per_1000img"] for p in d["points"]], lw=1.6, color=col,
                label="%s FP/千张" % name)
        gm = [(100 * p["group_miss"]) if p["group_miss"] is not None else np.nan
              for p in d["points"]]
        ax2.plot(c, gm, lw=1.6, ls="--", color=col, label="%s 组级漏检%%" % name)
    ax.set_xlabel("conf")
    ax.set_ylabel("FP / 1000 张", fontsize=9)
    ax2.set_ylabel("组级漏检率 (%)", fontsize=9)
    ax.set_title("代价曲线：误报 vs 漏检（左轴实线 / 右轴虚线）", fontsize=10.5)
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper center", ncol=3, framealpha=0.85)

    # (3) 损伤类召回随 conf 变化（这是本次诊断的关键类）
    ax = axes[2]
    for i, (name, d) in enumerate(sweeps):
        c = [p["conf"] for p in d["points"]]
        ys = [p["per_class"].get("0", {}).get("r", np.nan) for p in d["points"]]
        col = SPEC_STYLE.get(spec_key(d), ("#475569", ""))[0]
        ax.plot(c, ys, lw=1.8, color=col, label=name)
    ax.set_xlabel("conf")
    ax.set_ylabel("极片损伤 召回率", fontsize=9)
    ax.set_title("极片损伤召回随 conf 变化（ROI 版整体更低且抬不起来）", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)
    for a in axes:
        a.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out, ROOT))


def plot_fp_miss(reports, specs, out):
    rs = _order(reports, specs)
    n = len(rs)
    x = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(max(10, 1.15 * n + 4), 4.4))
    for ax, (getter, title, ylab, fmt) in zip(axes, [
            (lambda r: r["layer2_3_workpoint"]["fp_per_1000img"],
             "误报密度 FP/千张（越低越好）", "FP / 1000 张", "%.0f"),
            (lambda r: 100 * r["layer2_3_workpoint"]["group_level"]["miss_rate"],
             "组级漏检率（有缺陷组中至少一个缺陷漏检）", "漏检率 (%)", "%.1f")]):
        vals = [getter(r) for r in rs]
        bars = ax.bar(x, vals, color=[SPEC_STYLE[r["_spec"]][0] for r in rs],
                      hatch=[SPEC_STYLE[r["_spec"]][1] for r in rs],
                      edgecolor="white", linewidth=0.8, width=0.66)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02, fmt % v,
                    ha="center", fontsize=8)
        ax.set_xticks(x, [r["_name"] for r in rs], rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, max(vals) * 1.2)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=SPEC_STYLE[s][0],
                             hatch=SPEC_STYLE[s][1], edgecolor="white") for s in specs]
    axes[0].legend(handles, specs, fontsize=7.5, title="推理口径", title_fontsize=7.5)
    fig.suptitle("产线关心的两个代价：误报多了要人复核，漏检多了要流出", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out, ROOT))


def write_table(reports, specs, out):
    # 训练集覆盖了冻结 test 的 run 必须标注：它们的 mAP 被泄漏抬高，不可用于选型比较。
    # 依据 reports/splits/v5_split_manifest.csv 的泄漏量化（V4 训练集覆盖 test 93.9%）。
    LEAKED = {"V3", "V4", "v2mild",
              "CT_DATASET_FINAL_V4_AUG_1016_1616"}
    rs = _order(reports, specs)
    lines = ["# 评测汇总（plot_eval.py 自动生成，勿手改）", "",
             "- 同一清单、同一 split 才算同一张考卷；口径不同则绝对值不可比",
             "- 工作点 conf=0.25 / IoU=0.5", "",
             "| 报告 | 口径 | mAP50 | mAP50-95 | 损伤 AP50/AP | 暗斑 AP50/AP | 亮斑 AP50/AP | FP千张 | 组级漏检 | 备注 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rs:
        pc = r["layer1_coco"]["per_class"]
        wp = r["layer2_3_workpoint"]
        note = "⚠ 训练集覆盖 test 93.9%，数字被泄漏抬高，不可用于选型" \
            if any(k in r["_name"] for k in LEAKED) else ""
        lines.append("| %s | %s | %.3f | %.3f | %.3f / %.3f | %.3f / %.3f | %.3f / %.3f | %.1f | %.1f%% | %s |" % (
            r["_name"], r["_spec"],
            r["layer1_coco"]["mAP50"], r["layer1_coco"]["mAP50_95"],
            pc["damage"]["ap50"], pc["damage"]["ap"],
            pc["dark"]["ap50"], pc["dark"]["ap"],
            pc["light"]["ap50"], pc["light"]["ap"],
            wp["fp_per_1000img"], 100 * wp["group_level"]["miss_rate"], note))
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  -> %s" % os.path.relpath(out, ROOT))


def plot_by_res(out):
    """按原生分辨率分组的工作点对比（数据来自 scripts/eval_by_res.py）。"""
    p = os.path.join(EVAL_DIR, "by_resolution.json")
    if not os.path.isfile(p):
        return
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    models = d.get("models") or []
    if not models:
        return
    gnames = list(models[0]["groups"].keys())
    palette = ["#b91c1c", "#1d4ed8", "#059669", "#b45309", "#7c3aed"]
    fig, axes = plt.subplots(1, len(gnames) + 1, figsize=(6.0 * (len(gnames) + 1), 4.6))
    w = 0.8 / max(1, len(models))
    x = np.arange(len(CLS) + 2)          # 3 类召回 + 组级漏检 + FP千张(归一化另画)

    for gi, gname in enumerate(gnames):
        ax = axes[gi]
        for i, m in enumerate(models):
            g = m["groups"][gname]
            vals = [g["per_class"][c]["recall"] for c in CLS] + [g["group_miss"]]
            off = (i - (len(models) - 1) / 2) * w
            bars = ax.bar(np.arange(len(CLS) + 1) + off, vals, width=w * 0.92,
                          color=palette[i % len(palette)], edgecolor="white", linewidth=0.6,
                          alpha=0.6 + 0.4 * (i + 1) / len(models),
                          label="%s (%s)" % (m["name"], m["spec"]))
            ax.bar_label(bars, fmt="%.2f", fontsize=5.8, padding=1)
        ax.set_xticks(np.arange(len(CLS) + 1),
                      [CN[c] for c in CLS] + ["组级漏检率"], fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title("%s（%d 张）" % (gname, models[0]["groups"][gname]["n_images"]), fontsize=10.5)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(labelsize=8)
        if gi == 0:
            ax.legend(fontsize=7, loc="lower left")

    ax = axes[-1]
    for i, m in enumerate(models):
        off = (i - (len(models) - 1) / 2) * w
        vals = [m["groups"][g]["fp_per_1000img"] for g in gnames]
        bars = ax.bar(np.arange(len(gnames)) + off, vals, width=w * 0.92,
                      color=palette[i % len(palette)], edgecolor="white", linewidth=0.6,
                      alpha=0.6 + 0.4 * (i + 1) / len(models),
                      label="%s" % m["name"])
        ax.bar_label(bars, fmt="%.0f", fontsize=6.5, padding=1)
    ax.set_xticks(np.arange(len(gnames)), [g.split(" ")[0] for g in gnames], fontsize=8)
    ax.set_ylabel("FP / 千张", fontsize=9)
    ax.set_title("误报密度（越低越好）", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("按原生分辨率分组：ROI 主要改变的是低分辨率组，高分辨率组并未受益\n"
                 "（口径为 predict/scaleup=True，与产线一致；不是 model.val 的 scaleup=False）",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("  -> %s" % os.path.relpath(out, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="append", metavar="NAME=PATH")
    ap.add_argument("--sweep", action="append", metavar="NAME=PATH",
                    help="conf 扫描结果；不传则自动收录 reports/eval/sweep_*.json")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--outdir", default="reports/eval/figs")
    args = ap.parse_args()

    pairs = []
    if args.report:
        for s in args.report:
            if "=" not in s:
                raise SystemExit("--report 需写成 NAME=PATH，收到 %r" % s)
            a, b = s.split("=", 1)
            pairs.append((a, b))
    else:
        pairs = discover()

    reports = load_reports(pairs)
    if not reports:
        raise SystemExit("没有可用的评测报告")

    if args.no_sweep:
        sweep_pairs = []
    elif args.sweep:
        sweep_pairs = [tuple(s.split("=", 1)) for s in args.sweep if "=" in s]
    else:
        sweep_pairs = [(os.path.basename(p)[len("sweep_"):-len(".json")], p)
                       for p in sorted(glob.glob(os.path.join(EVAL_DIR, "sweep_*.json")))]
    sweeps = load_sweeps(sweep_pairs)

    specs = assign_styles(reports, sweeps)

    outdir = args.outdir if os.path.isabs(args.outdir) else os.path.join(ROOT, args.outdir)
    os.makedirs(outdir, exist_ok=True)
    print("收录 %d 份报告 / %d 份扫描，%d 种口径：%s"
          % (len(reports), len(sweeps), len(specs), " / ".join(specs)))

    plot_map(reports, specs, os.path.join(outdir, "eval_map.png"))
    plot_perclass(reports, specs, os.path.join(outdir, "eval_perclass_ap.png"))
    plot_workpoint(reports, specs, os.path.join(outdir, "eval_workpoint.png"))
    plot_fp_miss(reports, specs, os.path.join(outdir, "eval_fp_miss.png"))
    plot_by_res(os.path.join(outdir, "eval_by_res.png"))
    if sweeps:
        plot_sweep(sweeps, os.path.join(outdir, "eval_sweep.png"))
    write_table(reports, specs, os.path.join(EVAL_DIR, "eval_table.md"))


if __name__ == "__main__":
    sys.exit(main())
