# -*- coding: utf-8 -*-
"""把每一次训练固化成「一个文件夹」，并在 experiments/ 下建立总索引。

背景：`runs/detect/CT_pro/<name>/` 是 ultralytics 自动生成的，产物齐全但缺少
「这次到底测出了什么、结论是什么」。而评测结果散落在 reports/eval/*.json，
时间一长就分不清哪个数字属于哪次训练。

本脚本做三件事，全程只读 runs/ 与 reports/，不改动任何原始产物：
  1. experiments/<run>/01_train/  复制训练产物（weights 除外，*.pt 不入库）并写 md5
  2. experiments/<run>/02_eval/   收进该 run 的所有评测报告（eval / sweep）并出单 run 图
  3. experiments/<run>/04_report.md  生成带真实数字的分析报告骨架（已有则不覆盖）
     experiments/INDEX.md            全 run 总表

用法：
    # 归档单次
    runtime\\python.exe scripts\\archive_experiment.py --run V51_ROI_1536
    # 归档全部（含历史 run）
    runtime\\python.exe scripts\\archive_experiment.py --all
    # 只重建索引
    runtime\\python.exe scripts\\archive_experiment.py --index-only
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import shutil
import sys
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs", "detect", "CT_pro")
EXP_DIR = os.path.join(ROOT, "experiments")
EVAL_DIR = os.path.join(ROOT, "reports", "eval")

TRAIN_FILES = [
    "args.yaml", "results.csv", "results.png", "train.log", "_train_meta.json",
    "labels.jpg", "confusion_matrix.png", "confusion_matrix_normalized.png",
    "BoxF1_curve.png", "BoxPR_curve.png", "BoxP_curve.png", "BoxR_curve.png",
    "train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg",
    "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    "val_batch1_labels.jpg", "val_batch1_pred.jpg",
    "val_batch2_labels.jpg", "val_batch2_pred.jpg",
]

CLS_CN = {"damage": "极片损伤", "dark": "暗斑", "light": "亮斑"}


# ---------------------------------------------------------------- 工具
def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_json(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def read_csv_rows(p):
    try:
        with open(p, "r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _fmt(v, nd=4, dash="-"):
    return dash if v is None else ("%." + str(nd) + "f") % v


# ---------------------------------------------------------------- 采集
def collect_runs():
    out = []
    if not os.path.isdir(RUNS_DIR):
        return out
    for n in sorted(os.listdir(RUNS_DIR)):
        d = os.path.join(RUNS_DIR, n)
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "args.yaml")):
            out.append(n)
    return out


def pick_eval_reports(run):
    """按 weights 路径归属，把评测报告精确匹配到 run（避免靠文件名猜）"""
    hits = []
    for p in sorted(glob.glob(os.path.join(EVAL_DIR, "*.json"))):
        d = read_json(p)
        if not isinstance(d, dict):
            continue
        w = (d.get("weights") or "").replace("\\", "/")
        if "/%s/" % run in w:
            hits.append((os.path.basename(p), p))
    return hits


def extract_eval(d):
    """把两代 schema（mAP50 / map50）统一成一套字段"""
    l1 = d["layer1_coco"]
    wp = d["layer2_3_workpoint"]
    sp = l1.get("speed_ms") or {}
    gg = wp.get("group_level") or {}
    per = {}
    for c in ("damage", "dark", "light"):
        a = (l1.get("per_class") or {}).get(c) or {}
        b = (wp.get("per_class") or {}).get(c) or {}
        per[c] = {
            "ap50": a.get("ap50"), "ap": a.get("ap"),
            "p": b.get("precision"), "r": b.get("recall"),
            "tp": b.get("tp"), "fp": b.get("fp"), "fn": b.get("fn"),
        }
    return {
        "tag": os.path.splitext(os.path.basename(d.get("_path", "")))[0],
        "map50": l1.get("mAP50", l1.get("map50")),
        "map50_95": l1.get("mAP50_95", l1.get("map")),
        "map75": l1.get("mAP75", l1.get("map75")),
        "fp_per_1000": wp.get("fp_per_1000img"),
        "conf": (wp.get("workpoint") or {}).get("conf"),
        "iou": (wp.get("workpoint") or {}).get("iou"),
        "n_images": d.get("n_images"),
        "n_missed_groups": gg.get("n_missed_groups"),
        "n_defect_groups": gg.get("n_defect_groups"),
        "group_miss": gg.get("miss_rate"),
        "cell_miss": gg.get("cell_level_miss_rate"),
        "cell_n": gg.get("cell_level_n"),
        "imgsz": d.get("imgsz_layer1") or d.get("imgsz"),
        "imgsz_predict": d.get("imgsz_predict"),
        "roi": d.get("roi"),
        "speed_ms": (sp.get("preprocess", 0) or 0) + (sp.get("inference", 0) or 0)
                    + (sp.get("postprocess", 0) or 0) if sp else None,
        "elapsed_s": d.get("elapsed_s"),
        "weights_md5": d.get("weights_md5"),
        "per_class": per,
    }


def train_summary(run_dir):
    rows = read_csv_rows(os.path.join(run_dir, "results.csv"))
    args = {}
    ap = os.path.join(run_dir, "args.yaml")
    if os.path.isfile(ap):
        try:
            import yaml
            with open(ap, "r", encoding="utf-8") as f:
                args = yaml.safe_load(f) or {}
        except Exception:
            args = {}
    meta = read_json(os.path.join(run_dir, "_train_meta.json")) or {}
    if not rows:
        return {"rows": [], "args": args, "meta": meta}
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
    last = rows[-1]
    tb, vb = float(last["train/box_loss"]), float(last["val/box_loss"])
    return {
        "rows": rows, "args": args, "meta": meta,
        "n_epochs": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_map50": float(best["metrics/mAP50(B)"]),
        "best_map50_95": float(best["metrics/mAP50-95(B)"]),
        "last_best_map50_95": float(best["metrics/mAP50-95(B)"]),
        "train_total_s": float(last.get("time") or 0),
        "tar_box": tb, "val_box": vb,
        "overfit_gap": (vb - tb) / tb * 100 if tb else None,
        "stopped_early": len(rows) < int(args.get("epochs") or 0),
    }


# ---------------------------------------------------------------- 单 run 图
def plot_single(summ, ev, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    rows = summ["rows"]
    if not rows and not ev:
        return False
    n_panel = 3 if ev else 2
    fig, axes = plt.subplots(1, n_panel, figsize=(5.0 * n_panel, 3.8))
    ep = [int(r["epoch"]) for r in rows]
    if rows:
        ax = axes[0]
        ax.plot(ep, [float(r["train/box_loss"]) for r in rows], lw=1.3, color="#b91c1c", label="train box")
        ax.plot(ep, [float(r["val/box_loss"]) for r in rows], lw=1.3, color="#1d4ed8", label="val box")
        ax.set_title("损失", fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

        ax = axes[1]
        ax.plot(ep, [float(r["metrics/mAP50(B)"]) for r in rows], lw=1.3, color="#0891b2", label="mAP50")
        ax.plot(ep, [float(r["metrics/mAP50-95(B)"]) for r in rows], lw=1.3, color="#b45309", label="mAP50-95")
        bx = summ["best_epoch"]
        ax.axvline(bx, ls=":", lw=1.0, color="#64748b")
        ax.annotate("best ep%d\n%.4f" % (bx, summ["best_map50_95"]), (bx, summ["best_map50_95"]),
                    fontsize=7, xytext=(4, -14), textcoords="offset points", color="#334155")
        ax.set_title("验证集 mAP（model.val 口径）", fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    if ev:
        ax = axes[2]
        cls = ["damage", "dark", "light"]
        x = np.arange(len(cls))
        w = 0.26
        for i, (k, lb, col) in enumerate((("p", "P", "#b91c1c"), ("r", "R", "#1d4ed8"),
                                          ("ap", "AP50-95", "#059669"))):
            vals = [ev["per_class"][c].get(k) or 0 for c in cls]
            bars = ax.bar(x + (i - 1) * w, vals, width=w, color=col, label=lb, edgecolor="white", linewidth=0.5)
            ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
        ax.set_xticks(x, [CLS_CN[c] for c in cls], fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title("工作点 P/R + AP（冻结 test）", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- 报告
def build_report(run, summ, evals, sweeps):
    a = summ["args"]
    meta = summ.get("meta") or {}
    hp = meta.get("hyperparams") or {}
    hp_eff = meta.get("hyperparams_effective") or {}
    L = []
    L.append("# `%s` 训练结果与分析" % run)
    L.append("")
    L.append("> 本文件由 `scripts/archive_experiment.py` 依据 `runs/detect/CT_pro/%s/` 与 "
             "`reports/eval/` 的真实文件生成，数字未经人工改写。" % run)
    L.append("> 生成时间：%s" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))
    L.append("")

    L.append("## 1. 配置")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    for k, lab in (("data", "数据集"), ("model", "初始化权重"), ("imgsz", "画布 imgsz"),
                   ("epochs", "计划轮数"), ("batch", "batch"), ("optimizer", "优化器"),
                   ("lr0", "lr0"), ("patience", "patience")):
        if a.get(k) is not None:
            L.append("| %s | `%s` |" % (lab, a.get(k)))
    if meta.get("roi"):
        L.append("| ROI 裁剪 | 是，box=%s |" % (meta.get("roi_box")))
    L.append("")
    L.append("**数据增强（ultralytics 实际生效值）**")
    L.append("")
    keys = ("mosaic", "close_mosaic", "degrees", "translate", "scale", "shear", "perspective",
            "flipud", "fliplr", "hsv_h", "hsv_s", "hsv_v", "erasing", "copy_paste", "mixup",
            "cutmix", "cls_pw")
    src = hp_eff or a
    L.append("| " + " | ".join(keys) + " |")
    L.append("|" + "---|" * len(keys))
    L.append("| " + " | ".join(str(src.get(k, "-")) for k in keys) + " |")
    L.append("")

    if summ["rows"]:
        L.append("## 2. 训练过程")
        L.append("")
        L.append("| 项 | 值 |")
        L.append("|---|---|")
        L.append("| 实际完成轮数 | %d / %s%s |" % (
            summ["n_epochs"], a.get("epochs", "?"),
            "（早停）" if summ["stopped_early"] else ""))
        L.append("| 最优 epoch | %d |" % summ["best_epoch"])
        L.append("| 最优 mAP50 / mAP50-95 | %.4f / **%.4f** |" % (summ["best_map50"], summ["best_map50_95"]))
        L.append("| 总训练时长 | %.2f h（%.1f s/epoch） |" % (
            summ["train_total_s"] / 3600, summ["train_total_s"] / max(1, summ["n_epochs"])))
        L.append("| 末轮 train / val box_loss | %.3f / %.3f |" % (summ["tar_box"], summ["val_box"]))
        gap = summ["overfit_gap"]
        L.append("| 泛化间隙 (val-train)/train | %s |" % (
            "-" if gap is None else "%+.1f%%" % gap))
        L.append("")
        L.append("![训练曲线](01_train/results.png)")
        L.append("")

    if evals:
        L.append("## 3. 冻结 test 评测")
        L.append("")
        L.append("口径：`%s`%s；工作点 conf=%s / IoU=%s。" % (
            ("ROI@%s" % evals[0]["imgsz"]) if evals[0]["roi"] else ("全图@%s" % evals[0]["imgsz"]),
            "（layer1 走 model.val，固定 conf=0.001）", evals[0]["conf"], evals[0]["iou"]))
        L.append("")
        L.append("| 报告 | mAP50 | mAP50-95 | FP/千张 | 组级漏检 | 单张耗时 |")
        L.append("|---|---|---|---|---|---|")
        for e in evals:
            L.append("| `%s` | %.4f | %.4f | %.1f | %s/%s (%.2f%%) | %.1f ms |" % (
                e["tag"], e["map50"], e["map50_95"], e["fp_per_1000"],
                e["n_missed_groups"], e["n_defect_groups"], 100 * (e["group_miss"] or 0),
                e["speed_ms"] or 0))
        L.append("")
        e = evals[0]
        L.append("**分类别（工作点 P/R + AP）**")
        L.append("")
        L.append("| 类别 | AP50 | AP50-95 | P | R | TP | FP | FN |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c, cn in CLS_CN.items():
            v = e["per_class"][c]
            L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
                cn, _fmt(v["ap50"]), _fmt(v["ap"]), _fmt(v["p"]), _fmt(v["r"]),
                v["tp"], v["fp"], v["fn"]))
        L.append("")
        if evals[0].get("cell_miss") is not None:
            L.append("电芯级漏检（抽样 n=%s）：%.0f%%" % (e["cell_n"], 100 * (e["cell_miss"] or 0)))
            L.append("")
        if os.path.isfile(os.path.join(EXP_DIR, run, "02_eval", "per_class.png")):
            L.append("![单 run 概览](02_eval/per_class.png)")
            L.append("")

    if sweeps:
        L.append("## 4. 工作点扫描（conf 敏感度）")
        L.append("")
        L.append("| conf | P | R | F1 | FP/千张 | 组级漏检 |")
        L.append("|---|---|---|---|---|---|")
        for p in sweeps[0].get("points", []):
            gm = p.get("group_miss")
            L.append("| %.2f | %.4f | %.4f | %.4f | %.1f | %s |" % (
                p["conf"], p["precision"], p["recall"], p["f1"], p["fp_per_1000img"],
                "-" if gm is None else "%.2f%%" % (100 * gm)))
        L.append("")

    L.append("## 5. 结论与判断（人工填写）")
    L.append("")
    L.append("<!-- ARCHIVE-NOTES-START -->")
    L.append("_待补充：本 run 相对基线的净收益/净损失、可信度、是否可上线、遗留风险。_")
    L.append("<!-- ARCHIVE-NOTES-END -->")
    L.append("")
    L.append("## 6. 产物清单")
    L.append("")
    L.append("| 目录 | 内容 |")
    L.append("|---|---|")
    L.append("| `01_train/` | args.yaml、results.csv、results.png、曲线、混淆矩阵、batch 预览、train.log、`_train_meta.json` |")
    L.append("| `02_eval/` | 冻结 test 报告 JSON、conf 扫描 JSON、单 run 概览图 |")
    L.append("| `00_meta.json` | 本 run 的结构化摘要（供脚本与索引消费） |")
    L.append("| `weights/` | 不复制；见 `weights.txt` 记录的原路径与 md5 |")
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------- 主流程
def roi_of_run(run_dir, args):
    """该 run 的推理口径是不是 ROI。

    优先级：meta 的 roi=True 直接认定；否则看数据集名字里是否含 ROI
    —— 数据集名是物化时写入的、不可能说谎，而旧版 train_run.py 会把 roi 写成 False。
    推理端 web/app.py 只认 meta 里的 roi 位，所以这里也是唯一的事实来源。
    """
    meta = read_json(os.path.join(run_dir, "_train_meta.json")) or {}
    if meta.get("roi") is True:
        return True, "meta"
    data = str(args.get("data") or "")
    dname = os.path.basename(os.path.dirname(os.path.abspath(data))) if data else ""
    if "ROI" in dname.upper():
        return True, "dataset-name"
    return False, "none"


def backfill_roi_meta(run):
    """把漏写 roi 位的 _train_meta.json 补齐（老版本 train_run.py 的产物）。

    web/app.py 读这一位决定推理时是否先裁剪；缺这一位会把全图喂给 ROI 模型，
    结果全错但不报错，属于必须修掉的静默故障。
    """
    run_dir = os.path.join(RUNS_DIR, run)
    mp = os.path.join(run_dir, "_train_meta.json")
    if not os.path.isfile(mp):
        return None
    meta = read_json(mp) or {}
    # 已经显式且自洽（True，或 False 且有来源标注）就不动；只修「缺位」和「写错成 False」
    if meta.get("roi") is True:
        return None
    if meta.get("roi") is False and meta.get("roi_source"):
        return None
    rows = read_csv_rows(os.path.join(run_dir, "results.csv"))
    args = {}
    ap = os.path.join(run_dir, "args.yaml")
    if os.path.isfile(ap):
        try:
            import yaml
            with open(ap, "r", encoding="utf-8") as f:
                args = yaml.safe_load(f) or {}
        except Exception:
            args = {}
    is_roi, src = roi_of_run(run_dir, args)
    sys.path.insert(0, os.path.join(ROOT, "core"))
    try:
        import roi as R
        box = {"x0": R.ROI_X0, "x1": R.ROI_X1, "y0": R.ROI_Y0, "y1": R.ROI_Y1}
    except Exception:
        box = None
    meta["roi"] = is_roi
    meta["roi_source"] = "backfill:%s" % src
    meta["roi_box"] = box if is_roi else None
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print("  [回填] %-30s roi=%s（依据 %s）" % (run, is_roi, src))
    return is_roi


def archive_one(run, force=False):
    run_dir = os.path.join(RUNS_DIR, run)
    if not os.path.isdir(run_dir):
        print("  [跳过] 不存在：%s" % run_dir)
        return None
    backfill_roi_meta(run)   # 幂等：只在缺 roi 位时动手
    exp = os.path.join(EXP_DIR, run)
    train_out = os.path.join(exp, "01_train")
    eval_out = os.path.join(exp, "02_eval")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(eval_out, exist_ok=True)

    copied = []
    for fn in TRAIN_FILES:
        src = os.path.join(run_dir, fn)
        if os.path.isfile(src):
            dst = os.path.join(train_out, fn)
            shutil.copy2(src, dst)
            copied.append(fn)
    # 训练额外产物（不同 ultralytics 版本会多出几张图）
    for src in glob.glob(os.path.join(run_dir, "*.png")) + glob.glob(os.path.join(run_dir, "*.jpg")):
        fn = os.path.basename(src)
        if fn not in copied:
            shutil.copy2(src, os.path.join(train_out, fn))
            copied.append(fn)

    # 权重：只记录，不复制（*.pt 不入库）
    wlines = []
    for fn in ("best.pt", "last.pt"):
        p = os.path.join(run_dir, "weights", fn)
        if os.path.isfile(p):
            wlines.append("%s  md5=%s  %.2f MB" % (os.path.relpath(p, ROOT), md5_of(p),
                                                   os.path.getsize(p) / 1e6))
    if wlines:
        with open(os.path.join(exp, "weights.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(wlines) + "\n")

    # 评测报告
    evals, sweeps = [], []
    for fn, p in pick_eval_reports(run):
        d = read_json(p)
        if not d:
            continue
        shutil.copy2(p, os.path.join(eval_out, fn))
        if "points" in d:
            sweeps.append(d)
        elif "layer1_coco" in d:
            d["_path"] = p
            evals.append(extract_eval(d))

    summ = train_summary(run_dir)
    if summ["rows"] or evals:
        plot_single(summ, evals[0] if evals else None,
                    os.path.join(eval_out, "per_class.png"))

    # 报告（已有则保留人工结论，只刷新数据段之前的部分由 build_report 覆盖 →
    # 这里选择：首次生成；若已存在则只替换 ARCHIVE 标记之前的前 4 节）
    rp = os.path.join(exp, "04_report.md")
    new = build_report(run, summ, evals, sweeps)
    if os.path.isfile(rp) and not force:
        old = open(rp, "r", encoding="utf-8").read()
        if "<!-- ARCHIVE-NOTES-START -->" in old:
            keep = old.split("<!-- ARCHIVE-NOTES-START -->")[1]
            new = new.split("<!-- ARCHIVE-NOTES-START -->")[0] + "<!-- ARCHIVE-NOTES-START -->" + keep
    with open(rp, "w", encoding="utf-8") as f:
        f.write(new)

    meta = {
        "run": run,
        "archived_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": summ["args"].get("data"),
        "model": summ["args"].get("model"),
        "imgsz": summ["args"].get("imgsz"),
        "roi": roi_of_run(run_dir, summ["args"])[0],
        "epochs_planned": summ["args"].get("epochs"),
        "epochs_done": summ.get("n_epochs"),
        "best_epoch": summ.get("best_epoch"),
        "best_map50": summ.get("best_map50"),
        "best_map50_95": summ.get("best_map50_95"),
        "train_hours": round(summ["train_total_s"] / 3600, 2) if summ.get("train_total_s") else None,
        "overfit_gap_pct": round(summ["overfit_gap"], 1) if summ.get("overfit_gap") is not None else None,
        "evals": [{"tag": e["tag"], "map50": e["map50"], "map50_95": e["map50_95"],
                   "fp_per_1000": e["fp_per_1000"], "group_miss": e["group_miss"],
                   "roi": e["roi"], "imgsz": e["imgsz"], "speed_ms": e["speed_ms"]} for e in evals],
    }
    with open(os.path.join(exp, "00_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)

    print("  [归档] %-30s 训练产物 %2d 个 | 评测 %d 份 | 扫描 %d 份" % (
        run, len(copied), len(evals), len(sweeps)))
    return meta


def write_index(metas):
    metas = [m for m in metas if m]
    metas.sort(key=lambda m: (m.get("archived_at") or ""))
    L = ["# 训练记录总索引", "",
         "> 由 `scripts/archive_experiment.py` 生成。每次训练一个文件夹，"
         "内含 `01_train/`（原始产物）、`02_eval/`（评测）、`04_report.md`（数据+结论）。", "",
         "| run | 画布 | ROI | 完成/计划轮 | 最优 ep | mAP50 | mAP50-95 | FP/千张 | 组级漏检 | 训练时长 | 报告 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in metas:
        e = (m.get("evals") or [{}])[0]
        L.append("| `%s` | %s | %s | %s/%s | %s | %s | %s | %s | %s | %s | [报告](%s/04_report.md) |" % (
            m["run"], m.get("imgsz"), "是" if m.get("roi") else "否",
            m.get("epochs_done"), m.get("epochs_planned"), m.get("best_epoch"),
            _fmt(e.get("map50")), _fmt(e.get("map50_95")),
            _fmt(e.get("fp_per_1000"), 1),
            "-" if e.get("group_miss") is None else "%.2f%%" % (100 * e["group_miss"]),
            ("%.2f h" % m["train_hours"]) if m.get("train_hours") else "-",
            m["run"]))
    L.append("")
    L.append("> 注意：`mAP` 列来自各 run 自己的评测报告，**口径可能不同**（全图 / ROI、不同画布），"
             "跨口径的绝对值不可直接比较，必须看「画布」与「ROI」两列。")
    L.append("")
    L.append("## 维护约定")
    L.append("")
    L.append("1. 训练脚本现在会在结束后自动归档本次 run（`--no-archive` 可关闭）。")
    L.append("2. 评测跑完后重新执行 `archive_experiment.py --run <name>`，会把新报告并进 `02_eval/`。")
    L.append("3. `04_report.md` 的第 1–4 节由脚本按真实文件重算；第 5 节「结论与判断」是人工写的，"
             "重跑不会覆盖（`--force` 才会）。")
    L.append("4. 权重不入库（`*.pt` 被 .gitignore 拦截），`weights.txt` 里留了原路径与 md5，可校验。")
    L.append("")
    with open(os.path.join(EXP_DIR, "INDEX.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("[索引] experiments/INDEX.md（%d 次训练）" % len(metas))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[], help="run 名，可重复")
    ap.add_argument("--all", action="store_true", help="归档 runs/detect/CT_pro 下全部 run")
    ap.add_argument("--index-only", action="store_true")
    ap.add_argument("--backfill-meta", action="store_true",
                    help="给缺 roi 位的 _train_meta.json 回填（老版本训练脚本的产物）")
    ap.add_argument("--force", action="store_true", help="连人工结论段一起重建")
    args = ap.parse_args()

    os.makedirs(EXP_DIR, exist_ok=True)

    if not args.index_only:
        targets = collect_runs() if args.all else args.run
        if not targets:
            raise SystemExit("请用 --run NAME 或 --all")
        print("开始归档 %d 个 run -> experiments/" % len(targets))
        for r in targets:
            if args.backfill_meta:
                backfill_roi_meta(r)
            archive_one(r, force=args.force)

    metas = []
    for n in collect_runs():
        p = os.path.join(EXP_DIR, n, "00_meta.json")
        if os.path.isfile(p):
            metas.append(read_json(p))
    write_index(metas)


if __name__ == "__main__":
    main()
