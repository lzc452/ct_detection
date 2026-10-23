# -*- coding: utf-8 -*-
"""
多尺度预测融合：把两档 imgsz 的预测按类合并（NMS/加权平均），在不动模型的前提下换召回。

动机来自 scripts/oss_scale_probe.py 的实测：V5 的有效尺度窗口很窄（0.90 最佳、0.99 次优、
0.60 以下归零），单尺度必然漏掉一部分目标；两档互补可能比任何单一尺度都好。

用法：
  references\\.venv\\Scripts\\python.exe scripts\\oss_fuse.py \
      --a reports/oss_bench/preds_yolo_test.json --b reports/oss_bench/preds_yolo_2112.json
"""
import os
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
from detection_metrics import match_predictions, iou_xyxy  # noqa: E402
import dataset_split as ds  # noqa: E402

CLS_NAME = {0: "damage", 1: "dark", 2: "light"}


def fuse_nms(list_a, list_b, conf, iou_th):
    """按类合并两路预测：IoU 超阈值的保留高置信那个（等价于对并集做一次 NMS）。"""
    merged = [p for p in list_a + list_b if p["conf"] >= conf]
    out = []
    for cls_id in (0, 1, 2):
        cand = sorted([p for p in merged if p["cls"] == cls_id], key=lambda z: -z["conf"])
        kept = []
        for p in cand:
            if all(iou_xyxy(p["xyxy"], q["xyxy"]) <= iou_th for q in kept):
                kept.append(p)
        out.extend(kept)
    return out


def fuse_wbf(list_a, list_b, conf, iou_th):
    """加权平均融合：匹配上的两框按 conf 加权平均坐标、conf 取均值。"""
    a = [p for p in list_a if p["conf"] >= conf]
    b = [p for p in list_b if p["conf"] >= conf]
    used_b, out = set(), []
    for p in a:
        best, best_iou = -1, 0.0
        for j, q in enumerate(b):
            if j in used_b or q["cls"] != p["cls"]:
                continue
            v = iou_xyxy(p["xyxy"], q["xyxy"])
            if v > best_iou:
                best, best_iou = j, v
        if best >= 0 and best_iou >= iou_th:
            q = b[best]
            used_b.add(best)
            w1, w2 = p["conf"], q["conf"]
            xyxy = [(p["xyxy"][k] * w1 + q["xyxy"][k] * w2) / (w1 + w2) for k in range(4)]
            out.append({"cls": p["cls"], "conf": max(w1, w2), "xyxy": xyxy})
        else:
            out.append(p)
    for j, q in enumerate(b):
        if j not in used_b:
            out.append(q)
    return out


def gt_of(row):
    W, H = [int(x) for x in row["size"].split("x")]
    o = []
    if row["label_json"]:
        _, _, boxes = ds.json_to_yolo(os.path.join(ROOT, row["label_json"]))
        for cls, xc, yc, bw, bh in boxes:
            o.append({"cls": int(cls),
                      "xyxy": [(xc - bw / 2) * W, (yc - bh / 2) * H,
                               (xc + bw / 2) * W, (yc + bh / 2) * H]})
    return o


def score(rows, preds, conf, iou_wp):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(3)}
    grp = defaultdict(lambda: {"n_gt": 0, "missed": False})
    n_fp = 0
    for row in rows:
        gt = gt_of(row)
        p = [x for x in preds.get(row["name"], []) if x["conf"] >= conf]
        key = row["cell_id"] or row["group"]
        grp[key]["n_gt"] += len(gt)
        up, ug = match_predictions(p, gt, iou_wp)
        for i in up:
            stats[p[i]["cls"]]["tp"] += 1
        for gi, gb in enumerate(gt):
            if gi not in ug:
                stats[gb["cls"]]["fn"] += 1
                grp[key]["missed"] = True
        for pi, pb in enumerate(p):
            if pi not in up:
                stats[pb["cls"]]["fp"] += 1
        n_fp += len(p) - len(up)
    per = {}
    for c in range(3):
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        per[CLS_NAME[c]] = {"tp": tp, "fp": fp, "fn": fn,
                            "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    dg = {k: v for k, v in grp.items() if v["n_gt"] > 0}
    nm = sum(1 for v in dg.values() if v["missed"])
    return {"per_class": per, "fp_per_1000img": round(n_fp / len(rows) * 1000, 1),
            "group_miss": round(nm / len(dg), 4), "n_missed": nm, "n_groups": len(dg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="reports/oss_bench/preds_yolo_test.json")
    ap.add_argument("--b", default="reports/oss_bench/preds_yolo_2112.json")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--fuse-iou", type=float, default=0.5)
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    A = json.load(open(args.a, encoding="utf-8"))
    B = json.load(open(args.b, encoding="utf-8"))
    rows = [r for r in ds.load_manifest(args.manifest) if r["split"] == args.split]

    res = {}
    for name, fn in (("nms", fuse_nms), ("wbf", fuse_wbf)):
        fused = {r["name"]: fn(A.get(r["name"], []), B.get(r["name"], []),
                               args.conf, args.fuse_iou) for r in rows}
        s = score(rows, fused, args.conf, args.iou)
        res[name] = s
        print("[%s] 组漏检=%.2f%% (%d/%d) FP/千张=%.1f" % (
            name, s["group_miss"] * 100, s["n_missed"], s["n_groups"], s["fp_per_1000img"]))
        for c, v in s["per_class"].items():
            print("   %-7s TP=%-4d FP=%-4d FN=%-4d R=%s" % (c, v["tp"], v["fp"], v["fn"], v["recall"]))

    base = score(rows, A, args.conf, args.iou)
    print("[1920 单尺度] 组漏检=%.2f%% FP/千张=%.1f" % (base["group_miss"] * 100, base["fp_per_1000img"]))
    res["single_1920"] = base
    with open(os.path.join(ROOT, "reports", "oss_bench", "multiscale_fusion.json"),
              "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("已写: reports/oss_bench/multiscale_fusion.json")


if __name__ == "__main__":
    main()
