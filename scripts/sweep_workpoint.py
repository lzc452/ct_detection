# -*- coding: utf-8 -*-
"""
工作点扫描：同一模型、同一冻结测试集，扫多个 conf，输出 P/R/F1/FP千张/组级漏检

为什么需要：
`eval_report.py` 的层 2/3 固定在 conf=0.25（产线口径）。但不同模型的最优工作点不一样——
分辨率提高后 TP 和 FP 会一起涨，只看 conf=0.25 的单点会把"更敏感"误读成"更差"。

注意：`scripts/sweep_conf.py` 是旧脚本（硬编码 `D:\CT Detection` 与 V2 数据集，已失效），
本脚本取代它，全部走冻结清单 + core/roi.py。

用法：
  runtime\\python.exe scripts\\sweep_workpoint.py --weights runs\\detect\\CT_pro\\V51_ROI_1536\\weights\\best.pt ^
      --imgsz 1536 --roi --tag V51_ROI
"""
import os
import sys
import json
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import dataset_split as ds  # noqa: E402
import roi as R  # noqa: E402


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", default="1920")
    ap.add_argument("--roi", action="store_true")
    ap.add_argument("--confs", default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="0")
    ap.add_argument("--chunk", type=int, default=32)
    args = ap.parse_args()

    confs = sorted(float(x) for x in args.confs.split(","))
    rows = [r for r in ds.load_manifest(args.manifest) if r["split"] == args.split]
    if not rows:
        print("测试集为空")
        return 1

    from ultralytics import YOLO
    from ultralytics.utils.checks import check_imgsz
    from preprocess import resolve_imgsz
    sz = check_imgsz(resolve_imgsz(args.imgsz, for_what="predict"), stride=32, min_dim=2)
    m = YOLO(args.weights)
    print("权重 %s\nsplit=%s %d 张 | imgsz=%s | roi=%s | IoU=%s" % (
        args.weights, args.split, len(rows), sz, args.roi, args.iou))

    # 预处理：图 + GT 只算一次
    feeds, gts = [], []
    for r in rows:
        W0, H0 = [int(x) for x in r["size"].split("x")]
        if args.roi:
            img = cv2.imdecode(np.fromfile(os.path.join(ROOT, r["src_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
            roi_img, box = R.crop_roi(img)
            cw, ch = box[2] - box[0], box[3] - box[1]
            gt = []
            if r["label_json"]:
                lines, _, _ = R.labelme_boxes_in_roi(os.path.join(ROOT, r["label_json"]), box)
                for ln in lines:
                    cs = ln.split()
                    cls = int(cs[0])
                    xc, yc, bw, bh = (float(v) for v in cs[1:5])
                    gt.append({"cls": cls, "xyxy": [(xc - bw / 2) * cw, (yc - bh / 2) * ch,
                                                    (xc + bw / 2) * cw, (yc + bh / 2) * ch]})
            feeds.append(roi_img)
        else:
            gt = []
            if r["label_json"]:
                _, _, boxes = ds.json_to_yolo(os.path.join(ROOT, r["label_json"]))
                for cls, xc, yc, bw, bh in boxes:
                    gt.append({"cls": cls, "xyxy": [(xc - bw / 2) * W0, (yc - bh / 2) * H0,
                                                    (xc + bw / 2) * W0, (yc + bh / 2) * H0]})
            feeds.append(os.path.join(ROOT, r["src_path"]))
        gts.append(gt)

    low = min(confs) * 0.5
    preds = []
    t0 = time.time()
    for s in range(0, len(feeds), args.chunk):
        res = m.predict(feeds[s:s + args.chunk], imgsz=sz, conf=low, iou=0.7,
                        device=args.device, verbose=False, stream=False)
        for x in res:
            p = []
            if x.boxes is not None and len(x.boxes):
                xyxy = x.boxes.xyxy.cpu().numpy()
                cl = x.boxes.cls.cpu().numpy().astype(int)
                cf = x.boxes.conf.cpu().numpy()
                p = [{"cls": int(cl[k]), "conf": float(cf[k]),
                      "xyxy": [float(v) for v in xyxy[k]]} for k in range(len(cf))]
            preds.append(p)
    print("推理完成 %.1fs（低阈值 conf>=%.3f 共 %d 框）" % (
        time.time() - t0, low, sum(len(p) for p in preds)))

    points = []
    for cf in confs:
        st = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(3)}
        grp = {}
        n_img_with_gt = 0
        n_img_miss = 0
        for gt, pr, row in zip(gts, preds, rows):
            pp = [p for p in pr if p["conf"] >= cf]
            used = [False] * len(gt)
            for p in pp:
                best, bi = 0.0, -1
                for i, g in enumerate(gt):
                    if used[i] or g["cls"] != p["cls"]:
                        continue
                    v = iou(p["xyxy"], g["xyxy"])
                    if v > best:
                        best, bi = v, i
                if bi >= 0 and best >= args.iou:
                    used[bi] = True
                    st[p["cls"]]["tp"] += 1
                else:
                    st[p["cls"]]["fp"] += 1
            for i, g in enumerate(gt):
                if not used[i]:
                    st[g["cls"]]["fn"] += 1
            if gt:
                k = row["cell_id"] or row["group"]
                d = grp.setdefault(k, {"has_gt": 0, "miss": 0})
                d["has_gt"] = 1          # 分组去重：同一组的多个文件只算一个组
                if any(not used[i] for i in range(len(gt))):
                    d["miss"] = 1        # 组内任一缺陷漏检即记为漏检（OR，不看几张图）
                n_img_with_gt += 1
                if any(not used[i] for i in range(len(gt))):
                    n_img_miss += 1
        tp = sum(st[c]["tp"] for c in st)
        fp = sum(st[c]["fp"] for c in st)
        fn = sum(st[c]["fn"] for c in st)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        n_grp = sum(d["has_gt"] for d in grp.values())
        points.append({
            "conf": cf, "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
            "tp": tp, "fp": fp, "fn": fn,
            "fp_per_1000img": round(fp * 1000.0 / len(rows), 1),
            # 与 eval_report.py 完全同口径：分母是「有缺陷的分组数」
            "group_miss": round(sum(d["miss"] for d in grp.values()) / n_grp, 4) if n_grp else None,
            "n_defect_groups": n_grp,
            "n_missed_groups": sum(d["miss"] for d in grp.values()),
            # 参考口径：分母是「有缺陷的图片数」，一张图多组时会重复计数
            "img_miss": round(n_img_miss / n_img_with_gt, 4) if n_img_with_gt else None,
            "n_img_with_defect": n_img_with_gt,
            "per_class": {str(c): {
                "p": round(st[c]["tp"] / max(1, st[c]["tp"] + st[c]["fp"]), 4),
                "r": round(st[c]["tp"] / max(1, st[c]["tp"] + st[c]["fn"]), 4)} for c in st},
        })

    print("\n%-6s %-7s %-7s %-7s %-9s %-14s %-14s" % (
        "conf", "P", "R", "F1", "FP/千张", "组级漏检(组)", "图片漏检(参考)"))
    for o in points:
        print("%-6.2f %-7.3f %-7.3f %-7.3f %-9.1f %-14s %-14s" % (
            o["conf"], o["precision"], o["recall"], o["f1"], o["fp_per_1000img"],
            "-" if o["group_miss"] is None else "%.1f%% (%d/%d)" % (
                100 * o["group_miss"], o["n_missed_groups"], o["n_defect_groups"]),
            "-" if o["img_miss"] is None else "%.1f%%" % (100 * o["img_miss"])))

    tag = args.tag or os.path.splitext(os.path.basename(args.weights))[0]
    outp = os.path.join("reports", "eval", "sweep_%s.json" % tag)
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump({"weights": args.weights, "split": args.split, "imgsz": sz,
                   "roi": bool(args.roi), "iou": args.iou,
                   "manifest": args.manifest, "points": points}, f, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % outp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
