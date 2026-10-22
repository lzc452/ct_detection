# -*- coding: utf-8 -*-
"""诊断：ROI 口径下「极片损伤」为什么掉召回。

对同一批失败样例，同时跑 V5（全图）与 V5.1-ROI，比较：
  - 与真值最高的 IoU
  - 该框的置信度
  - 预测框宽度 vs 真值宽度（看是否有系统性低估）

用法：
    ./runtime/python.exe scripts/diag_damage_recall.py
产出：
    reports/eval/damage_recall_diag.json（同时打印表格）
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import roi as R                      # noqa: E402
import dataset_split as ds           # noqa: E402

V5_W = "runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.pt"
ROI_W = "runs/detect/CT_pro/V51_ROI_1536/weights/best.pt"


def iou(a, b):
    xx0, yy0 = max(a[0], b[0]), max(a[1], b[1])
    xx1, yy1 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, xx1 - xx0), max(0.0, yy1 - yy0)
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def preds_of(res):
    if res.boxes is None or not len(res.boxes):
        return []
    xy = res.boxes.xyxy.cpu().numpy()
    cl = res.boxes.cls.cpu().numpy().astype(int)
    cf = res.boxes.conf.cpu().numpy()
    return [(int(cl[i]), [float(v) for v in xy[i]], float(cf[i])) for i in range(len(cf))]


def best_damage(preds, gts, cls_id=0):
    """返回 (最高 IoU, 该框, 该框置信度)"""
    best, bbox, bcf = 0.0, None, 0.0
    for c, b, cf in preds:
        if c != cls_id:
            continue
        for gc, gb in gts:
            if gc != cls_id:
                continue
            v = iou(b, gb)
            if v > best:
                best, bbox, bcf = v, b, cf
    return best, bbox, bcf


def main():
    from ultralytics import YOLO
    rows = {r["name"]: r for r in ds.load_manifest("reports/splits/v5_split_manifest.csv")
            if r["split"] == "test"}
    names = ["0_0017", "0_0018", "0_0197", "0_0257", "0_0258", "0_0259"]
    sel = [rows[n] for n in names if n in rows]

    m5, mr = YOLO(V5_W), YOLO(ROI_W)
    full_paths, rois, gf, gr = [], [], [], []
    for r in sel:
        im = cv2.imdecode(np.fromfile(r["src_path"], np.uint8), cv2.IMREAD_COLOR)
        ri, bx = R.crop_roi(im)
        rois.append(ri)
        full_paths.append(r["src_path"])
        W0, H0 = [int(x) for x in r["size"].split("x")]
        cw, ch = bx[2] - bx[0], bx[3] - bx[1]
        a, b = [], []
        if r["label_json"]:
            lines, _, _ = R.labelme_boxes_in_roi(r["label_json"], bx)
            for ln in lines:
                cs = ln.split()
                c = int(cs[0])
                xc, yc, bw, bh = (float(v) for v in cs[1:5])
                a.append((c, [(xc - bw / 2) * cw, (yc - bh / 2) * ch,
                              (xc + bw / 2) * cw, (yc + bh / 2) * ch]))
            _, _, bb = ds.json_to_yolo(r["label_json"])
            for c, xc, yc, bw, bh in bb:
                b.append((c, [(xc - bw / 2) * W0, (yc - bh / 2) * H0,
                              (xc + bw / 2) * W0, (yc + bh / 2) * H0]))
        gr.append(a)
        gf.append(b)

    P5 = [preds_of(x) for x in m5.predict(full_paths, imgsz=1920, conf=0.01, iou=0.7,
                                          verbose=False)]
    PR = [preds_of(x) for x in mr.predict(rois, imgsz=1536, conf=0.01, iou=0.7,
                                          verbose=False)]

    out = []
    print("%-9s %-14s | %-24s | %-24s" % ("图片", "真值(ROI坐标)", "V5@1920", "ROI@1536"))
    print("%-9s %-14s | %-24s | %-24s" % ("", "", "宽 conf IoU", "宽 conf IoU"))
    for r, a, b, p5, pr in zip(sel, gr, gf, P5, PR):
        gts_d = [x for x in a if x[0] == 0]
        if not gts_d:
            continue
        gt = gts_d[0][1]
        gw = gt[2] - gt[0]
        v5i, v5b, v5c = best_damage(p5, b)
        ri, rb, rc = best_damage(pr, a)
        scale = 647.0 / (ROI_X_W := 647)     # 仅用于打印说明，两列宽度各自在本口径像素下
        rec = {
            "name": r["name"],
            "gt_w_roi_px": round(gw, 1),
            "v5": {"best_iou": round(v5i, 3), "conf": round(v5c, 5),
                   "pred_w_px": None if v5b is None else round(v5b[2] - v5b[0], 1)},
            "roi": {"best_iou": round(ri, 3), "conf": round(rc, 5),
                    "pred_w_px": None if rb is None else round(rb[2] - rb[0], 1)},
        }
        out.append(rec)
        print("%-9s %-14s | %-24s | %-24s" % (
            r["name"], "宽 %.1fpx" % gw,
            "none" if v5b is None else "%.1f %0.4f %.2f" % (v5b[2] - v5b[0], v5c, v5i),
            "none" if rb is None else "%.1f %.4f %.2f" % (rb[2] - rb[0], rc, ri)))

    p = os.path.join(ROOT, "reports", "eval", "damage_recall_diag.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"note": "conf 下限 0.01；宽度为各自口径下的像素",
                   "cases": out}, f, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
