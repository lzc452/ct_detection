# -*- coding: utf-8 -*-
# 生产口径评测：conf>=0.30，位置大致重合即算检出（IoU>=0.1 或中心互落）
import sys
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO

ROOT = Path(r"D:\CT Detection")
CONF = 0.30
CLS = {0: "损伤", 1: "暗斑", 2: "亮斑"}

def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1) * max(0, y2-y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

def center_in(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]

def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)

def loose_match(pb, gb):
    return iou(pb, gb) >= 0.1 or center_in(pb, center(gb)) or center_in(gb, center(pb))

def evaluate(model_path, img_dir, lab_dir, tag):
    m = YOLO(model_path)
    stats = {c: [0, 0, 0] for c in CLS}  # tp, fn, fp
    for p in sorted(Path(img_dir).glob("*.jpg")):
        img = imread(p); h, w = img.shape[:2]
        gt = []
        lab = Path(lab_dir) / (p.stem + ".txt")
        if lab.exists():
            for line in lab.read_text().splitlines():
                t = line.split()
                if len(t) < 5: continue
                c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
                gt.append({"cls": c, "xyxy": [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]})
        r = m.predict(str(p), imgsz=1920, conf=CONF, verbose=False, device=0)[0]
        preds = []
        if r.boxes is not None:
            for b in r.boxes:
                preds.append({"cls": int(b.cls), "conf": float(b.conf),
                              "xyxy": [float(v) for v in b.xyxy[0]]})
        mg, mp = set(), set()
        for pi, pb in enumerate(preds):
            for gi, gb in enumerate(gt):
                if gi in mg or gb["cls"] != pb["cls"]: continue
                if loose_match(pb["xyxy"], gb["xyxy"]):
                    mg.add(gi); mp.add(pi); break
        for gi, g in enumerate(gt):
            stats[g["cls"]][1 if gi not in mg else 0] += 1
        for pi, pb in enumerate(preds):
            if pi not in mp: stats[pb["cls"]][2] += 1
    print(f"--- {tag} ---")
    for c in CLS:
        tp, fn, fp = stats[c]
        P = tp/(tp+fp)*100 if tp+fp else 0
        R = tp/(tp+fn)*100 if tp+fn else 0
        print(f"{CLS[c]}: TP={tp} 漏={fn} 误={fp} | 精确率={P:.1f}% 召回={R:.1f}%")

evaluate(str(ROOT/"runs/detect/CT_pro/CT_DATASET_FINAL_V4_AUG_1016_1616/weights/best.pt"),
         ROOT/"datasets/考卷_清洗版/images", ROOT/"datasets/考卷_清洗版/labels", "V4 @ 清洗考卷(196张)")
evaluate(str(ROOT/"runs/detect/CT_pro/CT_DATASET_FINAL_V4_AUG_1016_1616/weights/best.pt"),
         ROOT/"datasets/CT_DATASET_FINAL_V3_AUG/val/images", ROOT/"datasets/CT_DATASET_FINAL_V3_AUG/val/labels", "V4 @ 大样本(649张)")
evaluate(str(ROOT/"runs/detect/CT_pro/CT_DATASET_FINAL_V4_AUG_1016_1616/weights/best.pt"),
         ROOT/"datasets/考卷_清洗版/images", ROOT/"datasets/考卷_清洗版/labels", "V4 @ 大样本复核(196张)")
