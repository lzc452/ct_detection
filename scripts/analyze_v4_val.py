# -*- coding: utf-8 -*-
# V4 错题本：大样本验证集(649) 漏检/误报导出，生产口径 conf>=0.30 + 位置大致重合
import csv
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO

ROOT = Path(r"D:\CT Detection")
VAL_IMG = ROOT / "datasets/CT_DATASET_FINAL_V4_AUG/val/images"
VAL_LAB = ROOT / "datasets/CT_DATASET_FINAL_V4_AUG/val/labels"
WEIGHTS = ROOT / "runs/detect/CT_pro/CT_DATASET_FINAL_V4_AUG_1016_1616/weights/best.pt"
OUT = ROOT / "复核_V4"
CONF = 0.30
CLS = {0: "damage", 1: "dark", 2: "light"}
CLS_ZH = {0: "损伤", 1: "暗斑", 2: "亮斑"}

def imread(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)
def imwrite(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img)[1].tofile(str(p))

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1) * max(0, y2-y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

def ctr(b): return ((b[0]+b[2])/2, (b[1]+b[3])/2)
def cin(b, pt): return b[0] <= pt[0] <= b[2] and b[1] <= pt[1] <= b[3]
def loose(pb, gb): return iou(pb, gb) >= 0.1 or cin(pb, ctr(gb)) or cin(gb, ctr(pb))

def draw(img, box, color, text):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(img, (x1, max(0, y1-th-10)), (x1+tw+6, y1), color, -1)
    cv2.putText(img, text, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

model = YOLO(str(WEIGHTS))
rows = []
stats = {}
for p in sorted(VAL_IMG.glob("*.jpg")):
    img = imread(p)
    if img is None: continue
    h, w = img.shape[:2]
    gt = []
    lab = VAL_LAB / (p.stem + ".txt")
    if lab.exists():
        for line in lab.read_text().splitlines():
            t = line.split()
            if len(t) < 5: continue
            c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
            gt.append({"cls": c, "xyxy": [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h],
                       "wh": [bw*w, bh*h]})
    r = model.predict(str(p), imgsz=1920, conf=CONF, verbose=False, device=0)[0]
    preds = []
    if r.boxes is not None:
        for b in r.boxes:
            preds.append({"cls": int(b.cls), "conf": float(b.conf),
                          "xyxy": [float(v) for v in b.xyxy[0]]})
    mg, mp = set(), set()
    for pi, pb in enumerate(preds):
        for gi, gb in enumerate(gt):
            if gi in mg or gb["cls"] != pb["cls"]: continue
            if loose(pb["xyxy"], gb["xyxy"]):
                mg.add(gi); mp.add(pi); break
    fn = [g for gi, g in enumerate(gt) if gi not in mg]
    fp = [pb for pi, pb in enumerate(preds) if pi not in mp]
    for gi, g in enumerate(gt):
        s = stats.setdefault(g["cls"], [0, 0, 0])
        s[1 if gi not in mg else 0] += 1
    for pi, pb in enumerate(preds):
        if pi not in mp: stats.setdefault(pb["cls"], [0, 0, 0])[2] += 1
    for g in fn:
        vis = img.copy()
        draw(vis, g["xyxy"], (0, 0, 255), f"MISS {CLS[g['cls']]}")
        imwrite(OUT / f"漏检_{CLS_ZH[g['cls']]}" / p.name, vis)
        imwrite(OUT / "原图_无框" / f"漏检_{CLS_ZH[g['cls']]}" / p.name, img)
        rows.append(["漏检", p.name, CLS_ZH[g["cls"]], "", *[round(v) for v in g["xyxy"]],
                     round(g["wh"][0]), round(g["wh"][1])])
    for pb in fp:
        vis = img.copy()
        draw(vis, pb["xyxy"], (255, 128, 0), f"FP {CLS[pb['cls']]} {pb['conf']:.2f}")
        imwrite(OUT / f"误报_{CLS_ZH[pb['cls']]}" / p.name, vis)
        imwrite(OUT / "原图_无框" / f"误报_{CLS_ZH[pb['cls']]}" / p.name, img)
        bw, bh = pb["xyxy"][2]-pb["xyxy"][0], pb["xyxy"][3]-pb["xyxy"][1]
        rows.append(["误报", p.name, CLS_ZH[pb["cls"]], round(pb["conf"], 3),
                     *[round(v) for v in pb["xyxy"]], round(bw), round(bh)])

with open(OUT / "错题清单.csv", "w", newline="", encoding="utf-8-sig") as f:
    wcsv = csv.writer(f)
    wcsv.writerow(["类型", "图片", "类别", "置信度", "x1", "y1", "x2", "y2", "宽px", "高px"])
    wcsv.writerows(rows)

print("===== V4 大样本错题本（conf=0.30 生产口径）=====")
for c in sorted(stats):
    tp, f_, fp_ = stats[c]
    P = tp/(tp+fp_)*100 if tp+fp_ else 0
    R = tp/(tp+f_)*100 if tp+f_ else 0
    print(f"{CLS_ZH[c]}: TP={tp} 漏检={f_} 误报={fp_} | P={P:.1f}% R={R:.1f}%")
