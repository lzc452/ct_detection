# -*- coding: utf-8 -*-
# 清洗版考卷亮斑漏检分析：哪些没抓到、多大、相邻切片是否抓到
import csv
from pathlib import Path
import numpy as np, cv2
from ultralytics import YOLO

ROOT = Path(r"D:\CT Detection")
EXAM = ROOT / "datasets" / "考卷_清洗版"
OUT = ROOT / "复核_考卷亮斑"
OUT.mkdir(exist_ok=True)
CLS = {0: "damage", 1: "dark", 2: "light"}

def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite(p, img):
    cv2.imencode(".jpg", img)[1].tofile(str(p))

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1) * max(0, y2-y1)
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i + 1e-9)

model = YOLO(str(ROOT / "runs/detect/CT_pro/V3/weights/best.pt"))
imgs = sorted(EXAM.joinpath("images").glob("*.jpg"))
fn_rows, all_light = [], 0
pred_cache = {}  # name -> light preds (for neighbor check)

for p in imgs:
    img = imread(p); h, w = img.shape[:2]
    gt = []
    lab = EXAM / "labels" / (p.stem + ".txt")
    if lab.exists():
        for line in lab.read_text().splitlines():
            t = line.split()
            if len(t) < 5: continue
            c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
            gt.append((c, [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h], bw*w, bh*h))
    rs = model.predict(str(p), imgsz=1920, conf=0.05, verbose=False, device=0)[0]
    preds = []
    if rs.boxes is not None:
        for b in rs.boxes:
            preds.append((int(b.cls), float(b.conf), [float(v) for v in b.xyxy[0]]))
    pred_cache[p.stem] = [(c, cf) for c, cf, _ in preds if c == 2 and cf >= 0.25]
    for c, box, bw, bh in gt:
        if c != 2: continue
        all_light += 1
        hit = any(pc == 2 and cf >= 0.25 and iou(pb, box) >= 0.5 for pc, cf, pb in preds)
        if not hit:
            fn_rows.append((p.name, round(bw,1), round(bh,1), box))
            vis = img.copy()
            x1,y1,x2,y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 3)
            imwrite(OUT / p.name, vis)

print(f"考卷亮斑真值共 {all_light} 个，漏检 {len(fn_rows)} 个（conf=0.25, IoU=0.5）")
for name, bw, bh, _ in fn_rows:
    stem = Path(name).stem
    # 相邻切片是否检出
    try:
        n = int(stem.split("_")[1])
        neigh = [f"{stem.split('_')[0]}_{n+d:04d}" for d in (-2,-1,1,2)]
        neigh_hit = sum(1 for nb in neigh if pred_cache.get(nb))
    except Exception:
        neigh_hit = -1
    print(f"  {name}: 框 {bw}x{bh} px | 相邻±2切片有检出: {neigh_hit}/4")
