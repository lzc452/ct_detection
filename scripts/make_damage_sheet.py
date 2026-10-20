# -*- coding: utf-8 -*-
# V4 损伤类漏检/误报放大拼图
from pathlib import Path
import numpy as np, cv2, csv

ROOT = Path(r"D:\CT Detection")
OUT = ROOT / "复核_V4"
VAL_IMG = ROOT / "datasets/CT_DATASET_FINAL_V4_AUG/val/images"

def imread(p): return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def make_sheet(kind):
    rows = []
    with open(OUT / "错题清单.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["类型"] == kind and r["类别"] == "损伤":
                rows.append(r)
    crops = []
    for r in rows:
        img = imread(VAL_IMG / r["图片"])
        if img is None: continue
        h, w = img.shape[:2]
        x1, y1, x2, y2 = int(float(r["x1"])), int(float(r["y1"])), int(float(r["x2"])), int(float(r["y2"]))
        mx = max(30, (x2-x1)*2); my = max(30, (y2-y1)*2)
        cx1, cy1 = max(0, int(x1-mx)), max(0, int(y1-my))
        cx2, cy2 = min(w, int(x2+mx)), min(h, int(y2+my))
        crop = img[cy1:cy2, cx1:cx2]
        scale = min(4.0, 300/max(1, crop.shape[0]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(crop, (int((x1-cx1)*scale), int((y1-cy1)*scale)),
                      (int((x2-cx1)*scale), int((y2-cy1)*scale)), (0,0,255), 2)
        bar = np.full((24, crop.shape[1], 3), 40, np.uint8)
        cv2.putText(bar, f"{r['图片']} {r['置信度']}", (4, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        crops.append(np.vstack([bar, crop]))
    if not crops: print(kind, "none"); return
    W = max(c.shape[1] for c in crops)
    crops = [cv2.copyMakeBorder(c, 0,0, 0, W-c.shape[1], cv2.BORDER_CONSTANT, value=(40,40,40)) for c in crops]
    cols = 4
    while len(crops) % cols: crops.append(np.full((60, W, 3), 40, np.uint8))
    grid = []
    for i in range(0, len(crops), cols):
        row = crops[i:i+cols]
        RH = max(c.shape[0] for c in row)
        row = [cv2.copyMakeBorder(c, 0, RH-c.shape[0], 0,0, cv2.BORDER_CONSTANT, value=(40,40,40)) for c in row]
        grid.append(np.hstack(row))
    sheet = np.vstack(grid)
    out = OUT / f"损伤{kind}_放大拼图.jpg"
    cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(str(out))
    print(out)

make_sheet("漏检")
make_sheet("误报")
