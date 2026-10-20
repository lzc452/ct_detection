# -*- coding: utf-8 -*-
"""阈值-召回扫描：对 val 集逐图低阈值推理，扫多档 conf，算分类召回与每图误报"""
import os, glob, sys
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = r'D:\CT Detection'
VAL_IMG = os.path.join(ROOT, r'datasets\CT_DATASET_FINAL_V2_AUG\val\images')
VAL_LBL = os.path.join(ROOT, r'datasets\CT_DATASET_FINAL_V2_AUG\val\labels')
THRESHOLDS = [0.35, 0.25, 0.15, 0.10, 0.05]
IOU_MATCH = 0.5

def load_gt(lbl_path, W, H):
    boxes = []  # (cls, x1, y1, x2, y2)
    if not os.path.exists(lbl_path):
        return boxes
    for line in open(lbl_path, encoding='utf-8'):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
        x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
        boxes.append((c, x1, y1, x2, y2))
    return boxes

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0

def evaluate(model_path):
    m = YOLO(model_path)
    names = [m.names[i] for i in sorted(m.names)]
    nc = len(names)
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    imgs = sorted(p for p in glob.glob(os.path.join(VAL_IMG, '*.*'))
                  if os.path.splitext(p)[1].lower() in exts)
    # 逐图收集 conf>=0.01 的预测
    all_preds = {}
    all_gt = {}
    for p in imgs:
        im = cv2.imread(p)
        H, W = im.shape[:2]
        stem = os.path.splitext(os.path.basename(p))[0]
        all_gt[stem] = load_gt(os.path.join(VAL_LBL, stem + '.txt'), W, H)
        r = m.predict(p, imgsz=1920, conf=0.01, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                preds.append((int(b.cls), float(b.conf), *map(float, b.xyxy[0])))
        all_preds[stem] = preds
    print('\n===== %s =====' % model_path)
    print('类别顺序: %s' % names)
    header = 'conf  | ' + ' | '.join('%-14s' % n for n in names) + '| 每图误报'
    print(header)
    for thr in THRESHOLDS:
        tp = [0]*nc; fn = [0]*nc; fp_total = 0
        for stem in all_gt:
            gts = all_gt[stem]
            preds = [x for x in all_preds[stem] if x[1] >= thr]
            matched = [False]*len(gts)
            # 按置信度从高到低贪心匹配
            for c, conf, x1, y1, x2, y2 in sorted(preds, key=lambda t: -t[1]):
                best_iou, best_j = 0, -1
                for j, (gc, gx1, gy1, gx2, gy2) in enumerate(gts):
                    if matched[j] or gc != c:
                        continue
                    v = iou((x1, y1, x2, y2), (gx1, gy1, gx2, gy2))
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= IOU_MATCH:
                    matched[best_j] = True
                    tp[c] += 1
                else:
                    fp_total += 1
            for j, (gc, *_r) in enumerate(gts):
                if not matched[j]:
                    fn[gc] += 1
        cells = []
        for c in range(nc):
            d = tp[c] + fn[c]
            cells.append('%2d/%-2d=%5.1f%%' % (tp[c], d, 100.0*tp[c]/d if d else 0))
        print('%.2f | ' % thr + ' | '.join('%-14s' % s for s in cells) + '| %.2f' % (fp_total/len(all_gt)))

if __name__ == '__main__':
    evaluate(os.path.join(ROOT, r'models\启用\20261013\best.pt'))
    evaluate(os.path.join(ROOT, r'runs\detect\CT_pro\v2mild\weights\best.pt'))
