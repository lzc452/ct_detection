# -*- coding: utf-8 -*-
"""阈值模拟：抽样跑V4，统计损伤类置信度分布，推算不同导出阈值的收割量。"""
import os, glob, time
import numpy as np, cv2
from ultralytics import YOLO

ROOT = r'D:\CT Detection'
SRC = os.path.join(ROOT, '积攒')
SKIP_DIRS = [os.path.join(ROOT, '收割结果_积攒', '极片损伤'),
             os.path.join(ROOT, '收割结果_积攒', '极片损伤_低置信备查')]

imgs = sorted(glob.glob(os.path.join(SRC, '*', '正面', '*.jpg')))
if not imgs:
    imgs = sorted(glob.glob(os.path.join(SRC, '*', '*', '正面', '*.jpg')))
sample = imgs[::75]  # 每75张抽1张 → ~5100张
print('总图 %d，抽样 %d 张' % (len(imgs), len(sample)))

skip = set()
for sd in SKIP_DIRS:
    if os.path.isdir(sd):
        skip.update(os.listdir(sd))

m = YOLO(os.path.join(ROOT, 'models', '启用', '20261019_V4', 'best.pt'))
buckets = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
hits = {b: 0 for b in buckets}      # 有损伤>=b 的图数
hits_noskip = {b: 0 for b in buckets}
t0 = time.time()
B = 16
for i in range(0, len(sample), B):
    chunk = sample[i:i+B]
    arrs = [cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR) for p in chunk]
    arrs = [a if a is not None else np.zeros((16, 16, 3), np.uint8) for a in arrs]
    rs = m.predict(arrs, imgsz=1920, conf=0.25, verbose=False)
    for p, r in zip(chunk, rs):
        dconfs = [float(b.conf) for b in r.boxes if int(b.cls) == 0] if r.boxes is not None else []
        if not dconfs:
            continue
        top = max(dconfs)
        group = os.path.basename(os.path.dirname(os.path.dirname(p)))
        out_name = '%s_%s' % (group, os.path.basename(p))
        for b in buckets:
            if top >= b:
                hits[b] += 1
                if out_name not in skip:
                    hits_noskip[b] += 1
    if (i // B) % 40 == 0:
        print('  抽样进度 %d/%d (%.0fs)' % (i + len(chunk), len(sample), time.time() - t0))

ratio = len(imgs) / len(sample)
print('\n===== 模拟结果（损伤类，按图计）=====')
print('阈值   抽样命中  推算全量  剔除已核老图后')
for b in buckets:
    print('%.2f   %6d    %8d    %8d' % (b, hits[b], round(hits[b] * ratio), round(hits_noskip[b] * ratio)))
print('\n用时 %.0fs，抽样的实际损伤图可再人工复核确认质量' % (time.time() - t0))
