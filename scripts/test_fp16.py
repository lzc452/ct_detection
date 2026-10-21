# -*- coding: utf-8 -*-
"""验证 FP16 + batch8 在 5060 上的可行性与一致性"""
import os, sys, time
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

import numpy as np
import cv2
import torch
from ultralytics import YOLO

MODEL = r'models\启用\20261019_V4\best.pt'
EXAM_IMG_DIR = r'datasets\考卷_清洗版\images'
INFER = 1920

def read_rgb(p):
    img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

files = []
for r, _, fs in os.walk(EXAM_IMG_DIR):
    for f in fs:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            files.append(os.path.join(r, f))
files = sorted(files)[:24]  # 取24张足够对比
print(f'test images: {len(files)}')

m = YOLO(MODEL, task='detect')
dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# warmup + 显存峰值测试 batch8 fp16
imgs = [read_rgb(p) for p in files[:8]]
try:
    m(imgs, imgsz=INFER, half=True, device=dev, verbose=False, conf=0.25)
    torch.cuda.reset_peak_memory_stats()
    m(imgs, imgsz=INFER, half=True, device=dev, verbose=False, conf=0.25)
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f'batch8 fp16 OK, peak VRAM: {peak:.2f} GB')
except RuntimeError as e:
    print(f'batch8 fp16 FAILED: {e}')
    sys.exit(1)

# 一致性对比 fp32 vs fp16（batch4）
def run(half, bs):
    outs = []
    for i in range(0, len(files), bs):
        chunk = [read_rgb(p) for p in files[i:i+bs]]
        rs = m(chunk, imgsz=INFER, half=half, device=dev, verbose=False, conf=0.25)
        for r in rs:
            n = len(r.boxes) if r.boxes is not None else 0
            if n:
                keep = r.boxes.conf >= 0.30
                b = r.boxes[keep]
                outs.append((len(b), sorted(round(float(c), 2) for c in b.conf.tolist())))
            else:
                outs.append((0, []))
    return outs

t0 = time.time(); a32 = run(False, 4); t32 = time.time() - t0
t0 = time.time(); a16 = run(True, 4);  t16_4 = time.time() - t0
t0 = time.time(); a16b8 = run(True, 8); t16_8 = time.time() - t0

mismatch = 0
for i, (x, y) in enumerate(zip(a32, a16)):
    if x[0] != y[0]:
        mismatch += 1
        print(f'  box-count diff #{i}: fp32={x[0]} fp16={y[0]}')
print(f'box-count mismatches: {mismatch}/{len(files)}')
print(f'time fp32/b4: {t32:.2f}s | fp16/b4: {t16_4:.2f}s | fp16/b8: {t16_8:.2f}s')
print(f'per-img fp32/b4: {t32/len(files)*1000:.0f}ms | fp16/b8: {t16_8/len(files)*1000:.0f}ms')
