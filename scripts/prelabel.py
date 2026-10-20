# -*- coding: utf-8 -*-
"""为收割结果生成 X-AnyLabeling 预标注 json（人工只需核对修正）"""
import os, glob, json
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = r'D:\CT Detection'
SRC = os.path.join(ROOT, '收割结果_积攒')
MODEL = os.path.join(ROOT, 'models', '启用', '20261014_v2mild', 'best.pt')
CONF = 0.5
NAMES = {0: 'damage', 1: 'dark', 2: 'light'}

def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def make_json(name, W, H, dets):
    shapes = []
    for c, cf, (x1, y1, x2, y2) in dets:
        shapes.append({
            "label": str(c), "score": None,
            "points": [[float(x1), float(y1)], [float(x2), float(y1)],
                       [float(x2), float(y2)], [float(x1), float(y2)]],
            "group_id": None, "description": NAMES[c], "difficult": False,
            "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []
        })
    return {"version": "4.0.6", "flags": {}, "checked": False, "shapes": shapes,
            "imagePath": name, "imageData": None, "imageHeight": H, "imageWidth": W}

def main():
    m = YOLO(MODEL)
    imgs = sorted(glob.glob(os.path.join(SRC, '*', '*.jpg')))
    print('共 %d 张待生成预标注' % len(imgs))
    n = 0
    B = 16
    for i in range(0, len(imgs), B):
        chunk = imgs[i:i+B]
        arrs = [imread_cn(p) for p in chunk]
        rs = m.predict([a if a is not None else np.zeros((16, 16, 3), np.uint8) for a in arrs],
                       imgsz=1920, conf=CONF, verbose=False)
        for p, im, r in zip(chunk, arrs, rs):
            if im is None:
                continue
            H, W = im.shape[:2]
            dets = [(int(b.cls), float(b.conf), b.xyxy[0].tolist()) for b in r.boxes] if r.boxes is not None else []
            jp = os.path.splitext(p)[0] + '.json'
            with open(jp, 'w', encoding='utf-8') as f:
                json.dump(make_json(os.path.basename(p), W, H, dets), f, ensure_ascii=False, indent=1)
            n += 1
        if (i // B) % 20 == 0:
            print('进度 %d/%d' % (i + len(chunk), len(imgs)))
    print('完成：%d 个 json 已生成（与图片同目录同名）' % n)

if __name__ == '__main__':
    main()
