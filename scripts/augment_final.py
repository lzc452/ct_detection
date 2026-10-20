# -*- coding: utf-8 -*-
"""
终版数据集训练集温和离线增强（只增强 train，val 保持原样不动）
- 保护微弱亮/暗斑：亮度 ±10%、伽马 0.9~1.15、噪声 σ≤5，不做模糊、不做强预处理
- 正样本：原图 + 4 个变体；负样本：原图 + 1 个左右翻转
- 输出：datasets\\CT_DATASET_FINAL_AUG（已存在则拒绝覆盖）
用法：runtime\\python.exe scripts\\augment_final.py
"""
import os, sys, shutil, random
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_NAME = sys.argv[1] if len(sys.argv) > 1 else "CT_DATASET_FINAL"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else IN_NAME + "_AUG"
PREPROCESS = len(sys.argv) > 3 and sys.argv[3] == "pp"   # 第3个参数传 pp = 先做 combo_v2 强预处理
SRC = os.path.join(ROOT, "datasets", IN_NAME, "train")
OUT = os.path.join(ROOT, "datasets", OUT_NAME, "train")

if PREPROCESS:
    sys.path.insert(0, os.path.join(ROOT, "core"))
    from ct_preprocess import preprocess_ct_image

random.seed(42); np.random.seed(42)

def read_yolo(p):
    boxes = []
    if os.path.exists(p):
        for line in open(p):
            parts = line.split()
            if len(parts) == 5:
                boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes

def write_yolo(p, boxes):
    with open(p, "w") as f:
        for b in boxes:
            f.write("%d %.6f %.6f %.6f %.6f\n" % tuple(b))

def hflip(img, boxes):
    return cv2.flip(img, 1), [[c, 1 - x, y, w, h] for c, x, y, w, h in boxes]

def vflip(img, boxes):
    return cv2.flip(img, 0), [[c, x, 1 - y, w, h] for c, x, y, w, h in boxes]

def bc(img):    # 亮度对比度（温和）
    a = random.uniform(0.9, 1.1); b = random.uniform(-10, 10)
    return np.clip(img.astype(float) * a + b, 0, 255).astype(np.uint8)

def gamma(img): # 伽马（温和）
    g = random.uniform(0.9, 1.15)
    table = np.array([(i / 255.0) ** (1.0 / g) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def noise(img): # 轻噪声
    s = random.uniform(0, 5)
    n = np.random.normal(0, s, img.shape).astype(np.float32)
    return np.clip(img.astype(float) + n, 0, 255).astype(np.uint8)

def rot(img, boxes, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img2 = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    import math
    rad = math.radians(angle); cx, cy = w / 2, h / 2
    sa, ca = math.sin(rad), math.cos(rad)
    out = []
    for c, x, y, bw, bh in boxes:
        x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
        x2, y2 = (x + bw / 2) * w, (y + bh / 2) * h
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        npts = [((px - cx) * ca - (py - cy) * sa + cx,
                 (px - cx) * sa + (py - cy) * ca + cy) for px, py in pts]
        xs = [p[0] for p in npts]; ys = [p[1] for p in npts]
        nx1, ny1 = max(0, min(xs)), max(0, min(ys))
        nx2, ny2 = min(w, max(xs)), min(h, max(ys))
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        out.append([c, (nx1 + nx2) / 2 / w, (ny1 + ny2) / 2 / h,
                    (nx2 - nx1) / w, (ny2 - ny1) / h])
    return img2, out

# 正样本变体（温和）
POS_VARIANTS = [
    lambda im, bx: (bc(im), bx),
    lambda im, bx: hflip(gamma(im), bx),
    lambda im, bx: (noise(bc(im)), bx),
    lambda im, bx: rot(gamma(im), bx, random.choice([-3, 3])),
]

def main():
    if os.path.exists(os.path.dirname(OUT)) and not os.environ.get('AUG_RESUME'):
        print("已存在 %s，如需重建请先手动删除（续跑请设 AUG_RESUME=1）。" % os.path.dirname(OUT)); sys.exit(1)
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)

    files = sorted(f for f in os.listdir(os.path.join(SRC, "images"))
                   if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    import time as _time
    _budget = float(os.environ.get('AUG_BUDGET', '0'))
    _t0 = _time.time()
    n_in = n_out = 0
    for fname in files:
        stem, ext = os.path.splitext(fname)
        if os.path.exists(os.path.join(OUT, "images", fname)):  # 断点续跑：已完成跳过
            continue
        if _budget > 0 and _time.time() - _t0 > _budget:
            print('到点暂停（已续跑保存），重跑自动继续。本次新增 %d 张' % n_out)
            raise SystemExit(0)
        img = cv2.imread(os.path.join(SRC, "images", fname))
        if img is None:
            continue
        if PREPROCESS:
            img = preprocess_ct_image(img, mode='combo_v2')
        boxes = read_yolo(os.path.join(SRC, "labels", stem + ".txt"))
        n_in += 1
        variants = POS_VARIANTS if boxes else []          # 负样本不加像素变体
        # 原图
        cv2.imwrite(os.path.join(OUT, "images", fname), img)
        write_yolo(os.path.join(OUT, "labels", stem + ".txt"), boxes)
        n_out += 1
        for i, fn in enumerate(variants, 1):
            im2, bx2 = fn(img, boxes)
            cv2.imwrite(os.path.join(OUT, "images", "%s_a%d%s" % (stem, i, ext)), im2)
            write_yolo(os.path.join(OUT, "labels", "%s_a%d.txt" % (stem, i)), bx2)
            n_out += 1
        if not boxes:                                      # 负样本补一个左右翻转
            im2, bx2 = hflip(img, boxes)
            cv2.imwrite(os.path.join(OUT, "images", "%s_a1%s" % (stem, ext)), im2)
            write_yolo(os.path.join(OUT, "labels", "%s_a1.txt" % (stem,)), bx2)
            n_out += 1

    # val 原样复制 + data.yaml
    for split in ("val",):
        for kind in ("images", "labels"):
            src_d = os.path.join(ROOT, "datasets", IN_NAME, split, kind)
            dst_d = os.path.join(ROOT, "datasets", OUT_NAME, split, kind)
            os.makedirs(dst_d, exist_ok=True)
            for f in os.listdir(src_d):
                shutil.copy2(os.path.join(src_d, f), os.path.join(dst_d, f))
    out_root = os.path.join(ROOT, "datasets", OUT_NAME)
    with open(os.path.join(out_root, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# 终版数据集（训练集温和增强，验证集原图）\n")
        f.write("path: %s\n" % out_root.replace("\\", "/"))
        f.write("train: train/images\nval: val/images\n\nnc: 3\n")
        f.write("names:\n  0: damage\n  1: dark\n  2: light\n")

    n_val = len(os.listdir(os.path.join(out_root, "val", "images")))
    print("[完成] train: %d 张原图 -> %d 张（含变体） | val: %d 张（原图未动）" % (n_in, n_out, n_val))
    print("[输出] %s" % out_root)

if __name__ == "__main__":
    main()
