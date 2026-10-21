# -*- coding: utf-8 -*-
"""
旋转增强回归测试（#1）：验证"框跟着像素走"，而不是转反方向。

做法：纯黑图上画一个偏心白块 → 旋转 → 从像素反算白块真实外接框 → 与 rot() 返回的
YOLO 框比 IoU。旧实现（手工套 [[cos,-sin],[sin,cos]]）在 ±3° 时 IoU 只有 0.1~0.3，
修复后应 ≈1.0。

用法：runtime\\python.exe scripts/test_augment_rot.py
返回码：0 全通过；1 有失败。
"""
import os
import sys
import math

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from augment_final import rot   # noqa: E402  （模块级只算路径，可直接导入）


def iou(a, b):
    """a/b = [x1,y1,x2,y2] 像素坐标"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def yolo_to_xyxy(box, w, h):
    c, x, y, bw, bh = box
    return [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]


def rot_old(img, boxes, angle):
    """旧实现（v1_manual），仅用于在报告里对照，不参与断言。"""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img2 = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    rad = math.radians(angle)
    cx, cy = w / 2, h / 2
    sa, ca = math.sin(rad), math.cos(rad)
    out = []
    for c, x, y, bw, bh in boxes:
        x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
        x2, y2 = (x + bw / 2) * w, (y + bh / 2) * h
        pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        npts = [((px - cx) * ca - (py - cy) * sa + cx,
                 (px - cx) * sa + (py - cy) * ca + cy) for px, py in pts]
        xs = [p[0] for p in npts]
        ys = [p[1] for p in npts]
        nx1, ny1 = max(0, min(xs)), max(0, min(ys))
        nx2, ny2 = min(w, max(xs)), min(h, max(ys))
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        out.append([c, (nx1 + nx2) / 2 / w, (ny1 + ny2) / 2 / h,
                    (nx2 - nx1) / w, (ny2 - ny1) / h])
    return img2, out


def pixel_box(img):
    ys, xs = np.where(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 127)
    if not len(xs):
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def case_follow_pixels():
    """核心用例：偏心白块，±3°/±5°/15° 都应 IoU>0.85（旧实现 0.1~0.3）"""
    H, W = 480, 640
    img = np.zeros((H, W, 3), np.uint8)
    cv2.rectangle(img, (520, 60), (600, 140), (255, 255, 255), -1)   # 80x80，偏心 560px
    boxes = [[0, 560 / W, 100 / H, 80 / W, 80 / H]]
    fails = []
    print('  angle   新实现IoU   旧实现IoU')
    for ang in (-15, -5, -3, 3, 5, 15):
        im2, bx = rot(img, boxes, ang)
        _, bx_old = rot_old(img, boxes, ang)
        real = pixel_box(im2)
        if real is None:
            fails.append(f'angle={ang}: 白块被转出画布，说明变换矩阵变了')
            continue
        if not bx:
            fails.append(f'angle={ang}: 框被丢弃但像素仍在（假阴性）')
            continue
        new_iou = iou(yolo_to_xyxy(bx[0], W, H), real)
        old_iou = iou(yolo_to_xyxy(bx_old[0], W, H), real) if bx_old else 0.0
        print(f'  {ang:+4d}°    {new_iou:.3f}      {old_iou:.3f}')
        if new_iou < 0.85:
            fails.append(f'angle={ang}: 新实现 IoU={new_iou:.3f} < 0.85')
    return fails


def case_drop_outside():
    """整框转出画布的框必须被丢掉（不能留贴边假框）"""
    H, W = 480, 640
    img = np.zeros((H, W, 3), np.uint8)
    cv2.rectangle(img, (0, 0), (40, 40), (255, 255, 255), -1)
    boxes = [[0, 20 / W, 20 / H, 40 / W, 40 / H]]
    # 绕中心转 180°，左上角的小块跑到右下角，仍在画布内 → 应保留且 IoU 高
    im2, bx = rot(img, boxes, 180)
    fails = []
    real = pixel_box(im2)
    if bx and real is not None:
        if iou(yolo_to_xyxy(bx[0], W, H), real) < 0.85:
            fails.append('180° 后框位置不对')
    else:
        fails.append('180° 后框被误丢')
    # 人造越界：把框直接放在画布外
    img3 = np.zeros((H, W, 3), np.uint8)
    boxes_out = [[0, -0.2, -0.2, 0.1, 0.1]]
    _, bx_out = rot(img3, boxes_out, 0)
    if bx_out:
        fails.append(f'完全在图外的框没被丢弃：{bx_out}')
    return fails


def case_range():
    """输出必须是 [0,1] 归一化且宽高为正"""
    rng = np.random.RandomState(0)
    img = np.zeros((480, 640, 3), np.uint8)
    fails = []
    for _ in range(200):
        x, y = rng.uniform(0.05, 0.95, 2)
        bw, bh = rng.uniform(0.02, 0.3, 2)
        boxes = [[int(rng.randint(0, 3)), x, y, bw, bh]]
        _, out = rot(img, boxes, float(rng.uniform(-15, 15)))
        for c, bx, by, bbw, bbh in out:
            vals = (bx, by, bbw, bbh)
            if any(v < 0 or v > 1 for v in vals) or bbw <= 0 or bbh <= 0:
                fails.append(f'越界输出：{vals}')
                break
    return fails


def main():
    print('[#1 旋转回归] 白块跟像素走')
    all_fails = []
    for name, fn in (('follow_pixels', case_follow_pixels),
                     ('drop_outside', case_drop_outside),
                     ('range', case_range)):
        fails = fn()
        print(f'  -> {name}: {"OK" if not fails else "FAIL"}')
        all_fails += [f'{name}: {f}' for f in fails]
    if all_fails:
        print('\n[FAIL] %d 项' % len(all_fails))
        for f in all_fails:
            print('  -', f)
        return 1
    print('\n[PASS] 旋转标签与像素一致')
    return 0


if __name__ == '__main__':
    sys.exit(main())
