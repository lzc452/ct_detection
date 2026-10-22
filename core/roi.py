#coding:utf-8
# -*- coding: utf-8 -*-
"""
ROI 裁剪：只保留图像中真正含检测目标的区域

## 为什么需要（2026-10-21 实测）

本项目的原始 CT 图是一张 16:9 的"界面拼图"，不是纯电芯图：

    2129x1198 的典型构成
    ├─ 左 ~33%            纯黑/深灰空区
    ├─ 中 17%（x 0.35~0.60）  ★ 电芯正视投影（唯一含缺陷的区域）
    ├─ 左下角              Z/X 坐标系指示
    ├─ 左上角              "场景 坐标系 13.33mm" 文字
    ├─ 底部中间            5mm 尺度条
    └─ 右下角              另一个角度的电芯视图（**不含任何标注框**）

对冻结清单全部 2945 个标注框统计（`reports/splits/v5_split_manifest.csv`）：

    框的外包（归一化）  x ∈ [0.3459, 0.5998]   y ∈ [0.1431, 0.9196]
    外包面积占整图      19.72%
    加 2.5% 余量后      25.1%（裁剪后降 75% 无效像素）

也就是说：**训练时 75% 的像素和算力都花在与任务无关的黑边、文字和其它视图上**，
而真正的缺陷（亮斑原生仅 16.8px、对比度 3.3 灰度级、单像素 SNR<1）却挤在 17% 的
宽度里被一起缩小。

## 收益（几何上确定，非估计）

裁剪后 647x990（AR 0.654），放进方形画布：

| 画布 | 缩放率 | 亮斑尺寸 | 画布算力 | 相对 2112 全图 |
|---|---|---|---|---|
| 1024 | 1.034 | 17.4 px | 1.05 MP | −76% |
| 1216 | 1.228 | 20.6 px | 1.48 MP | −67% |
| **1536** | **1.551** | **26.1 px** | **2.36 MP** | **−47%** |
| 2112 | 2.134 | 35.8 px | 4.46 MP | 0% |

作为对照：V5 用 mosaic=1 训练时，亮斑实际只有 **7.6px**。

## 布局一致性（已验证）

两种原生分辨率（2129x1198 占 68%、1052x580~585 占 32%）的布局是**按比例一致**的：
归一化坐标相同，所以固定归一化边界对两组都成立。已用 `0_0001`（高分辨率）与
`0_0006`（低分辨率）验证裁剪后标注框仍完整落在 ROI 内且对准缺陷。

## 用法

    from roi import crop_roi, map_boxes_from_roi, roi_box, roi_looks_valid
    roi_img, (x0, y0, x1, y1) = crop_roi(img)
    boxes_orig = map_boxes_from_roi(res.boxes.xyxy.cpu().numpy(), (x0, y0, x1, y1))

自检：
    runtime\\python.exe core\\roi.py --selfcheck
"""
import os
import sys
import json
import argparse

import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================ ROI 边界 ============================
# 来自冻结清单 2945 个框的外包 + 2.5% 余量；改这里等于改数据分布，必须重跑评估
ROI_X0, ROI_X1 = 0.3209, 0.6248
ROI_Y0, ROI_Y1 = 0.1181, 0.9446

# ROI 有效性阈值（2026-10-21 全量标定，4592 张）
# 判别量用**灰度标准差**而不是亮度：生产数据亮度波动极大（ROI p99.5 从 153 到 255），
# 且非 ROI 区也可能有 255（文字标注、右下角视图），所以亮度判据完全不可靠。
#   实测 ROI std（4592 张全量）: min 21.94 / p1 28.85 / p50 56.55 / max 102.10
#   实测纯填充区 std（12 张抽样）: 9.7 ~ 18.8
# 取 20.0：全部 4592 张真实图通过，填充区被拦，两侧余量 >1.1 倍且无重叠。
VALID_STD = 20.0


def roi_box(W, H):
    """按归一化边界算像素坐标 (x0, y0, x1, y1)，并裁到图内"""
    x0 = int(round(max(0.0, min(1.0, ROI_X0)) * W))
    x1 = int(round(max(0.0, min(1.0, ROI_X1)) * W))
    y0 = int(round(max(0.0, min(1.0, ROI_Y0)) * H))
    y1 = int(round(max(0.0, min(1.0, ROI_Y1)) * H))
    x0, x1 = min(x0, W - 1), max(x1, x0 + 1)
    y0, y1 = min(y0, H - 1), max(y1, y0 + 1)
    return x0, y0, x1, y1


def crop_roi(img):
    """裁剪 ROI，返回 (roi_img, (x0,y0,x1,y1))"""
    H, W = img.shape[:2]
    box = roi_box(W, H)
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1], box


def map_boxes_to_roi(boxes, box):
    """原图坐标 -> ROI 坐标（推理前的真值变换 / 评估用）"""
    x0, y0 = box[0], box[1]
    b = np.asarray(boxes, dtype=np.float32).copy()
    if b.size:
        b[:, [0, 2]] -= x0
        b[:, [1, 3]] -= y0
    return b


def map_boxes_from_roi(boxes, box):
    """ROI 坐标 -> 原图坐标（推理后把框映射回去）"""
    x0, y0 = box[0], box[1]
    b = np.asarray(boxes, dtype=np.float32).copy()
    if b.size:
        b[:, [0, 2]] += x0
        b[:, [1, 3]] += y0
    return b


def roi_looks_valid(roi):
    """布局自检：ROI 内是否真的含有电芯结构。

    产线若更换成像设备/改变拼图布局，裁剪边界会失效并把缺陷裁到画面外。
    这里用灰度标准差做一次廉价检查（纯填充区 std<19，全部真实图 std>=21.9），
    失败说明 ROI 很可能落在了空白区，**必须报警人工检查，不能静默当成正常图**。
    """
    if roi is None or roi.size == 0:
        return False
    gray = roi if roi.ndim == 2 else cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    try:
        return float(gray.std()) >= VALID_STD
    except Exception:
        return False


def labelme_boxes_in_roi(label_json_path, box, cls_ok=(0, 1, 2)):
    """把 labelme JSON 的框变换到 ROI 的归一化 YOLO 坐标。

    返回 (lines, n_kept, n_dropped)：
      lines     ["cls xc yc bw bh", ...]，坐标已归一化到 ROI 尺寸
      n_kept    保留下来的框数
      n_dropped 因越界被丢弃的框数（**必须为 0**，否则说明 ROI 边界不够宽）

    只保留**完整落在 ROI 内**的框，不做截断——截断会改变框的真实尺寸，等于污染真值。
    `materialize_roi_dataset.py` 与 `eval_report.py --roi` 共用这一份实现，
    避免训练集和评估集的标注变换逻辑出现分歧。
    """
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    lines, kept, dropped = [], 0, 0
    try:
        with open(label_json_path, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return lines, 0, 0
    for s in d.get("shapes", []):
        try:
            c = int(s["label"])
        except (TypeError, ValueError):
            continue
        if c not in cls_ok:
            continue
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        bx0, bx1 = min(xs) - x0, max(xs) - x0
        by0, by1 = min(ys) - y0, max(ys) - y0
        if bx0 < -0.5 or by0 < -0.5 or bx1 > cw + 0.5 or by1 > ch + 0.5:
            dropped += 1
            continue
        bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)
        xc, yc = (bx0 + bx1) / 2.0 / cw, (by0 + by1) / 2.0 / ch
        lines.append("%d %.6f %.6f %.6f %.6f" % (c, xc, yc, bw / cw, bh / ch))
        kept += 1
    return lines, kept, dropped


# ============================ 自检 ============================
def selfcheck():
    sys.path.insert(0, os.path.join(ROOT, "core"))
    import dataset_split as ds
    import preprocess as pp

    rows = ds.load_manifest(os.path.join(ROOT, "reports", "splits", "v5_split_manifest.csv"))
    print("ROI 归一化边界: x[%.4f, %.4f] y[%.4f, %.4f]  面积占比 %.1f%%"
          % (ROI_X0, ROI_X1, ROI_Y0, ROI_Y1,
             100 * (ROI_X1 - ROI_X0) * (ROI_Y1 - ROI_Y0)))

    # 1) 抽查标注框是否都在 ROI 内
    CLS = {"0", "1", "2"}
    bad = 0
    nbox = 0
    for r in rows:
        if not r["label_json"]:
            continue
        W, H = (int(x) for x in r["size"].split("x"))
        b = roi_box(W, H)
        try:
            d = json.load(open(r["label_json"], encoding="utf-8"))
        except Exception:
            continue
        for s in d.get("shapes", []):
            if str(s.get("label")) not in CLS:
                continue
            xs = [p[0] for p in s["points"]]
            ys = [p[1] for p in s["points"]]
            if len(xs) < 2:
                continue
            nbox += 1
            if min(xs) < b[0] or max(xs) > b[2] or min(ys) < b[1] or max(ys) > b[3]:
                bad += 1
                if bad <= 5:
                    print("  [越界] %s 框 x[%.1f,%.1f] y[%.1f,%.1f] ROI %s"
                          % (r["name"], min(xs), max(xs), min(ys), max(ys), b))
    print("标注框抽查: %d 个，越界 %d 个 -> %s"
          % (nbox, bad, "全部在 ROI 内" if bad == 0 else "★有越界，边界需放宽"))

    # 2) 两种分辨率裁剪后的尺寸与画布对照
    print("\n裁剪后尺寸与画布规划：")
    print("%-12s %-12s %-8s %-10s %-10s" % ("原生", "ROI", "AR", "画布1536缩放", "亮斑@1536"))
    for size in ("2129x1198", "1052x585", "1052x582"):
        W, H = (int(x) for x in size.split("x"))
        x0, y0, x1, y1 = roi_box(W, H)
        cw, ch = x1 - x0, y1 - y0
        m = pp.plan_letterbox(cw, ch, 1536)
        print("%-12s %-12s %-8.3f %-10.3f %-10.1f" % (
            size, "%dx%d" % (cw, ch), cw / ch, m["scale"], 16.8 * m["scale"]))

    # 3) 真实图片的 ROI 有效性
    print("\nROI 有效性抽查：")
    import cv2
    cnt_ok = cnt = 0
    for r in rows[::400]:
        p = os.path.join(ROOT, r["src_path"])
        if not os.path.exists(p):
            continue
        img = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        roi, _ = crop_roi(img)
        cnt += 1
        ok = roi_looks_valid(roi)
        cnt_ok += int(ok)
        print("  %-10s %s -> %s" % (r["name"], r["size"],
                                    "有效" if ok else "★无效（应回退全图）"))
    print("小结: %d/%d 通过有效性检查" % (cnt_ok, cnt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    else:
        print(__doc__)
