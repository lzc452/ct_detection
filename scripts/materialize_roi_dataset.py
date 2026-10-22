# -*- coding: utf-8 -*-
"""
按冻结清单落地 **ROI 裁剪版** 数据集（V5.1 的第二阶段）

与 `materialize_dataset.py` 的区别：
- 那个只做硬链接（零拷贝，不改变像素）
- 这个必须**真实裁剪并重新编码**（因为要改像素内容），输出到独立目录 `datasets_v51/`

母本仍然只读：只从 `问题电芯图片汇总/` 读，只往 `datasets_v51/` 写。

标注处理：
- 读原始 labelme JSON 的**像素坐标**，减去 ROI 偏移，再按 ROI 尺寸归一化
- 逐框校验必须完整落在 ROI 内（`core/roi.py --selfcheck` 已用 2945 个框验证边界，这里再兜一层）
- 生成 YOLO txt（cls xc yc bw bh，归一化到 ROI）

用法：
  runtime\\python.exe scripts\\materialize_roi_dataset.py --name CT_DATASET_V51_ROI
  runtime\\python.exe scripts\\materialize_roi_dataset.py --verify-only
"""
import os
import sys
import json
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import dataset_split as ds  # noqa: E402
import roi as R  # noqa: E402

CLS_OK = {0, 1, 2}


def read_cn(p):
    b = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(b, cv2.IMREAD_COLOR) if b.size else None


def write_cn(p, img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        buf.tofile(p)
    return ok


def boxes_in_roi(label_json, box):
    """复用 core/roi.py 的唯一实现，避免与评估脚本出现分歧"""
    return R.labelme_boxes_in_roi(os.path.join(ROOT, label_json), box) if label_json \
        else ([], 0, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--name", default="CT_DATASET_V51_ROI")
    ap.add_argument("--root", default="datasets_v51")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--margin-k", type=float, default=0.0,
                    help="ROI 外扩系数：各向外扩 k×损伤长边（单位 px 换算见图宽）。"
                         "0 = 原始 roi_box（历史行为）；1.5 = P0-2 带余量裁剪")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="只统计（不写图），用于确认边界是否安全")
    args = ap.parse_args()

    rows = ds.load_manifest(args.manifest)
    wants = [s.strip() for s in args.splits.split(",") if s.strip()]
    out = os.path.join(ROOT, args.root, args.name)
    # 必须在任何 crop_roi 之前设置：crop_roi 与 labelme_boxes_in_roi 共用这同一个 k，
    # 否则会出现「图按一种裁法、标签按另一种算」的静默错位（L-012）。
    R.set_margin_k(args.margin_k)
    print("ROI 余量 k=%s（0 = 原始 roi_box）" % args.margin_k)

    if not args.verify_only:
        if os.path.isdir(out) and args.force:
            shutil.rmtree(out)
        if os.path.isdir(out) and not args.force:
            print("已存在 %s；如需重建加 --force" % out)
            return 1

    stats = {"imgs": 0, "kept": 0, "dropped": 0, "invalid_roi": 0, "missing": 0, "neg": 0}
    dims = {}
    for sp in wants:
        img_dir = os.path.join(out, sp, "images")
        lab_dir = os.path.join(out, sp, "labels")
        if not args.verify_only:
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lab_dir, exist_ok=True)
        n = 0
        for r in rows:
            if r["split"] != sp:
                continue
            src = os.path.join(ROOT, r["src_path"])
            if not os.path.exists(src):
                stats["missing"] += 1
                continue
            img = read_cn(src)
            if img is None:
                stats["missing"] += 1
                continue
            roi_img, box = R.crop_roi(img)
            if not R.roi_looks_valid(roi_img):
                stats["invalid_roi"] += 1
            lines, kept, dropped = boxes_in_roi(r["label_json"], box) if r["label_json"] else ([], 0, 0)
            stats["kept"] += kept
            stats["dropped"] += dropped
            if not lines:
                stats["neg"] += 1
            h, w = roi_img.shape[:2]
            dims["%dx%d" % (w, h)] = dims.get("%dx%d" % (w, h), 0) + 1
            if not args.verify_only:
                write_cn(os.path.join(img_dir, r["name"] + ".jpg"), roi_img, args.quality)
                with open(os.path.join(lab_dir, r["name"] + ".txt"), "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
            n += 1
        stats["imgs"] += n
        print("  %-5s %4d 张" % (sp, n))

    print("\nROI 裁剪统计：")
    print("  图片 %d | 保留框 %d | 丢弃框(越界) %d | ROI 有效性失败 %d | 无缺陷图 %d | 缺失 %d" % (
        stats["imgs"], stats["kept"], stats["dropped"], stats["invalid_roi"], stats["neg"], stats["missing"]))
    print("  裁剪后尺寸分布（前 6）：")
    for k, v in sorted(dims.items(), key=lambda x: -x[1])[:6]:
        print("    %-12s %d 张" % (k, v))
    if stats["dropped"]:
        print("  [警告] 有 %d 个框被丢弃，说明 ROI 边界不够宽 —— 不要用这份数据训练！" % stats["dropped"])

    if args.verify_only:
        return 0

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# 由 scripts/materialize_roi_dataset.py 生成（母本只读，本目录为 ROI 裁剪副本）\n")
        f.write("# ROI 边界: x[%.4f, %.4f] y[%.4f, %.4f]（来自 core/roi.py）\n" % (
            R.ROI_X0, R.ROI_X1, R.ROI_Y0, R.ROI_Y1))
        f.write("# 余量 k=%s（各向外扩 %.1fpx @2129 宽；0 = 原始 ROI）\n" % (
            args.margin_k, args.margin_k * R.DAMAGE_LONG_SIDE_PX))
        f.write("# 冻结清单：%s\n" % args.manifest.replace("\\", "/"))
        f.write("path: %s\n" % out.replace("\\", "/"))
        f.write("train: train/images\nval: val/images\n\n")
        f.write("nc: 3\nnames:\n  0: damage\n  1: dark\n  2: light\n")

    total_mb = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(out) for f in fs) / 1e6
    print("\n落地完成：%s（%.0f MB）" % (out, total_mb))
    print("  data.yaml: %s" % yaml_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
