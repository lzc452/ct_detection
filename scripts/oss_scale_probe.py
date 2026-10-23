# -*- coding: utf-8 -*-
"""
尺度容忍窗口探针：V5 在"有效缩放倍率"变化时还剩多少检出能力。

背景：切片的收益完全依赖"模型能在不同尺度上工作"。若模型只在一个很窄的尺度上有效，
切片（把图切小 → 每块被迫上采样）必然负收益。本脚本先把这个前提量出来，再决定要不要跑完整 A/B。

有效缩放 = imgsz 内实际 resize 比例（ultralytics letterbox 的 r），不是 imgsz 本身。

用法：
  references\\.venv\\Scripts\\python.exe scripts\\oss_scale_probe.py --n 20
"""
import os
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
import sys
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
import dataset_split as ds  # noqa: E402


def letterbox_scale(h, w, imgsz):
    """ultralytics 方形 imgsz 的 letterbox 缩放因子（min 边对齐）。"""
    return min(imgsz / h, imgsz / w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.pt")
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20, help="取样张数（取含损伤的图）")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    from ultralytics import YOLO
    import cv2

    rows = [r for r in ds.load_manifest(args.manifest) if r["split"] == args.split]
    dam = [r for r in rows if int(r["n_damage"]) > 0][:args.n]
    m = YOLO(args.weights)

    print("样本：%d 张含损伤图；判据=conf>=0.25 的框数" % len(dam))
    print("\n[A] 整图（2129x1198）改变 imgsz")
    print("imgsz\tscale\t总框数\t损伤框\t平均耗时ms")
    full_res = {}
    for imgsz in (640, 960, 1280, 1600, 1920, 2112, 2560):
        tot = dmn = 0
        import time
        t = 0.0
        for r in dam:
            img = cv2.imdecode(np.fromfile(r["src_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            t0 = time.time()
            res = m.predict(img, imgsz=imgsz, conf=0.25, iou=0.7, device=args.device,
                            verbose=False)[0]
            t += time.time() - t0
            cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
            tot += len(cls)
            dmn += int((cls == 0).sum())
        sc = letterbox_scale(1198, 2129, imgsz)
        full_res[imgsz] = {"scale": round(sc, 3), "boxes": tot, "damage": dmn,
                           "ms": round(t / len(dam) * 1000, 1)}
        print("%-6d\t%.3f\t%-7d\t%-8d\t%.1f" % (imgsz, sc, tot, dmn, t / len(dam) * 1000))

    print("\n[B] 960 切片（模拟切片后单块）改变 imgsz")
    print("imgsz\tscale\t总框数\t损伤框\t备注")
    tile_res = {}
    for imgsz in (480, 720, 960, 1280, 1920):
        tot = dmn = 0
        for r in dam:
            img = cv2.imdecode(np.fromfile(r["src_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
            tile = img[:960, :960]
            res = m.predict(tile, imgsz=imgsz, conf=0.25, iou=0.7, device=args.device,
                            verbose=False)[0]
            cls = res.boxes.cls.cpu().numpy().astype(int) if res.boxes is not None else []
            tot += len(cls)
            dmn += int((cls == 0).sum())
        sc = letterbox_scale(960, 960, imgsz)
        note = "原生" if imgsz == 960 else ("上采样%.1fx" % sc if sc > 1 else "下采样%.2fx" % sc)
        tile_res[imgsz] = {"scale": round(sc, 3), "boxes": tot, "damage": dmn}
        print("%-6d\t%.3f\t%-7d\t%-8d\t%s" % (imgsz, sc, tot, dmn, note))

    out = {"weights": args.weights, "split": args.split, "n_images": len(dam),
           "full_image": full_res, "tile_960": tile_res}
    p = os.path.join(ROOT, "reports", "oss_bench", "scale_probe.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写: %s" % p)


if __name__ == "__main__":
    main()
