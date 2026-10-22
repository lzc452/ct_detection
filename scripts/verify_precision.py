# -*- coding: utf-8 -*-
"""小样本真实GPU精度核验；FP32/FP16必须分两个独立进程运行，避免复用预测器污染dtype。
用法：runtime/python.exe -B scripts/verify_precision.py --weights <best.pt> --half 0 --out reports/system_audit/fp32_probe.json
再次运行 --half 1 并换输出名。只按冻结清单读8张图，不修改图片/标签/权重，不启动服务。
这是精度契约冒烟检查，不代替全量验收或生产P95基准。
"""
import os
import sys
import json
import csv
import time
import hashlib
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--half", type=int, choices=(0, 1), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise SystemExit("此项需GPU；不在CPU上虚报FP16核验通过")
    with open(ROOT / "reports/splits/v5_split_manifest.csv", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    selected = []
    for cls in ("damage", "dark", "light"):
        for high in (False, True):
            row = next(r for r in rows if int(r["n_" + cls]) > 0
                       and (int(r["size"].split("x")[0]) > 1500) == high)
            if row["name"] not in {r["name"] for r in selected}:
                selected.append(row)
    selected += [r for r in rows if r["is_neg"] == "1"][:2]
    inputs = []
    for row in selected:
        bgr = cv2.imdecode(np.fromfile(str(ROOT / row["src_path"]), dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("抽样解码失败: " + row["name"])
        # 保持当前Web的RGB numpy输入口径，颜色契约的单独修复不混入本对照。
        inputs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    model = YOLO(args.weights, task="detect")
    kw = dict(imgsz=1920, device=0, half=bool(args.half), conf=0.30, iou=0.7, max_det=300, verbose=False)
    for _ in range(3):
        model(inputs[0], **kw)
    dtype = str(next(model.predictor.model.model.parameters()).dtype)
    expected = "torch.float16" if args.half else "torch.float32"
    if dtype != expected:
        raise AssertionError("网络dtype不符: " + dtype + " != " + expected)
    timings, preds = [], []
    for repeat in range(2):
        for row, image in zip(selected, inputs):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(image, **kw)[0]
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000)
            if repeat == 1:
                preds.append(dict(name=row["name"], size=row["size"],
                                  xyxy=result.boxes.xyxy.cpu().tolist(),
                                  cls=result.boxes.cls.cpu().tolist(),
                                  conf=result.boxes.conf.cpu().tolist()))
    out = Path(args.out).resolve()
    if not out.is_relative_to(ROOT / "reports"):
        raise SystemExit("输出仅允许放在reports内")
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dict(schema="ct-precision-smoke/1", weights=args.weights,
                weights_md5=hashlib.md5(Path(args.weights).read_bytes()).hexdigest(),
                torch=torch.__version__, device=torch.cuda.get_device_name(0), half=bool(args.half),
                actual_dtype=dtype, config=kw, n_images=len(selected), n_timing_calls=len(timings),
                call_wall_ms=dict(p50=float(np.median(timings)), p95=float(np.percentile(timings, 95)), values=timings),
                predictions=preds,
                limitation="8图batch1冒烟，不能外推生产吞吐或P95，也未用此样本调整阈值")
    with out.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: data[k] for k in ("actual_dtype", "n_images", "n_timing_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
