# -*- coding: utf-8 -*-
"""
onnxruntime 等价性与延迟验证。

要回答两个问题，必须分开：
Q1 运行时是否"透明"——同一个输入张量，ORT 与 torch 的输出是否一致（H-5）。
   做法：把图先 resize 到 1920x1088，**同一张数组**分别喂给 ORT 和 torch（imgsz 恰好等于数组尺寸，
   ultralytics 不再二次缩放），再按 IoU 一对一比对框。这样差异只可能来自运行时。
Q2 整条链路（含预处理）换到 ORT 后，业务指标动多少——由 oss_bench.py --backend ort 回答。

用法：
  references\\.venv\\Scripts\\python.exe scripts\\oss_ort_check.py --n 60
"""
import os
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
import sys
import json
import time
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
from detection_metrics import iou_xyxy  # noqa: E402
import dataset_split as ds  # noqa: E402

H, W = 1088, 1920


def preprocess(img):
    import cv2
    x = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    x = x[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x)


def decode(out, conf_th=0.25, nms_iou=0.7):
    import cv2
    o = out[0] if out.ndim == 3 else out
    if o.shape[0] < o.shape[1]:
        o = o.T
    scores = o[:, 4:]
    clss = scores.argmax(1)
    conf = scores[np.arange(len(scores)), clss]
    keep = conf >= conf_th
    if not keep.any():
        return []
    b, c, s = o[keep, :4], clss[keep], conf[keep]
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    x1, y1 = cx - bw / 2, cy - bh / 2
    res = []
    for cid in np.unique(c):
        idx = np.where(c == cid)[0]
        boxes = np.stack([x1[idx], y1[idx], bw[idx], bh[idx]], 1).astype(np.float32)
        sc = s[idx].astype(np.float32)
        ki = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), conf_th, nms_iou)
        for j in np.array(ki).flatten():
            bb = boxes[j]
            res.append({"cls": int(cid), "conf": float(sc[j]),
                        "xyxy": [float(bb[0]), float(bb[1]),
                                 float(bb[0] + bb[2]), float(bb[1] + bb[3])]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.pt")
    ap.add_argument("--onnx", default="runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.onnx")
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    import cv2
    import onnxruntime as ort
    from ultralytics import YOLO

    rows = [r for r in ds.load_manifest(args.manifest) if r["split"] == "test"][:args.n]
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print("ORT 输入:", inp.name, inp.shape, inp.type)
    m = YOLO(args.weights)

    n_pred_ort = n_pred_torch = 0
    matched = 0
    exact = 0            # IoU>=0.99 的配对数
    ious = []
    t_ort, t_gpu, t_cpu = [], [], []

    for r in rows:
        img = cv2.imdecode(np.fromfile(r["src_path"], dtype=np.uint8), cv2.IMREAD_COLOR)
        x = preprocess(img)
        arr = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        t0 = time.time()
        out = sess.run(None, {inp.name: x})[0]
        t_ort.append((time.time() - t0) * 1000)
        p_ort = decode(out)

        r_gpu = m.predict(arr, imgsz=(H, W), conf=0.25, iou=0.7, device="0", verbose=False)[0]
        t_gpu.append(float(r_gpu.speed["inference"]))
        p_gpu = [{"cls": int(c), "conf": float(s), "xyxy": [float(v) for v in b]}
                 for c, s, b in zip(r_gpu.boxes.cls.cpu().numpy().astype(int),
                                    r_gpu.boxes.conf.cpu().numpy(),
                                    r_gpu.boxes.xyxy.cpu().numpy())] if len(r_gpu.boxes) else []

        r_cpu = m.predict(arr, imgsz=(H, W), conf=0.25, iou=0.7, device="cpu", verbose=False)[0]
        t_cpu.append(float(r_cpu.speed["inference"]))

        n_pred_ort += len(p_ort)
        n_pred_torch += len(p_gpu)
        used = set()
        for pb in sorted(p_gpu, key=lambda z: -z["conf"]):
            best, best_iou = -1, 0.0
            for j, pa in enumerate(p_ort):
                if j in used or pa["cls"] != pb["cls"]:
                    continue
                v = iou_xyxy(pb["xyxy"], pa["xyxy"])
                if v > best_iou:
                    best, best_iou = j, v
            if best >= 0 and best_iou >= 0.5:
                used.add(best)
                matched += 1
                ious.append(best_iou)
                if best_iou >= 0.99:
                    exact += 1

    agree = matched / max(1, n_pred_torch)
    out = {
        "n_images": len(rows),
        "n_pred_ort": n_pred_ort, "n_pred_torch": n_pred_torch,
        "matched_iou50": matched, "agree_rate": round(agree, 4),
        "exact_iou99": exact, "exact_rate_of_torch": round(exact / max(1, n_pred_torch), 4),
        "iou_median": round(float(np.median(ious)), 4) if ious else None,
        "latency_ms": {
            "ort_cpu_p50": round(float(np.median(t_ort)), 1),
            "torch_gpu_infer_p50": round(float(np.median(t_gpu)), 1),
            "torch_cpu_infer_p50": round(float(np.median(t_cpu)), 1),
        },
        "onnx_mb": round(os.path.getsize(args.onnx) / 1e6, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    os.makedirs(os.path.join(ROOT, "reports", "oss_bench"), exist_ok=True)
    p = os.path.join(ROOT, "reports", "oss_bench", "ort_equivalence.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写:", p)


if __name__ == "__main__":
    main()
