# -*- coding: utf-8 -*-
"""
开源模块复用验证（OSS reuse benchmark）：同一个考卷、同一套层2/3 口径，换推理后端跑 A/B。

设计约束（与 eval_report.py 对齐，避免"两把尺子"）：
1. 数据集只用冻结清单 reports/splits/v5_split_manifest.csv，不新建 split。
2. 匹配器复用 core/detection_metrics.match_predictions(max-iou-v2)，不另写一套。
3. 工作点固定 conf=0.25 / 匹配 IoU=0.5 / 模型内 NMS iou=0.7，与产线开关一致。
4. 预测坐标统一回到**原图坐标系**再打分；切片后端必须自己做坐标还原，脚本不替它擦屁股。

后端：
  yolo    基线，ultralytics 原生整图推理（生产口径）
  sahi    obss/sahi 切片推理（NMM 合并）
  slicer  roboflow/supervision `InferenceSlicer`（NMM 合并）
  ort     microsoft/onnxruntime 直接跑导出的 ONNX（CPU，只测本机 CPU 路径）

用法：
  references\\.venv\\Scripts\\python.exe scripts\\oss_bench.py --backend sahi --split test
"""
import os
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

from detection_metrics import match_predictions, MATCHING_VERSION  # noqa: E402
import dataset_split as ds  # noqa: E402

CLS_NAME = {0: "damage", 1: "dark", 2: "light"}
OUT_DIR = os.path.join(ROOT, "reports", "oss_bench")


# ----------------------------- 数据 -----------------------------
def load_rows(manifest, split):
    rows = [r for r in ds.load_manifest(manifest) if r["split"] == split]
    if not rows:
        raise ValueError("manifest 中 split=%s 无数据" % split)
    return rows


def imread_cn(path):
    """中文路径必须走 imdecode（AGENTS.md 硬约定）。"""
    import cv2
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("图片解码失败: " + path)
    return img


def gt_of(row):
    W, H = [int(x) for x in row["size"].split("x")]
    out = []
    if row["label_json"]:
        _, _, boxes = ds.json_to_yolo(os.path.join(ROOT, row["label_json"]))
        for cls, xc, yc, bw, bh in boxes:
            out.append({"cls": int(cls),
                        "xyxy": [(xc - bw / 2) * W, (yc - bh / 2) * H,
                                 (xc + bw / 2) * W, (yc + bh / 2) * H]})
    return out


# ----------------------------- 打分（与 eval_report 层2/3 同口径）-----------------------------
def score(rows, preds_by_name, conf_wp, iou_wp, matching=MATCHING_VERSION):
    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(3)}
    grp = defaultdict(lambda: {"n_gt": 0, "n_miss": 0, "missed": False})
    n_fp_total = 0
    n_pred_total = 0

    for row in rows:
        gt = gt_of(row)
        preds = [p for p in preds_by_name.get(row["name"], []) if p["conf"] >= conf_wp]
        n_pred_total += len(preds)

        key = row["cell_id"] or row["group"]
        g = grp[key]
        g["n_gt"] += len(gt)

        used_p, used_g = match_predictions(preds, gt, iou_wp, matching)
        for pi in used_p:
            stats[preds[pi]["cls"]]["tp"] += 1
        for gi, gb in enumerate(gt):
            if gi not in used_g:
                stats[gb["cls"]]["fn"] += 1
                g["missed"] = True
        for pi, pb in enumerate(preds):
            if pi not in used_p:
                stats[pb["cls"]]["fp"] += 1
        n_fp_total += len(preds) - len(used_p)

    per_cls = {}
    for c in range(3):
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        per_cls[CLS_NAME[c]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    defect_groups = {k: v for k, v in grp.items() if v["n_gt"] > 0}
    n_missed = sum(1 for v in defect_groups.values() if v["missed"])
    return {
        "per_class": per_cls,
        "fp_per_1000img": round(n_fp_total / max(1, len(rows)) * 1000.0, 1),
        "n_pred_total": n_pred_total,
        "group_level": {
            "n_defect_groups": len(defect_groups),
            "n_missed_groups": n_missed,
            "miss_rate": round(n_missed / len(defect_groups), 4) if defect_groups else None,
        },
        "n_images": len(rows),
    }


# ----------------------------- 后端 -----------------------------
def run_yolo(rows, args):
    """基线：整图推理，生产口径。"""
    from ultralytics import YOLO
    m = YOLO(args.weights)
    preds, lat = {}, []
    for s in range(0, len(rows), args.batch):
        batch = rows[s:s + args.batch]
        t0 = time.time()
        feeds = [os.path.join(ROOT, r["src_path"]) for r in batch]
        results = m.predict(feeds, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                            device=args.device, verbose=False)
        lat.append((time.time() - t0) * 1000.0 / len(batch))
        for r, row in zip(results, batch):
            lst = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for k in range(len(cf)):
                    lst.append({"cls": int(cls[k]), "conf": float(cf[k]),
                                "xyxy": [float(v) for v in xyxy[k]]})
            preds[row["name"]] = lst
        print("  yolo %d/%d" % (min(s + args.batch, len(rows)), len(rows)), end="\r")
    print()
    return preds, lat


def run_sahi(rows, args):
    """obss/sahi：切片推理，坐标由 SAHI 还原到原图。

    注意：SAHI 的 UltralyticsDetectionModel 内部写死 `img[:, :, ::-1]`（它假定入参是 RGB，
    要翻成 YOLO 期望的 BGR）。本项目的图是非严格灰度（三通道不相等），所以这里必须先转 RGB
    再喂，否则等于让模型吃反了的通道，检出会偏低。
    """
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    dm = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=args.weights,
        confidence_threshold=args.conf, image_size=args.imgsz, device=args.device)
    preds, lat = {}, []
    for i, row in enumerate(rows):
        img_bgr = imread_cn(os.path.join(ROOT, row["src_path"]))
        img_rgb = img_bgr[:, :, ::-1]
        t0 = time.time()
        res = get_sliced_prediction(
            img_rgb, dm,
            slice_height=args.slice_h, slice_width=args.slice_w,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            perform_standard_pred=False,
            postprocess_type="NMM", postprocess_match_metric="IOU",
            postprocess_match_threshold=0.5, batch_size=args.slice_batch, verbose=0)
        lat.append((time.time() - t0) * 1000.0)
        lst = []
        for op in res.object_prediction_list:
            lst.append({"cls": int(op.category.id), "conf": float(op.score.value),
                        "xyxy": [float(v) for v in op.bbox.to_xyxy()]})
        preds[row["name"]] = lst
        if (i + 1) % 20 == 0:
            print("  sahi %d/%d" % (i + 1, len(rows)), end="\r")
    print()
    return preds, lat


def run_slicer(rows, args):
    """roboflow/supervision：InferenceSlicer + NMM 合并。"""
    import supervision as sv
    from ultralytics import YOLO
    m = YOLO(args.weights)

    def callback(tile):
        r = m.predict(tile, imgsz=args.imgsz, conf=args.conf, iou=args.nms_iou,
                      device=args.device, verbose=False)[0]
        return sv.Detections.from_ultralytics(r)

    slicer = sv.InferenceSlicer(
        callback=callback, slice_wh=(args.slice_w, args.slice_h),
        overlap_wh=(int(args.slice_w * args.overlap), int(args.slice_h * args.overlap)),
        overlap_filter=sv.OverlapFilter.NON_MAX_MERGE, iou_threshold=0.5,
        thread_workers=1)
    preds, lat = {}, []
    for i, row in enumerate(rows):
        img = imread_cn(os.path.join(ROOT, row["src_path"]))
        t0 = time.time()
        det = slicer(img)
        lat.append((time.time() - t0) * 1000.0)
        lst = []
        if det is not None and len(det):
            for k in range(len(det.xyxy)):
                lst.append({"cls": int(det.class_id[k]), "conf": float(det.confidence[k]),
                            "xyxy": [float(v) for v in det.xyxy[k]]})
        preds[row["name"]] = lst
        if (i + 1) % 20 == 0:
            print("  slicer %d/%d" % (i + 1, len(rows)), end="\r")
    print()
    return preds, lat


def run_ort(rows, args):
    """onnxruntime：直接跑导出的 ONNX（CPU），自己解码 + NMS。"""
    import onnxruntime as ort
    providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    inp = sess.get_inputs()[0]
    H, W = args.ort_hw
    preds, lat = {}, []
    for i, row in enumerate(rows):
        img = imread_cn(os.path.join(ROOT, row["src_path"]))
        import cv2
        x = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        x = x[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        t0 = time.time()
        out = sess.run(None, {inp.name: np.ascontiguousarray(x)})[0]
        lat.append((time.time() - t0) * 1000.0)
        preds[row["name"]] = decode_ultralytics_out(out, W, H, img.shape[1], img.shape[0],
                                                   args.conf, args.nms_iou)
        if (i + 1) % 20 == 0:
            print("  ort %d/%d" % (i + 1, len(rows)), end="\r")
    print()
    return preds, lat


def decode_ultralytics_out(out, in_w, in_h, W0, H0, conf_th, nms_iou):
    """ONNX 输出 (1, 4+nc, N)：前 4 为 cx,cy,w,h（输入尺度像素），后 nc 为类分数。

    本脚本用的是**纯 resize**（非 letterbox），所以 x 方向与 y 方向缩放不同，
    必须分开反算；用 letterbox 的单一 scale 会把框压扁。
    """
    import cv2
    o = out[0] if out.ndim == 3 else out
    if o.shape[0] < o.shape[1]:          # (4+nc, N) -> (N, 4+nc)
        o = o.T
    scores = o[:, 4:]
    clss = scores.argmax(1)
    conf = scores[np.arange(len(scores)), clss]
    keep = conf >= conf_th
    if not keep.any():
        return []
    b, c, s = o[keep, :4], clss[keep], conf[keep]
    cx, cy, bw, bh = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    sx, sy = W0 / in_w, H0 / in_h
    x1 = (cx - bw / 2) * sx
    y1 = (cy - bh / 2) * sy
    w = bw * sx
    h = bh * sy

    out_lst = []
    # 逐类 NMS：不同类别的框互不抑制（与 ultralytics 的 agnostic=False 一致）
    for cls_id in np.unique(c):
        idx = np.where(c == cls_id)[0]
        boxes = np.stack([x1[idx], y1[idx], w[idx], h[idx]], 1).astype(np.float32)
        sc = s[idx].astype(np.float32)
        keep_i = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), conf_th, nms_iou)
        if len(keep_i) == 0:
            continue
        for j in np.array(keep_i).flatten():
            out_lst.append({"cls": int(cls_id), "conf": float(sc[j]),
                            "xyxy": [float(boxes[j][0]), float(boxes[j][1]),
                                     float(boxes[j][0] + boxes[j][2]),
                                     float(boxes[j][1] + boxes[j][3])]})
    return out_lst


# ----------------------------- 主流程 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["yolo", "sahi", "slicer", "ort"])
    ap.add_argument("--weights", default="runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.pt")
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--slice", type=int, default=960, help="切片边长（原图像素，未给 slice-h/w 时两向同值）")
    ap.add_argument("--slice-h", type=int, default=0, help="切片高（原图像素），0=用 --slice")
    ap.add_argument("--slice-w", type=int, default=0, help="切片宽（原图像素），0=用 --slice")
    ap.add_argument("--overlap", type=float, default=0.2, help="切片重叠比例")
    ap.add_argument("--slice-batch", type=int, default=8)
    ap.add_argument("--onnx", default="", help="ort 后端使用的 onnx 路径")
    ap.add_argument("--ort-hw", default="1088x1920", help="ONNX 输入尺寸 h x w")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 张（调试用，不准用于结论）")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dump-preds", action="store_true")
    args = ap.parse_args()

    args.ort_hw = tuple(int(v) for v in args.ort_hw.lower().split("x"))
    args.slice_h = args.slice_h or args.slice
    args.slice_w = args.slice_w or args.slice
    rows = load_rows(args.manifest, args.split)
    if args.limit:
        rows = rows[:args.limit]
    print("后端=%s 数据集=%s(%d) 权重=%s" % (args.backend, args.split, len(rows), args.weights))

    fn = {"yolo": run_yolo, "sahi": run_sahi, "slicer": run_slicer, "ort": run_ort}[args.backend]
    t0 = time.time()
    preds, lat = fn(rows, args)
    wall = time.time() - t0

    res = score(rows, preds, args.conf, args.iou)
    lat = np.array(lat, dtype=float)
    out = {
        "backend": args.backend,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": args.weights,
        "split": args.split,
        "n_images": len(rows),
        "workpoint": {"conf": args.conf, "match_iou": args.iou, "nms_iou": args.nms_iou,
                      "matching_version": MATCHING_VERSION},
        "config": {k: v for k, v in vars(args).items()
                   if k in ("imgsz", "slice", "slice_h", "slice_w", "overlap", "slice_batch", "batch",
                            "onnx", "ort_hw", "device", "limit")},
        "latency_ms": {"mean": round(float(lat.mean()), 1),
                       "p50": round(float(np.median(lat)), 1),
                       "p90": round(float(np.percentile(lat, 90)), 1)},
        "wall_s": round(wall, 1),
        "elapsed_s": round(time.time() - t0, 1),
    }
    out.update(res)

    if args.split == "test" and not args.limit:
        base_p = os.path.join(OUT_DIR, "yolo_test.json")
        if args.backend != "yolo" and os.path.isfile(base_p):
            with open(base_p, encoding="utf-8") as f:
                b = json.load(f)
            out["delta_vs_yolo"] = {
                "fp_per_1000img": round(res["fp_per_1000img"] - b["fp_per_1000img"], 1),
                "group_miss_pt": round((res["group_level"]["miss_rate"]
                                        - b["group_level"]["miss_rate"]) * 100, 2),
                "damage_recall_pt": round((res["per_class"]["damage"]["recall"]
                                           - b["per_class"]["damage"]["recall"]) * 100, 2),
                "latency_p50_ratio": round(float(np.median(lat))
                                           / b["latency_ms"]["p50"], 2),
            }

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.tag or (args.backend + "_" + args.split)
    path = os.path.join(OUT_DIR, tag + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n== %s ==" % tag)
    for c, v in res["per_class"].items():
        print("  %-7s TP=%-4d FP=%-4d FN=%-4d P=%-6s R=%-6s" % (
            c, v["tp"], v["fp"], v["fn"], v["precision"], v["recall"]))
    print("  FP/千张=%.1f | 组级漏检=%.2f%% (%d/%d) | 预测框=%d" % (
        res["fp_per_1000img"], (res["group_level"]["miss_rate"] or 0) * 100,
        res["group_level"]["n_missed_groups"], res["group_level"]["n_defect_groups"],
        res["n_pred_total"]))
    print("  延迟 mean/p50/p90 = %.1f / %.1f / %.1f ms" % (
        out["latency_ms"]["mean"], out["latency_ms"]["p50"], out["latency_ms"]["p90"]))
    if "delta_vs_yolo" in out:
        d = out["delta_vs_yolo"]
        print("  Δvs基线: FP%+.1f/千张 组漏检%+.2fpt 损伤召回%+.2fpt 延迟×%.2f" % (
            d["fp_per_1000img"], d["group_miss_pt"], d["damage_recall_pt"],
            d["latency_p50_ratio"]))

    if args.dump_preds:
        pp = os.path.join(OUT_DIR, "preds_%s.json" % tag)
        with open(pp, "w", encoding="utf-8") as f:
            json.dump(preds, f)
        print("  预测已存: %s" % pp)
    print("  报告: %s" % path)


if __name__ == "__main__":
    main()
