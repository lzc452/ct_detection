# -*- coding: utf-8 -*-
"""
统一评估报告（#11）：三层口径一次跑完，输出可存档的 JSON + Markdown 表

层 1（标准 COCO 指标）：直接复用 ultralytics YOLO.val，避免自研 AP 口径
    产出 mAP50 / mAP50-95 / 每类 AP(50,50-95) / P / R
层 2（固定工作点）：conf=上线阈值、IoU>=0.5、同类、按 conf 降序贪心一对一
    产出每类 P / R —— 与产线实际阈值一致，可复现（不依赖模型输出顺序）
层 3（业务口径）：FP/千张、电芯/组级漏检率、宽松命中率（仅参考，禁止与 IoU0.5 并列比较）

不修改任何原始数据：评估用的临时数据集只做 **硬链接**（同卷 NTFS，零拷贝），
放在 .cache/ 下，可随时删除。

用法：
  runtime\\python.exe scripts\\eval_report.py --weights models\\启用\\20261019_V4\\best.pt ^
      --manifest reports\\splits\\v5_split_manifest.csv --split test --tag V4_v5test
"""
import os
import sys
import json
import time
import shutil
import hashlib
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import dataset_split as ds  # noqa: E402
from preprocess import resolve_imgsz  # noqa: E402
from ultralytics.utils.checks import check_imgsz  # noqa: E402

CLS_NAME = {0: "damage", 1: "dark", 2: "light"}
CACHE = os.path.join(ROOT, ".cache")


# ----------------------------- 数据准备 -----------------------------
def prepare_eval_dataset(manifest_rows, tag, verbose=True, roi=False):
    """把 manifest 指定的 split 物化成 ultralytics 可读的临时数据集。

    roi=False：硬链接原图（零拷贝），标注用原始归一化坐标
    roi=True ：真实裁剪 ROI 并重编码，标注经 core.roi.labelme_boxes_in_roi 变换
               （必须与训练集的裁剪口径完全一致，否则评估的是另一个分布）
    """
    out = os.path.join(CACHE, "eval_%s%s" % (tag, "_roi" if roi else ""))
    img_dir = os.path.join(out, "images")
    lab_dir = os.path.join(out, "labels")
    yaml_path = os.path.join(out, "data.yaml")
    if os.path.isdir(out) and os.path.exists(yaml_path) and \
            len(os.listdir(img_dir)) == len(manifest_rows):
        if verbose:
            print("复用已有临时数据集:", out)
        return yaml_path

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    dropped_total = 0
    for r in manifest_rows:
        src = os.path.join(ROOT, r["src_path"])
        stem = r["name"]
        dst = os.path.join(img_dir, stem + ".jpg")
        if roi:
            import cv2
            import numpy as np
            import roi as R
            img = cv2.imdecode(np.fromfile(src, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            roi_img, box = R.crop_roi(img)
            ok, buf = cv2.imencode(".jpg", roi_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if ok:
                buf.tofile(dst)
            lines, _, dropped = R.labelme_boxes_in_roi(
                os.path.join(ROOT, r["label_json"]), box) if r["label_json"] else ([], 0, 0)
            dropped_total += dropped
        else:
            if not os.path.exists(dst):
                try:
                    os.link(src, dst)                  # 硬链接：零拷贝
                except OSError:
                    try:
                        shutil.copy2(src, dst)         # 跨卷兜底（会占空间）
                    except shutil.SameFileError:
                        pass
            lines = []
            if r["label_json"]:
                W, H, boxes = ds.json_to_yolo(os.path.join(ROOT, r["label_json"]))
                for cls, xc, yc, bw, bh in boxes:
                    lines.append("%d %.6f %.6f %.6f %.6f" % (cls, xc, yc, bw, bh))
        with open(os.path.join(lab_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    if roi and dropped_total:
        print("[警告] ROI 评估集有 %d 个框越界被丢弃 —— 与训练集口径不一致，结果不可比！" % dropped_total)

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        # ultralytics 的 check_det_dataset 要求 train/val 键同时存在；
        # 这里两者指向同一目录，只为让 val() 跑起来，不会用于训练
        f.write("# 评估临时集（%s生成，源为问题电芯图片汇总）\n" % ("ROI 裁剪" if roi else "硬链接"))
        f.write("path: %s\ntrain: images\nval: images\n\nnc: 3\n"
                "names:\n  0: damage\n  1: dark\n  2: light\n" % out.replace("\\", "/"))
    if verbose:
        print("临时数据集就绪：%s（%d 张）" % (out, len(manifest_rows)))
    return yaml_path


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / ua if ua > 0 else 0.0


def center_in(box, pt):
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def loose_match(pb, gb):
    """仅业务参考口径（评审已判定不可作为主指标）"""
    return (iou_xyxy(pb, gb) >= 0.1 or center_in(pb, center(gb)) or center_in(gb, center(pb)))


# ----------------------------- 层 1 -----------------------------
def layer1_coco(weights, yaml_path, imgsz, device, verbose=True):
    from ultralytics import YOLO
    m = YOLO(weights)
    r = m.val(data=yaml_path, imgsz=imgsz, conf=0.001, iou=0.7, device=device,
              plots=False, verbose=False, workers=2)
    per_cls = {}
    idx = list(getattr(r.box, "ap_class_index", []))
    for k, ci in enumerate(idx):
        per_cls[CLS_NAME.get(int(ci), str(ci))] = {
            "ap50": float(r.box.ap50[k]),
            "ap": float(r.box.ap[k]),
            "p": float(r.box.p[k]),
            "r": float(r.box.r[k]),
        }
    out = {
        "mAP50": float(r.box.map50),
        "mAP50_95": float(r.box.map),
        "mAP75": float(r.box.map75),
        "mean_precision": float(r.box.mp),
        "mean_recall": float(r.box.mr),
        "per_class": per_cls,
        "speed_ms": {k: float(v) for k, v in r.speed.items()},
    }
    if verbose:
        print("层1 mAP50=%.4f mAP50-95=%.4f" % (out["mAP50"], out["mAP50_95"]))
        for c, v in per_cls.items():
            print("    %-7s AP50=%.3f AP=%.3f P=%.3f R=%.3f" % (c, v["ap50"], v["ap"], v["p"], v["r"]))
    return out


# ----------------------------- 层 2/3 -----------------------------
def layer23_workpoint(weights, rows, imgsz, conf_wp, iou_wp, device, chunk=32, verbose=True, roi_dir=None):
    """roi_dir 给定时，预测用该目录下的 ROI 裁剪图，GT 也用 ROI 坐标（与训练同口径）"""
    from ultralytics import YOLO
    import roi as R
    m = YOLO(weights)
    if roi_dir:
        paths = [os.path.join(roi_dir, r["name"] + ".jpg") for r in rows]
    else:
        paths = [os.path.join(ROOT, r["src_path"]) for r in rows]

    stats = {c: {"tp": 0, "fp": 0, "fn": 0} for c in range(3)}
    loose = {c: {"tp": 0, "fn": 0} for c in range(3)}
    n_fp_total = 0
    # 电芯/组级：组内有缺陷且至少一个缺陷未被检出
    grp = defaultdict(lambda: {"n_gt": 0, "n_miss": 0, "missed": False})
    t0 = time.time()

    for s in range(0, len(rows), chunk):
        batch_rows = rows[s:s + chunk]
        batch_paths = paths[s:s + chunk]
        results = m.predict(batch_paths, imgsz=imgsz, conf=conf_wp, iou=0.7, device=device,
                            verbose=False, stream=False)
        for r, row, pth in zip(results, batch_rows, batch_paths):
            gt = []
            if roi_dir:
                # ROI 坐标系：GT 由 core.roi 变换，尺寸取 ROI 尺寸
                W0, H0 = [int(x) for x in row["size"].split("x")]
                box = R.roi_box(W0, H0)
                cw, ch = box[2] - box[0], box[3] - box[1]
                if row["label_json"]:
                    lines, _, _ = R.labelme_boxes_in_roi(os.path.join(ROOT, row["label_json"]), box)
                    for ln in lines:
                        cs = ln.split()
                        cls = int(cs[0])
                        xc, yc, bw, bh = (float(v) for v in cs[1:5])
                        gt.append({"cls": cls,
                                   "xyxy": [(xc - bw / 2) * cw, (yc - bh / 2) * ch,
                                            (xc + bw / 2) * cw, (yc + bh / 2) * ch]})
            else:
                W, H = [int(x) for x in row["size"].split("x")]
                if row["label_json"]:
                    _, _, boxes = ds.json_to_yolo(os.path.join(ROOT, row["label_json"]))
                    for cls, xc, yc, bw, bh in boxes:
                        gt.append({"cls": cls,
                                   "xyxy": [(xc - bw / 2) * W, (yc - bh / 2) * H,
                                            (xc + bw / 2) * W, (yc + bh / 2) * H]})
            preds = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for k in range(len(cf)):
                    preds.append({"cls": int(cls[k]), "conf": float(cf[k]),
                                  "xyxy": [float(v) for v in xyxy[k]]})

            key = row["cell_id"] or row["group"]
            g = grp[key]
            g["n_gt"] += len(gt)

            # ---- 主口径：IoU>=iou_wp、同类、conf 降序贪心 ----
            used_p = set()
            used_g = set()
            for pi, pb in enumerate(sorted(range(len(preds)), key=lambda z: -preds[z]["conf"])):
                pass
            order_p = sorted(range(len(preds)), key=lambda z: -preds[z]["conf"])
            for pi in order_p:
                pb = preds[pi]
                for gi, gb in enumerate(gt):
                    if gi in used_g or gb["cls"] != pb["cls"]:
                        continue
                    if iou_xyxy(pb["xyxy"], gb["xyxy"]) >= iou_wp:
                        used_g.add(gi)
                        used_p.add(pi)
                        stats[pb["cls"]]["tp"] += 1
                        break
            for gi, gb in enumerate(gt):
                if gi not in used_g:
                    stats[gb["cls"]]["fn"] += 1
                    g["missed"] = True
            for pi, pb in enumerate(preds):
                if pi not in used_p:
                    stats[pb["cls"]]["fp"] += 1
            n_fp_total += len(preds) - len(used_p)

            # ---- 参考口径：宽松匹配（不参与决策）----
            mg = set()
            for pb in sorted(preds, key=lambda z: -z["conf"]):
                for gi, gb in enumerate(gt):
                    if gi in mg or gb["cls"] != pb["cls"]:
                        continue
                    if loose_match(pb["xyxy"], gb["xyxy"]):
                        mg.add(gi)
                        loose[pb["cls"]]["tp"] += 1
                        break
            for gi, gb in enumerate(gt):
                if gi not in mg:
                    loose[gb["cls"]]["fn"] += 1
        if verbose:
            done = min(s + chunk, len(rows))
            print("  工作点推理 %d/%d  (%.0fs)" % (done, len(rows), time.time() - t0), end="\r")
    if verbose:
        print()

    per_cls = {}
    for c in range(3):
        tp, fp, fn = stats[c]["tp"], stats[c]["fp"], stats[c]["fn"]
        per_cls[CLS_NAME[c]] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        }
    loose_cls = {}
    for c in range(3):
        tp, fn = loose[c]["tp"], loose[c]["fn"]
        loose_cls[CLS_NAME[c]] = {"tp": tp, "fn": fn,
                                  "recall": round(tp / (tp + fn), 4) if tp + fn else None}

    # 组级漏检率（cell 已知时用 cell，否则用近重复组近似）
    defect_groups = {k: v for k, v in grp.items() if v["n_gt"] > 0}
    n_missed = sum(1 for v in defect_groups.values() if v["missed"])
    by_cell = {k: v for k, v in defect_groups.items() if k and "-" in k and k.count("-") == 1}
    cell_miss = (sum(1 for v in by_cell.values() if v["missed"]) / len(by_cell)) if by_cell else None

    out = {
        "workpoint": {"conf": conf_wp, "iou": iou_wp, "nms_iou": 0.7,
                      "matching": "同类 + IoU>=%.2f + conf 降序贪心一对一" % iou_wp},
        "per_class": per_cls,
        "fp_per_1000img": round(n_fp_total / max(1, len(rows)) * 1000.0, 1),
        "group_level": {
            "n_defect_groups": len(defect_groups),
            "n_missed_groups": n_missed,
            "miss_rate": round(n_missed / len(defect_groups), 4) if defect_groups else None,
            "cell_level_miss_rate": round(cell_miss, 4) if cell_miss is not None else None,
            "cell_level_n": len(by_cell),
        },
        "loose_reference": {
            "note": "仅业务参考：IoU>=0.1 或中心互落；禁止与 IoU0.5 口径或 V3 数字并列比较",
            "per_class": loose_cls,
        },
        "n_images": len(rows),
    }
    if verbose:
        print("层2 工作点 conf=%.2f IoU=%.2f" % (conf_wp, iou_wp))
        for c, v in per_cls.items():
            print("    %-7s TP=%-4d FP=%-4d FN=%-4d P=%s R=%s" % (
                c, v["tp"], v["fp"], v["fn"],
                "%.3f" % v["precision"] if v["precision"] is not None else "-",
                "%.3f" % v["recall"] if v["recall"] is not None else "-"))
        print("层3 FP/千张=%.1f | 组级漏检率=%.3f (%d/%d 组) | 电芯级(有电芯号)=%s" % (
            out["fp_per_1000img"],
            out["group_level"]["miss_rate"] or 0, n_missed, len(defect_groups),
            ("%.3f (n=%d)" % (cell_miss, len(by_cell))) if cell_miss is not None else "样本不足"))
    return out


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", default="1920",
                    help="1920 或矩形 '1184x2112'(h x w)。必须与训练时一致才可同表比较")
    ap.add_argument("--conf", type=float, default=0.25, help="工作点阈值（与产线一致）")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-layer1", action="store_true")
    ap.add_argument("--roi", action="store_true",
                    help="按 core/roi.py 的边界裁剪后再评估（必须与训练口径一致；不裁剪评估裁剪训练的模型等于测另一个分布）")
    args = ap.parse_args()
    # 关键：ultralytics 的 train/val 只接受方形 int，predict 才接受矩形。
    # 这里把"请求值"和"实际生效值"分开算，避免 JSON 里记录一个并未生效的画布。
    _sz_req = resolve_imgsz(args.imgsz, for_what="predict")
    imgsz_l1 = check_imgsz(_sz_req, stride=32, max_dim=1)      # YOLO.val 实际用的
    imgsz_pred = check_imgsz(_sz_req, stride=32, min_dim=2)    # predict 实际用的


    rows = [r for r in ds.load_manifest(args.manifest) if r["split"] == args.split]
    if not rows:
        print("manifest 中 split=%s 无数据" % args.split)
        return
    tag = args.tag or ("%s_%s" % (os.path.splitext(os.path.basename(args.weights))[0], args.split))
    print("评估集：%s（%d 张）%s" % (args.split, len(rows), " [ROI 裁剪]" if args.roi else ""))
    print("权重：%s (md5=%s)" % (args.weights, md5_file(args.weights)[:12]))

    yaml_path = prepare_eval_dataset(rows, tag, roi=args.roi)
    roi_img_dir = os.path.join(CACHE, "eval_%s_roi" % tag, "images") if args.roi else None
    t0 = time.time()
    report = {
        "schema": "ct-eval-report/1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "weights": args.weights,
        "weights_md5": md5_file(args.weights),
        "manifest": args.manifest,
        "manifest_md5": md5_file(args.manifest),
        "split": args.split,
        "n_images": len(rows),
        "roi": bool(args.roi),
        "imgsz_requested": args.imgsz,
        "imgsz_layer1": imgsz_l1,
        "imgsz_predict": imgsz_pred,
    }
    if not args.skip_layer1:
        report["layer1_coco"] = layer1_coco(args.weights, yaml_path, imgsz_l1, args.device)
    report["layer2_3_workpoint"] = layer23_workpoint(
        args.weights, rows, imgsz_pred, args.conf, args.iou, args.device, roi_dir=roi_img_dir)
    report["elapsed_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.join(ROOT, "reports", "eval"), exist_ok=True)
    outp = os.path.join("reports", "eval", "%s.json" % tag)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n报告已写出：%s（%.0fs）" % (outp, report["elapsed_s"]))


if __name__ == "__main__":
    main()
