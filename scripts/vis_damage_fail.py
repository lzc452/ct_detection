# -*- coding: utf-8 -*-
"""失败样例对照：同一张测试图，V5（全图）检到了、V5.1-ROI 没检到的损伤。

用途：把「损伤召回从 0.52 掉到 0.37」这句话落成看得见的原因，
而不是停在指标上猜。输出三联图：
    左   全图 + V5 预测（绿=命中，红=误报）
    中   ROI 裁剪 + 真值框（蓝）
    右   ROI 裁剪 + ROI 模型预测

用法：
    ./runtime/python.exe scripts/vis_damage_fail.py
产出：
    reports/train/damage_fail_cases.jpg
    reports/eval/damage_fail_cases.json
"""
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

import roi as R                      # noqa: E402
import dataset_split as ds           # noqa: E402

V5_W = "runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.pt"
ROI_W = "runs/detect/CT_pro/V51_ROI_1536/weights/best.pt"
CONF = 0.25
IOU = 0.5
CLS_CN = {0: "damage", 1: "dark", 2: "light"}


def iou(a, b):
    xx0, yy0 = max(a[0], b[0]), max(a[1], b[1])
    xx1, yy1 = min(a[2], b[2]), min(a[3], b[3])
    w, h = max(0.0, xx1 - xx0), max(0.0, yy1 - yy0)
    inter = w * h
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def gt_in_roi(row):
    W0, H0 = [int(x) for x in row["size"].split("x")]
    box = R.roi_box(W0, H0)
    cw, ch = box[2] - box[0], box[3] - box[1]
    out = []
    if row["label_json"]:
        lines, _, _ = R.labelme_boxes_in_roi(row["label_json"], box)
        for ln in lines:
            cs = ln.split()
            c = int(cs[0])
            xc, yc, bw, bh = (float(v) for v in cs[1:5])
            out.append((c, [(xc - bw / 2) * cw, (yc - bh / 2) * ch,
                            (xc + bw / 2) * cw, (yc + bh / 2) * ch]))
    return out, box


def gt_full(row):
    W0, H0 = [int(x) for x in row["size"].split("x")]
    out = []
    if row["label_json"]:
        _, _, boxes = ds.json_to_yolo(row["label_json"])
        for c, xc, yc, bw, bh in boxes:
            out.append((c, [(xc - bw / 2) * W0, (yc - bh / 2) * H0,
                            (xc + bw / 2) * W0, (yc + bh / 2) * H0]))
    return out


def preds_of(res):
    out = []
    if res.boxes is not None and len(res.boxes):
        xyxy = res.boxes.xyxy.cpu().numpy()
        cl = res.boxes.cls.cpu().numpy().astype(int)
        cf = res.boxes.conf.cpu().numpy()
        out = [(int(cl[i]), [float(v) for v in xyxy[i]], float(cf[i])) for i in range(len(cf))]
    return out


def tp_set(gt, pr):
    """返回被命中的 gt 下标集合"""
    used, hit = [False] * len(gt), set()
    for c, b, _cf in sorted(pr, key=lambda z: -z[2]):
        best, bi = 0.0, -1
        for i, (gc, gb) in enumerate(gt):
            if used[i] or gc != c:
                continue
            v = iou(b, gb)
            if v > best:
                best, bi = v, i
        if bi >= 0 and best >= IOU:
            used[bi] = True
            hit.add(bi)
    return hit


def draw(img, gt, pr, hit):
    im = img.copy()
    for i, (c, b) in enumerate(gt):
        col = (255, 128, 0) if i not in hit else (200, 200, 0)   # 未命中=蓝, 命中=青
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 2)
        cv2.putText(im, "GT:" + CLS_CN[c], (int(b[0]), max(12, int(b[1]) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    for c, b, cf in pr:
        col = (60, 60, 220) if c == 0 else (60, 200, 60)
        cv2.rectangle(im, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), col, 1)
        cv2.putText(im, "%s %.2f" % (CLS_CN[c], cf), (int(b[0]), int(b[3]) + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    return im


def fit(w, h, tw):
    s = tw / float(w)
    return cv2.resize(img_holder[0], (tw, int(h * s))) if False else s


def main():
    from ultralytics import YOLO
    rows = [r for r in ds.load_manifest("reports/splits/v5_split_manifest.csv")
            if r["split"] == "test"]
    # 只看高分辨率、含损伤的图
    cand = [r for r in rows if r["size"] == "2129x1198" and r["n_damage"] not in ("0", 0)]
    print("含损伤的高分辨率测试图：%d 张" % len(cand))

    m_v5 = YOLO(V5_W)
    m_roi = YOLO(ROI_W)

    cases = []
    for s in range(0, len(cand), 16):
        chunk = cand[s:s + 16]
        full_paths = [r["src_path"] for r in chunk]
        roi_imgs, boxes = [], []
        for r in chunk:
            im = cv2.imdecode(np.fromfile(r["src_path"], np.uint8), cv2.IMREAD_COLOR)
            ri, bx = R.crop_roi(im)
            roi_imgs.append(ri)
            boxes.append(bx)
        pv = [preds_of(x) for x in m_v5.predict(full_paths, imgsz=1920, conf=CONF,
                                                iou=0.7, verbose=False)]
        pr_ = [preds_of(x) for x in m_roi.predict(roi_imgs, imgsz=1536, conf=CONF,
                                                  iou=0.7, verbose=False)]
        for r, im_roi, bx, pf, prr in zip(chunk, roi_imgs, boxes, pv, pr_):
            gf = gt_full(r)
            gr, _ = gt_in_roi(r)
            hf = tp_set(gf, pf)
            hr = tp_set(gr, prr)
            # 损伤真值里，V5 命中而 ROI 未命中的
            dmg_f = {i for i in range(len(gf)) if gf[i][0] == 0 and i in hf}
            dmg_r = {i for i in range(len(gr)) if gr[i][0] == 0 and i in hr}
            lost = len(dmg_f) - len(dmg_r)
            if lost > 0:
                cases.append({"name": r["name"], "src": r["src_path"],
                              "dmg_v5_hit": len(dmg_f), "dmg_roi_hit": len(dmg_r),
                              "n_dmg": r["n_damage"], "roi_img": im_roi, "box": bx,
                              "gf": gf, "gr": gr, "pf": pf, "pr": prr,
                              "hf": hf, "hr": hr})
    cases.sort(key=lambda c: -(c["dmg_v5_hit"] - c["dmg_roi_hit"]))
    print("V5 命中但 ROI 漏掉的损伤图：%d 张" % len(cases))
    for c in cases[:6]:
        print("  %-10s 损伤 %s | V5 命中 %d | ROI 命中 %d" % (
            c["name"], c["n_dmg"], c["dmg_v5_hit"], c["dmg_roi_hit"]))

    picks = cases[:3]
    if not picks:
        print("没有差异样例")
        return 1

    TW = 420
    rows_img = []
    for c in picks:
        full = cv2.imdecode(np.fromfile(c["src"], np.uint8), cv2.IMREAD_COLOR)
        a = draw(full, c["gf"], c["pf"], c["hf"])
        b = draw(c["roi_img"], c["gr"], [], set(range(len(c["gr"]))))
        d = draw(c["roi_img"], c["gr"], c["pr"], c["hr"])
        tiles = []
        for t in (a, b, d):
            h, w = t.shape[:2]
            t = cv2.resize(t, (TW, int(TW * h / w)))
            tiles.append(t)
        H = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, H - t.shape[0], 0, 0, cv2.BORDER_CONSTANT,
                                    value=(30, 30, 30)) for t in tiles]
        row = np.hstack(tiles)
        cv2.putText(row, "%s  全图+V5预测 / ROI+真值 / ROI+ROI模型预测" % c["name"],
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        rows_img.append(row)

    Wmax = max(r.shape[1] for r in rows_img)
    rows_img = [cv2.copyMakeBorder(r, 0, 0, 0, Wmax - r.shape[1], cv2.BORDER_CONSTANT,
                                   value=(30, 30, 30)) for r in rows_img]
    out = np.vstack(rows_img)
    os.makedirs("reports/train", exist_ok=True)
    p = "reports/train/damage_fail_cases.jpg"
    cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tofile(p)
    print("已写出 %s  %s" % (p, out.shape))

    meta = [{k: v for k, v in c.items() if k not in ("roi_img", "gf", "gr", "pf", "pr")}
            for c in cases[:12]]
    with open("reports/eval/damage_fail_cases.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
