# -*- coding: utf-8 -*-
"""
预处理专项分析（只读：母本图片与标注零改动，只写 reports/）

回答四个问题：
  1) 图像原生分辨率/长宽比到底长什么样 → 当前 imgsz=1920 的方形画布浪费多少
  2) 三类缺陷的**像素尺寸**分布（原图 / letterbox 后 / mosaic 后）→ 谁在被 resize 杀掉
  3) 缺陷的**灰度对比度**有多微弱（缺陷内均值 vs 周边环均值）→ 8bit + 插值后的存活率
  4) 不同插值方式（NEAREST/LINEAR/AREA/CUBIC/LANCZOS4）对缺陷对比度的保留率

用法：
  runtime\\python.exe scripts\\analyze_preprocess.py --sample 600
"""
import os
import sys
import csv
import json
import argparse
import statistics as st
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import dataset_split as ds  # noqa: E402

CLS = {0: "damage", 1: "dark", 2: "light"}
CN = {"damage": "极片损伤", "dark": "暗斑", "light": "亮斑"}
OUT_DIR = os.path.join(ROOT, "reports", "preprocess")


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    k = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
    return v[k]


def load_boxes(jpath):
    """返回 [(cls, x1, y1, x2, y2)] 像素坐标"""
    try:
        d = json.load(open(jpath, encoding="utf-8"))
    except Exception:
        return [], None, None
    W, H = d.get("imageWidth"), d.get("imageHeight")
    out = []
    for s in d.get("shapes", []):
        try:
            c = int(s["label"])
        except (ValueError, KeyError):
            continue
        pts = s.get("points") or []
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((c, max(0.0, min(xs)), max(0.0, min(ys)),
                    min(float(W), max(xs)), min(float(H), max(ys))))
    return out, W, H


def ring_contrast(gray, box, pad=6):
    """缺陷内部均值 - 周边环均值（有符号）。返回 (diff, 内部均值, 环内 std)"""
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    inner = gray[y1:y2, x1:x2].astype(np.float32)
    X1, Y1 = max(0, x1 - pad), max(0, y1 - pad)
    X2, Y2 = min(W, x2 + pad), min(H, y2 + pad)
    outer = gray[Y1:Y2, X1:X2].astype(np.float32).copy()
    m = np.ones(outer.shape, np.float32)
    m[(y1 - Y1):(y1 - Y1) + (y2 - y1), (x1 - X1):(x1 - X1) + (x2 - x1)] = 0
    ring = outer[m > 0]
    if ring.size < 10:
        return None
    return float(inner.mean() - ring.mean()), float(inner.mean()), float(ring.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--sample", type=int, default=600, help="参与像素级分析的图片数")
    ap.add_argument("--pad", type=int, default=6, help="周边环的像素膨胀量")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = ds.load_manifest(args.manifest)
    print("清单 %d 张" % len(rows))

    # ---------------- 1. 分辨率 / 长宽比 ----------------
    sizes = Counter(r["size"] for r in rows)
    ars, ws, hs = [], [], []
    for r in rows:
        w, h = (int(x) for x in r["size"].split("x"))
        ws.append(w)
        hs.append(h)
        ars.append(w / h)

    # 当前 imgsz=1920 方形 letterbox 的实际缩放率与填充率
    lb = []
    for w, h in zip(ws, hs):
        s = min(1920.0 / w, 1920.0 / h)
        nw, nh = round(w * s), round(h * s)
        lb.append({"W": w, "H": h, "scale": s, "content": (nw, nh),
                   "fill": (nw * nh) / (1920.0 * 1920.0)})
    scale_by_size = defaultdict(list)
    fill_by_size = defaultdict(list)
    for d in lb:
        scale_by_size["%dx%d" % (d["W"], d["H"])].append(d["scale"])
        fill_by_size["%dx%d" % (d["W"], d["H"])].append(d["fill"])

    # ---------------- 2. 边框/背景均匀度（抽样） ----------------
    idx = np.linspace(0, len(rows) - 1, min(args.sample, len(rows))).astype(int)
    sample = [rows[i] for i in idx]

    border_frac, nchan = [], Counter()
    for r in sample:
        img = ds.imread_cn(os.path.join(ROOT, r["src_path"]), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 2:
            nchan[1] += 1
            g = img
        else:
            nchan[img.shape[2]] += 1
            g = img[..., 0]
            if img.shape[2] == 3 and np.array_equal(img[..., 0], img[..., 1]) and \
                    np.array_equal(img[..., 1], img[..., 2]):
                nchan["3ch_gray"] += 1
        H, W = g.shape[:2]
        b = 8
        edges = np.concatenate([g[:b, :].ravel(), g[-b:, :].ravel(),
                                g[:, :b].ravel(), g[:, -b:].ravel()])
        mode = np.bincount(edges.astype(np.uint8) if edges.max() < 256 else edges.astype(np.int32).ravel()).argmax()
        border_frac.append(float((np.abs(edges.astype(np.float32) - mode) <= 2).mean()))

    # ---------------- 3. 每类框的像素尺寸 ----------------
    per_cls = defaultdict(lambda: {"w": [], "h": [], "a": [], "ar": [],
                                   "w1920": [], "a1920": [], "a1920_mosaic": [],
                                   "n_boxes": 0, "neg_img": 0})
    boxes_per_img = []
    for r in rows:
        jp = r.get("label_json")
        if not jp:
            per_cls["__none__"]["neg_img"] += 1
            boxes_per_img.append(0)
            continue
        bs, W, H = load_boxes(os.path.join(ROOT, jp))
        if not W:
            continue
        boxes_per_img.append(len(bs))
        s1920 = min(1920.0 / W, 1920.0 / H)
        for c, x1, y1, x2, y2 in bs:
            name = CLS.get(c)
            if not name:
                continue
            bw, bh = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
            d = per_cls[name]
            d["n_boxes"] += 1
            d["w"].append(bw)
            d["h"].append(bh)
            d["a"].append(bw * bh)
            d["ar"].append(max(bw, bh) / max(1e-6, min(bw, bh)))
            d["w1920"].append(bw * s1920)
            d["a1920"].append(bw * bh * s1920 * s1920)
            # mosaic：2x2 拼接后每个子图再缩一半
            d["a1920_mosaic"].append(bw * bh * (s1920 / 2.0) ** 2)

    # ---------------- 4. 灰度对比度 + 插值保留率 ----------------
    methods = {"NEAREST": cv2.INTER_NEAREST, "LINEAR": cv2.INTER_LINEAR,
               "AREA": cv2.INTER_AREA, "CUBIC": cv2.INTER_CUBIC,
               "LANCZOS4": cv2.INTER_LANCZOS4}
    contrast = defaultdict(list)          # cls -> [contrast]
    retention = defaultdict(lambda: defaultdict(list))  # cls -> method -> [ratio]
    upscale = defaultdict(list)

    for r in sample:
        jp = r.get("label_json")
        if not jp:
            continue
        bs, W, H = load_boxes(os.path.join(ROOT, jp))
        if not bs or not W:
            continue
        img = ds.imread_cn(os.path.join(ROOT, r["src_path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if img.shape[1] != W or img.shape[0] != H:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        s = min(1920.0 / W, 1920.0 / H)
        upscale["__all__"].append(s)
        for c, x1, y1, x2, y2 in bs:
            name = CLS.get(c)
            if not name:
                continue
            base = ring_contrast(img, (x1, y1, x2, y2), args.pad)
            if base is None:
                continue
            diff0 = base[0]
            if abs(diff0) < 1e-6:
                continue
            contrast[name].append(diff0)
            for mname, mflag in methods.items():
                nw, nh = max(1, int(round(W * s))), max(1, int(round(H * s)))
                rs = cv2.resize(img, (nw, nh), interpolation=mflag)
                got = ring_contrast(rs, (x1 * s, y1 * s, x2 * s, y2 * s),
                                    max(1, int(round(args.pad * s))))
                if got is None:
                    continue
                retention[name][mname].append(got[0] / diff0)

    # ---------------- 汇总 ----------------
    def summ(name):
        d = per_cls[name]
        a = d["a"]
        return {
            "n_boxes": d["n_boxes"],
            "w_px": {"p5": pct(d["w"], .05), "p50": pct(d["w"], .5), "p95": pct(d["w"], .95)},
            "h_px": {"p5": pct(d["h"], .05), "p50": pct(d["h"], .5), "p95": pct(d["h"], .95)},
            "area_px": {"p5": pct(a, .05), "p25": pct(a, .25), "p50": pct(a, .5),
                        "p75": pct(a, .75), "p95": pct(a, .95)},
            "equiv_side_px": {"p5": pct([x ** .5 for x in a], .05),
                              "p50": pct([x ** .5 for x in a], .5),
                              "p95": pct([x ** .5 for x in a], .95)},
            "aspect_ratio": {"p50": pct(d["ar"], .5), "p90": pct(d["ar"], .9)},
            "equiv_side_1920": {"p5": pct([x ** .5 for x in d["a1920"]], .05),
                                "p50": pct([x ** .5 for x in d["a1920"]], .5),
                                "p95": pct([x ** .5 for x in d["a1920"]], .95)},
            "equiv_side_1920_mosaic": {"p5": pct([x ** .5 for x in d["a1920_mosaic"]], .05),
                                       "p50": pct([x ** .5 for x in d["a1920_mosaic"]], .5),
                                       "p95": pct([x ** .5 for x in d["a1920_mosaic"]], .95)},
            "frac_side_lt8_after_mosaic": float(np.mean([x ** .5 < 8 for x in d["a1920_mosaic"]])),
            "frac_side_lt16_after_mosaic": float(np.mean([x ** .5 < 16 for x in d["a1920_mosaic"]])),
            "contrast_gray": {"p5": pct(contrast[name], .05) if contrast[name] else None,
                              "p50": pct(contrast[name], .5) if contrast[name] else None,
                              "p95": pct(contrast[name], .95) if contrast[name] else None,
                              "n": len(contrast[name])},
            "interp_retention": {m: (round(float(np.mean(v)), 4) if v else None)
                                 for m, v in retention[name].items()},
        }

    result = {
        "n_images": len(rows),
        "resolution": {
            "distinct": {k: v for k, v in sizes.most_common()},
            "width": {"min": min(ws), "p50": pct(ws, .5), "max": max(ws)},
            "height": {"min": min(hs), "p50": pct(hs, .5), "max": max(hs)},
            "aspect_ratio": {"min": round(min(ars), 4), "p50": round(pct(ars, .5), 4),
                             "max": round(max(ars), 4)},
            "letterbox_1920_square": {
                "scale_by_size": {k: round(float(np.mean(v)), 4) for k, v in scale_by_size.items()},
                "canvas_fill_by_size": {k: round(float(np.mean(v)), 4) for k, v in fill_by_size.items()},
                "canvas_fill_overall": round(float(np.mean([d["fill"] for d in lb])), 4),
            },
            "channels": dict(nchan),
            "border_uniform_frac": {"p50": round(pct(border_frac, .5), 4) if border_frac else None},
        },
        "boxes_per_image": {
            "n_images_with_label": len(boxes_per_img),
            "neg_images": sum(1 for x in boxes_per_img if x == 0),
            "p50": pct(boxes_per_img, .5), "p90": pct(boxes_per_img, .9),
            "max": max(boxes_per_img) if boxes_per_img else 0,
        },
        "per_class": {n: summ(n) for n in ("damage", "dark", "light")},
        "resize_scale_overall": {"min": round(min(upscale["__all__"]), 4) if upscale["__all__"] else None,
                                 "p50": round(pct(upscale["__all__"], .5), 4) if upscale["__all__"] else None,
                                 "max": round(max(upscale["__all__"]), 4) if upscale["__all__"] else None},
        "params": {"sample": args.sample, "ring_pad": args.pad,
                   "letterbox_target": 1920, "manifest": args.manifest},
    }

    out = os.path.join(OUT_DIR, "preprocess_analysis.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # ---------------- 控制台摘要 ----------------
    print("\n== 分辨率 ==")
    for k, v in sizes.most_common():
        print("  %-12s %5d 张 | 1920方形缩放 %.3f | 画布填充 %.1f%%"
              % (k, v, np.mean(scale_by_size[k]), 100 * np.mean(fill_by_size[k])))
    print("  长宽比 p50=%.3f（全部横向 16:9）| 画布平均填充 %.1f%%"
          % (pct(ars, .5), 100 * result["resolution"]["letterbox_1920_square"]["canvas_fill_overall"]))

    print("\n== 每类框尺寸（像素）==")
    print("%-7s %6s %-22s %-24s %-22s" % ("类", "框数", "等效边长(原图)", "等效边长@1920", "等效边长@1920+mosaic"))
    for n in ("damage", "dark", "light"):
        s = result["per_class"][n]
        f = lambda d: "p5 %.1f / p50 %.1f / p95 %.1f" % (d["p5"], d["p50"], d["p95"])
        print("%-7s %6d %-22s %-24s %-22s" % (n, s["n_boxes"],
              f(s["equiv_side_px"]), f(s["equiv_side_1920"]), f(s["equiv_side_1920_mosaic"])))

    print("\n== mosaic 后小目标占比 ==")
    for n in ("damage", "dark", "light"):
        s = result["per_class"][n]
        print("  %-7s 边长<8px: %5.1f%% | <16px: %5.1f%%"
              % (n, 100 * s["frac_side_lt8_after_mosaic"], 100 * s["frac_side_lt16_after_mosaic"]))

    print("\n== 灰度对比度（缺陷内 - 周边环，单位：灰度级）==")
    for n in ("damage", "dark", "light"):
        c = result["per_class"][n]["contrast_gray"]
        if c["p50"] is not None:
            print("  %-7s p5 %+.1f | p50 %+.1f | p95 %+.1f (n=%d)"
                  % (n, c["p5"], c["p50"], c["p95"], c["n"]))

    print("\n== 插值方式对比度保留率（1920 letterbox 实际缩放率下）==")
    print("%-7s %s" % ("类", "  ".join("%-10s" % m for m in methods)))
    for n in ("damage", "dark", "light"):
        r = result["per_class"][n]["interp_retention"]
        print("%-7s %s" % (n, "  ".join(
            "%-10s" % ("%.3f" % r[m] if r.get(m) is not None else "-") for m in methods)))

    print("\n写出 %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
