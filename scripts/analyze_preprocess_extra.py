# -*- coding: utf-8 -*-
"""
预处理分析的补充测量（只读）
  1) 局部噪声 std 与"可检出指数" SNR = |缺陷对比度| / (噪声/sqrt(面积))
     —— 解释为什么 mosaic / 降采样会杀死 3 灰度级的微弱缺陷
  2) 两种原生分辨率（2129x1198 vs 1052x582）是否互为缩放副本（同一张图导出两次）
  3) 通道数 / 边框均匀度 / JPEG 再编码对对比度的影响
用法：runtime\\python.exe scripts\\analyze_preprocess_extra.py --sample 600
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import dataset_split as ds  # noqa: E402

CLS = {0: "damage", 1: "dark", 2: "light"}
OUT = os.path.join(ROOT, "reports", "preprocess")


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))]


def load_boxes(jp):
    d = json.load(open(jp, encoding="utf-8"))
    W, H = d["imageWidth"], d["imageHeight"]
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
        out.append((c, max(0., min(xs)), max(0., min(ys)), min(float(W), max(xs)), min(float(H), max(ys))))
    return out, W, H


def ring_stats(gray, box, pad=6):
    H, W = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    inner = gray[y1:y2, x1:x2].astype(np.float32)
    X1, Y1, X2, Y2 = max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)
    out = gray[Y1:Y2, X1:X2].astype(np.float32)
    m = np.ones(out.shape, np.float32)
    m[(y1 - Y1):(y1 - Y1) + (y2 - y1), (x1 - X1):(x1 - X1) + (x2 - x1)] = 0
    ring = out[m > 0]
    if ring.size < 10:
        return None
    return float(inner.mean() - ring.mean()), float(ring.std()), inner.size


def snr(diff, std, area_fraction):
    """面积缩小到 area_fraction 时，均值噪声 ~ std/sqrt(N*frac)，SNR 随之下降"""
    if std <= 1e-6 or area_fraction <= 0:
        return None
    n = 0.0
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--sample", type=int, default=600)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    rows = ds.load_manifest(args.manifest)

    # ---------- 2) 两组分辨率是否互为缩放副本 ----------
    grp_sizes = defaultdict(set)
    for r in rows:
        grp_sizes[r["group"]].add(r["size"])
    mixed = {g: s for g, s in grp_sizes.items() if len(s) > 1}
    n_mixed_img = sum(1 for r in rows if r["group"] in mixed)

    # ---------- 1)+3) 抽样：噪声、SNR、通道、边框、JPEG ----------
    idx = np.linspace(0, len(rows) - 1, min(args.sample, len(rows))).astype(int)
    sample = [rows[i] for i in idx]

    cls_stats = defaultdict(lambda: {"diff": [], "std": [], "area": [], "px_per_side": []})
    nchan = Counter()
    border = []
    jpeg_ret = defaultdict(list)
    bg_mode_frac = []

    for r in sample:
        p = os.path.join(ROOT, r["src_path"])
        img = ds.imread_cn(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if img.ndim == 3:
            nchan[img.shape[2]] += 1
            if img.shape[2] == 3 and np.array_equal(img[..., 0], img[..., 1]) and \
                    np.array_equal(img[..., 1], img[..., 2]):
                nchan["3ch_all_equal"] += 1
        else:
            nchan[1] += 1
        g = img if img.ndim == 2 else (img[..., 0] if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
        H, W = g.shape[:2]
        b = 8
        e = np.concatenate([g[:b, :].ravel(), g[-b:, :].ravel(), g[:, :b].ravel(), g[:, -b:].ravel()])
        border.append(float(e.std()))

        jp = r.get("label_json")
        if not jp:
            continue
        bs, _, _ = load_boxes(os.path.join(ROOT, jp))
        if not bs:
            continue

        # JPEG 再编码（q=90 / 75）后对比度保留率
        tri = [cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])[1]
               for q in (90, 75)]
        redec = [cv2.imdecode(t, cv2.IMREAD_GRAYSCALE) for t in tri]

        for c, x1, y1, x2, y2 in bs:
            name = CLS.get(c)
            if not name:
                continue
            st0 = ring_stats(g, (x1, y1, x2, y2))
            if st0 is None:
                continue
            diff0, std0, area0 = st0
            if abs(diff0) < 1e-6:
                continue
            cls_stats[name]["diff"].append(diff0)
            cls_stats[name]["std"].append(std0)
            cls_stats[name]["area"].append(float(area0))
            cls_stats[name]["px_per_side"].append(float(np.sqrt(area0)))
            for qi, q in enumerate((90, 75)):
                st1 = ring_stats(redec[qi], (x1, y1, x2, y2))
                if st1:
                    jpeg_ret[(name, q)].append(st1[0] / diff0)

    # SNR 计算：单点 SNR → 面积缩小后 SNR（均值滤波把噪声降 sqrt(N)）
    snr_tbl = {}
    for name, d in cls_stats.items():
        if not d["diff"]:
            continue
        ad = [abs(x) for x in d["diff"]]
        sides = d["px_per_side"]
        # 原始：SNR = |diff| / (std/sqrt(N))
        s_orig = [a / (s / np.sqrt(n)) if s > 1e-6 else np.nan
                  for a, s, n in zip(ad, d["std"], d["area"])]
        # mosaic（线性缩 1/2，面积 1/4）
        s_mos = [a / (s / np.sqrt(max(1.0, n * 0.25))) if s > 1e-6 else np.nan
                 for a, s, n in zip(ad, d["std"], d["area"])]
        # 1920 letterbox 对低分辨率组是放大、高分辨率组是 0.9 缩小 → 用 0.8 倍面积近似（保守）
        s_lb = [a / (s / np.sqrt(max(1.0, n * 0.81))) if s > 1e-6 else np.nan
                for a, s, n in zip(ad, d["std"], d["area"])]
        snr_tbl[name] = {
            "n": len(ad),
            "contrast_abs_p50": float(pct(ad, .5)),
            "noise_std_p50": float(pct(d["std"], .5)),
            "side_px_p50": float(pct(sides, .5)),
            "snr_orig_p50": float(pct(s_orig, .5)),
            "snr_letterbox_p50": float(pct(s_lb, .5)),
            "snr_mosaic_p50": float(pct(s_mos, .5)),
        }

    res = {
        "resolution_groups_mixed": {
            "n_groups_with_multiple_sizes": len(mixed),
            "n_images_in_those_groups": n_mixed_img,
            "examples": [{"group": g, "sizes": sorted(s)} for g, s in list(mixed.items())[:8]],
        },
        "channels": dict(nchan),
        "border_std": {"p50": round(float(pct(border, .5)), 2) if border else None,
                       "p90": round(float(pct(border, .9)), 2) if border else None},
        "snr": snr_tbl,
        "jpeg_retention": {("%s_q%d" % (k[0], k[1])): round(float(np.mean(v)), 4)
                           for k, v in jpeg_ret.items() if v},
    }
    out = os.path.join(OUT, "preprocess_analysis_extra.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("== 分辨率组是否混装 ==")
    print("  含多种尺寸的组: %d 个 | 涉及图片 %d 张" % (len(mixed), n_mixed_img))
    for e in res["resolution_groups_mixed"]["examples"][:5]:
        print("    group %s -> %s" % (e["group"], e["sizes"]))
    print("\n== 通道 ==", dict(nchan))
    print("边框灰度 std: p50=%.2f p90=%.2f" % (res["border_std"]["p50"], res["border_std"]["p90"] or -1))
    print("\n== 可检出指数 SNR（p50）==")
    print("%-7s %6s %8s %8s %8s %10s %10s %10s" % ("类", "n", "|对比度|", "噪声std", "边长px", "SNR原图", "SNR@1920", "SNR+mosaic"))
    for n, v in snr_tbl.items():
        print("%-7s %6d %8.1f %8.1f %8.1f %10.1f %10.1f %10.1f" % (
            n, v["n"], v["contrast_abs_p50"], v["noise_std_p50"], v["side_px_p50"],
            v["snr_orig_p50"], v["snr_letterbox_p50"], v["snr_mosaic_p50"]))
    print("\n== JPEG 再编码后的对比度保留率 ==")
    for k, v in sorted(res["jpeg_retention"].items()):
        print("  %-14s %.3f" % (k, v))
    print("\n写出 %s" % out)


if __name__ == "__main__":
    main()
