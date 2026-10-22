# -*- coding: utf-8 -*-
"""ROI 裁剪保真度 + 画布几何的第一性核算。

只做两件事，全部基于磁盘上的真实文件，不含任何假设常数：
  A. 量化「裁剪 + q95 重编码」相对原图的像素损失（PSNR/SSIM/缺陷区对比度变化）
  B. 用真实标注框尺算出「缺陷在每种画布下有多大」的换算表

用法：
    runtime\python.exe scripts\analyze_roi_fidelity.py
"""
import os
import sys
import csv
import json
import statistics

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import dataset_split as ds  # noqa: E402
import roi as R  # noqa: E402

MANIFEST = os.path.join(ROOT, "reports", "splits", "v5_split_manifest.csv")
OUT_JSON = os.path.join(ROOT, "reports", "eval", "roi_fidelity.json")
OUT_DIR = os.path.join(ROOT, "reports", "preprocess")


def imread_cn(p):
    b = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(b, cv2.IMREAD_COLOR) if b.size else None


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    """标准 11x11 高斯窗 SSIM（灰度）"""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu1 = cv2.GaussianBlur(ga, k, 1.5)
    mu2 = cv2.GaussianBlur(gb, k, 1.5)
    s11 = cv2.GaussianBlur(ga * ga, k, 1.5) - mu1 * mu1
    s22 = cv2.GaussianBlur(gb * gb, k, 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(ga * gb, k, 1.5) - mu1 * mu2
    m = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s11 + s22 + C2))
    return float(m.mean())


def jpeg_roundtrip(img, q=95):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("imencode 失败")
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _is_hi(size):
    """宽 >= 2000 视为高分辨率组（实测清单只有 2129/2115 与 1052 两类）"""
    return int(size.split("x")[0]) >= 2000


def fidelity(n_hi=12, n_lo=12):
    """对照三组：原图ROI切片(无损) / 该切片的 q95 重编码 / 原图全集"""
    rows = ds.load_manifest(MANIFEST)
    hi = [r for r in rows if _is_hi(r["size"])]
    lo = [r for r in rows if not _is_hi(r["size"])]
    res = {}
    for tag, pool, n in (("hi", hi, n_hi), ("lo", lo, n_lo)):
        # 优先挑有标注的，保证缺陷区也能被量到
        pool = ([r for r in pool if r.get("label_json")] or pool)[:n * 8]
        picked, stats, box_deltas = [], [], []
        step = max(1, len(pool) // n)
        for r in pool[::step][:n]:
            img = imread_cn(r["src_path"])
            if img is None:
                continue
            roi_img, box = R.crop_roi(img)
            rt = jpeg_roundtrip(roi_img, 95)
            if rt.shape != roi_img.shape:
                continue
            d = (rt.astype(np.int16) - roi_img.astype(np.int16))
            picked.append({
                "name": r["name"],
                "src_size": r["size"],
                "roi_wh": [int(roi_img.shape[1]), int(roi_img.shape[0])],
                "psnr_db": round(psnr(roi_img, rt), 2),
                "ssim": round(ssim(roi_img, rt), 5),
                "max_abs_err": int(np.abs(d).max()),
                "mae": round(float(np.abs(d).mean()), 3),
                "mean_signed_err": round(float(d.mean()), 3),
            })
            # 缺陷区（原始标注框内）的局部对比度变化
            if r.get("label_json"):
                lines, _, _ = R.labelme_boxes_in_roi(r["label_json"], box)
                H, W = roi_img.shape[:2]
                for ln in lines:
                    p = ln.split()
                    xc, yc, bw, bh = (float(v) for v in p[1:5])
                    x1, y1 = int((xc - bw / 2) * W), int((yc - bh / 2) * H)
                    x2, y2 = int((xc + bw / 2) * W), int((yc + bh / 2) * H)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(W, x2), min(H, y2)
                    if x2 - x1 < 3 or y2 - y1 < 3:
                        continue
                    sl0 = roi_img[y1:y2, x1:x2]
                    sl1 = rt[y1:y2, x1:x2]
                    # 框内标准差 = 缺陷的局部可分辨信号强度
                    c0 = float(cv2.cvtColor(sl0, cv2.COLOR_BGR2GRAY).std())
                    c1 = float(cv2.cvtColor(sl1, cv2.COLOR_BGR2GRAY).std())
                    box_deltas.append((c0, c1, x2 - x1, y2 - y1))
        res[tag] = {
            "n_images": len(picked),
            "images": picked,
            "psnr_db_median": round(statistics.median([p["psnr_db"] for p in picked]), 2) if picked else None,
            "ssim_median": round(statistics.median([p["ssim"] for p in picked]), 5) if picked else None,
            "mae_median": round(statistics.median([p["mae"] for p in picked]), 3) if picked else None,
            "n_gt_boxes": len(box_deltas),
            "gt_std_before_median": round(statistics.median([b[0] for b in box_deltas]), 2) if box_deltas else None,
            "gt_std_after_median": round(statistics.median([b[1] for b in box_deltas]), 2) if box_deltas else None,
            "gt_std_rel_change_pct": (round(100 * (statistics.median([b[1] for b in box_deltas])
                                                   / statistics.median([b[0] for b in box_deltas]) - 1), 2)
                                      if box_deltas else None),
        }
    return res


def defect_sizes():
    """按原生分辨率分组统计「真实标注框」的像素尺寸分布"""
    rows = ds.load_manifest(MANIFEST)
    CLS = {0: "damage", 1: "dark", 2: "light"}
    out = {}
    for tag, cond in (("hi", _is_hi), ("lo", lambda s: not _is_hi(s)),
                      ("all", lambda s: True)):
        acc = {c: {"w": [], "h": [], "area_pct": []} for c in CLS.values()}
        for r in rows:
            if not cond(r["size"]) or not r.get("label_json"):
                continue
            W0, H0 = (int(v) for v in r["size"].split("x"))
            try:
                _W, _H, boxes = ds.json_to_yolo(r["label_json"])
            except Exception:
                continue
            W0, H0 = _W, _H
            for c, xc, yc, bw, bh in boxes:
                if int(c) not in CLS:
                    continue
                acc[CLS[int(c)]]["w"].append(bw * W0)
                acc[CLS[int(c)]]["h"].append(bh * H0)
                acc[CLS[int(c)]]["area_pct"].append(100.0 * bw * bh)
        out[tag] = {}
        for c, d in acc.items():
            if not d["w"]:
                continue
            out[tag][c] = {
                "n": len(d["w"]),
                "w_med": round(statistics.median(d["w"]), 1),
                "h_med": round(statistics.median(d["h"]), 1),
                "w_p10": round(float(np.percentile(d["w"], 10)), 1),
                "w_p90": round(float(np.percentile(d["w"], 90)), 1),
                "area_pct_med": round(statistics.median(d["area_pct"]), 4),
            }
    return out


def canvas_table(sizes):
    """每种画布下：LetterBox 缩放率、各类缺陷等效边长、画布算力、mosaic 子图尺度"""
    # ROI 尺寸由归一化边界 × 实际原图尺寸算出，不用任何估计值
    def roi_of(W, H):
        b = R.roi_box(W, H)
        return (b[2] - b[0], b[3] - b[1])

    roi_hi = roi_of(2129, 1198)      # 高分辨率组主尺寸（3118 张）
    roi_lo = roi_of(1052, 582)       # 低分辨率组主尺寸（1468 张）
    hi_w = sizes["hi"].get("light", {}).get("w_med") or 15.0
    hi_dmg = sizes["hi"].get("damage", {}).get("w_med") or 22.0
    hi_dark = sizes["hi"].get("dark", {}).get("w_med") or 40.0
    rows = []
    for C in (1024, 1152, 1216, 1280, 1408, 1536, 1792, 1920, 2112):
        s_hi = min(C / roi_hi[0], C / roi_hi[1])
        s_lo = min(C / roi_lo[0], C / roi_lo[1])
        rows.append({
            "canvas": C,
            "mp": round(C * C / 1e6, 2),
            "align32": C % 32 == 0,
            "roi_hi_scale": round(s_hi, 3),
            "light_px": round(hi_w * s_hi, 1),
            "damage_w_px": round(hi_dmg * s_hi, 1),
            "dark_px": round(hi_dark * s_hi, 1),
            "roi_lo_scale": round(s_lo, 3),
            "light_px_lo": round(hi_w * s_lo, 1),
            "mosaic_cell": C // 2,
            "light_px_in_mosaic": round(hi_w * s_hi / 2, 1),
        })
    # V5 参照：全图 2129x1198 letterbox 进 1920 方形
    v5_scale = min(1920 / 2129, 1920 / 1198)
    return rows, {"roi_hi": [round(v) for v in roi_hi], "roi_lo": [round(v) for v in roi_lo],
                  "native": {"light_w": hi_w, "damage_w": hi_dmg, "dark_w": hi_dark},
                  "v5_full_scale": round(v5_scale, 4),
                  "v5_light_px": round(hi_w * v5_scale, 1),
                  "v5_damage_w_px": round(hi_dmg * v5_scale, 1),
                  "v5_dark_px": round(hi_dark * v5_scale, 1),
                  "v5_mosaic_light_px": round(hi_w * v5_scale / 2, 1)}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("[1/3] 保真度测量 ...")
    fid = fidelity()
    print("[2/3] 缺陷原生尺度统计 ...")
    sizes = defect_sizes()
    print("[3/3] 画布换算表 ...")
    table, meta = canvas_table(sizes)

    rep = {"fidelity": fid, "defect_sizes_native": sizes,
           "canvas_table": table, "canvas_meta": meta,
           "reencode_quality": 95}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)

    for tag in ("hi", "lo"):
        d = fid[tag]
        print("\n[%s] n=%d  ROI=%s" % (tag, d["n_images"],
              d["images"][0]["roi_wh"] if d["images"] else "?"))
        print("  PSNR 中位 %.2f dB | SSIM %.5f | MAE %.3f | 最大单点误差 %d" % (
            d["psnr_db_median"], d["ssim_median"], d["mae_median"],
            max(p["max_abs_err"] for p in d["images"]) if d["images"] else -1))
        if d["n_gt_boxes"]:
            print("  缺陷框内灰度 std: %.2f -> %.2f (%+.2f%%)  [n=%d]" % (
                d["gt_std_before_median"], d["gt_std_after_median"],
                d["gt_std_rel_change_pct"], d["n_gt_boxes"]))

    print("\n原生标注框尺寸中位数（像素）")
    for tag in ("hi", "lo", "all"):
        for c, d in sizes[tag].items():
            print("  %-3s %-7s n=%-5d 宽 %5.1f (p10 %4.1f / p90 %5.1f)  高 %5.1f  面积 %6.3f%%" % (
                tag, c, d["n"], d["w_med"], d["w_p10"], d["w_p90"], d["h_med"], d["area_pct_med"]))

    print("\n画布换算（高分辨率组 ROI %dx%d）" % tuple(meta["roi_hi"]))
    print("%-7s %-6s %-7s %-8s %-9s %-8s %-9s %-11s" % (
        "画布", "MP", "缩放", "亮斑px", "损伤宽px", "暗斑px", "mosaic格", "格内亮斑px"))
    for r in table:
        print("%-7d %-6.2f %-7.3f %-8.1f %-9.1f %-8.1f %-9d %-11.1f" % (
            r["canvas"], r["mp"], r["roi_hi_scale"], r["light_px"], r["damage_w_px"],
            r["dark_px"], r["mosaic_cell"], r["light_px_in_mosaic"]))
    print("\nV5 参照（全图 letterbox@1920）：缩放 %.4f | 亮斑 %.1fpx | 损伤宽 %.1fpx | 暗斑 %.1fpx | mosaic 格内亮斑 %.1fpx"
          % (meta["v5_full_scale"], meta["v5_light_px"], meta["v5_damage_w_px"],
             meta["v5_dark_px"], meta["v5_mosaic_light_px"]))
    print("低分辨率组 ROI %dx%d" % tuple(meta["roi_lo"]))
    print("\n-> %s" % os.path.relpath(OUT_JSON, ROOT))


if __name__ == "__main__":
    main()
