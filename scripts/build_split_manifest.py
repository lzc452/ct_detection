# -*- coding: utf-8 -*-
"""
按电芯/批次分组重划数据集划分（#10），只产出清单，不动原始数据

目的：消除"同一张切片同时出现在 train 与 val"的泄漏（评审实测 V4 val 中 82.7%）。
本脚本默认 **dry-run**：只读母本、只写 `reports/splits/*.csv` 与统计，不复制/移动任何图片。

用法：
  runtime\\python.exe scripts\\build_split_manifest.py                    # 默认参数，出清单
  runtime\\python.exe scripts\\build_split_manifest.py --calibrate        # 只做 MAE 阈值标定
  runtime\\python\\python.exe scripts\\build_split_manifest.py --cos 0.99 --mae 1.0
  runtime\\python.exe scripts\\build_split_manifest.py --out reports/splits/v5_split_manifest.csv
"""
import os
import sys
import json
import time
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import dataset_split as ds  # noqa: E402


def calibrate(items, cos_th=0.995):
    """标定 MAE 阈值：打印高余弦候选对的 MAE 分布，用于确认"同一切片"的界限"""
    desc = np.stack([it["desc"] for it in items])
    vecs = np.stack([it["mae_vec"] for it in items])
    sim = desc @ desc.T
    iu = np.triu_indices(len(items), k=1)
    s = sim[iu]
    sel = np.where(s >= cos_th)[0]
    print("候选对（余弦>=%.3f）：%d 对" % (cos_th, len(sel)))

    maes = []
    same_size = 0
    step = 2000
    for k0 in range(0, len(sel), step):
        chunk = sel[k0:k0 + step]
        for k in chunk:
            i, j = int(iu[0][k]), int(iu[1][k])
            if items[i]["h"] != items[j]["h"] or items[i]["w"] != items[j]["w"]:
                continue
            same_size += 1
            d = np.abs(vecs[i].astype(np.int16) - vecs[j].astype(np.int16))
            maes.append(float(d.mean()))
    maes = np.array(maes)
    print("同尺寸候选 %d 对；MAE@%d 分位数：" % (same_size, ds._MAE_SIZE))
    for p in (1, 5, 25, 50, 75, 90, 95, 99, 99.9):
        print("  p%-5s = %.3f" % (p, np.percentile(maes, p)))
    for th in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
        print("  MAE<=%.1f 的对数：%d (%.1f%%)" % (th, int((maes <= th).sum()),
                                                 100.0 * (maes <= th).mean()))
    return maes


def check_v4_baseline(items, cos_th=0.999, mae_th=1.0):
    """同一判据下测旧 V4 划分（val×train）的像素级重复率，用于 before/after 对比"""
    v4 = os.path.join(ROOT, "datasets", "CT_DATASET_FINAL_V4")
    if not os.path.isdir(v4):
        return None
    by_md5 = {}
    for i, it in enumerate(items):
        by_md5.setdefault(it["md5"], i)
    split_idx = {"train": [], "val": []}
    for sp in ("train", "val"):
        d = os.path.join(v4, sp, "images")
        for f in os.listdir(d):
            h = ds.md5_of(os.path.join(d, f))
            i = by_md5.get(h)
            if i is not None:
                split_idx[sp].append(i)
    if not split_idx["val"] or not split_idx["train"]:
        return None
    desc = np.stack([it["desc"] for it in items])
    vecs = np.stack([it["mae_vec"] for it in items])
    ia, ib = split_idx["train"], split_idx["val"]
    sim = desc[ib] @ desc[ia].T
    best = sim.max(axis=1)
    ri, ci = np.where(sim >= cos_th)
    n_dupe, n_val_dupe = 0, set()
    for r, c in zip(ri, ci):
        i, j = ib[int(r)], ia[int(c)]
        if items[i]["h"] != items[j]["h"]:
            continue
        mae = float(np.abs(vecs[i].astype(np.int16) - vecs[j].astype(np.int16)).mean())
        if mae <= mae_th:
            n_dupe += 1
            n_val_dupe.add(int(r))
    out = {
        "n_train": len(ia), "n_val": len(ib),
        "val_with_pixel_dupe_in_train": len(n_val_dupe),
        "val_dupe_ratio": round(len(n_val_dupe) / max(1, len(ib)), 4),
        "dupe_pairs": n_dupe,
        "val_maxcos_p50": round(float(np.percentile(best, 50)), 6),
        "val_maxcos_p90": round(float(np.percentile(best, 90)), 6),
    }
    print("\n[旧 V4 基线] val %d 张中 %d 张(%.1f%%) 在 train 里有像素级重复副本；最大余弦 p50=%.6f" % (
        out["n_val"], out["val_with_pixel_dupe_in_train"], 100 * out["val_dupe_ratio"],
        out["val_maxcos_p50"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cos", type=float, default=0.999, help="近重复粗筛余弦阈值")
    ap.add_argument("--mae", type=float, default=1.0, help="近重复精筛 MAE 阈值（%d×%d 灰度）" % (ds._MAE_SIZE, ds._MAE_SIZE))
    ap.add_argument("--ratios", default="0.70,0.15,0.15", help="train,val,test 比例（按组）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--calibrate", action="store_true", help="只做阈值标定")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    items = ds.scan_master()
    if args.limit:
        items = items[:args.limit]

    if args.calibrate:
        calibrate(items, args.cos)
        return

    group_of, ginfo = ds.build_groups(items, cos_th=args.cos, mae_th=args.mae)
    v4_base = check_v4_baseline(items, cos_th=args.cos, mae_th=args.mae)
    ratios = tuple(float(x) for x in args.ratios.split(","))
    split_of, total = ds.assign_splits(items, group_of, ratios=ratios, seed=args.seed)
    problems = ds.assert_no_leak(items, group_of, split_of, cos_th=args.cos, mae_th=args.mae)
    out = ds.build_manifest(items, group_of, split_of, args.out)

    # 统计：无电芯号的组占比（诚实披露用）
    cells = ds.load_provenance(items, verbose=False)
    n_with_cell = sum(1 for it in items if it["name"] in cells)
    n_groups_no_cell = len({group_of[i] for i, it in enumerate(items) if it["name"] not in cells})

    stats = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_images": len(items),
        "group_info": ginfo,
        "split_totals": total,
        "leak_problems": problems,
        "v4_baseline": v4_base,
        "coverage": {
            "images_with_cell_id": n_with_cell,
            "ratio": round(n_with_cell / max(1, len(items)), 4),
            "groups_without_cell_id": n_groups_no_cell,
        },
        "params": {"cos": args.cos, "mae": args.mae, "ratios": ratios, "seed": args.seed},
        "elapsed_s": round(time.time() - t0, 1),
    }
    statp = os.path.splitext(args.out)[0] + "_stats.json"
    with open(statp, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\n统计已写出：%s" % statp)
    print("清单已写出：%s" % out)
    print("总耗时 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
