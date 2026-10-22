# -*- coding: utf-8 -*-
"""
冻结划分的守卫校验（CI 用）：证明"考卷/测试集"没有被训练数据污染

检查三件事：
  1) 清单自身一致性：组键、MD5 在 train/val/test 之间不相交；文件仍在且 MD5 未变
  2) 与守卫数据集（如某个训练集目录）的重复：先 MD5，再（--deep）像素级 MAE 复核
  3) 报告 test 集规模与各类框数，便于冻结存档

用法：
  runtime\\python.exe scripts/verify_split.py --manifest reports\\splits\\v5_split_manifest.csv
  runtime\\python.exe scripts\\verify_split.py --manifest ... --guard datasets\\CT_DATASET_FINAL_V4_AUG --deep

退出码：0 通过；1 发现泄漏或清单漂移（可直接挂到 CI / 提交前钩子）
"""
import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import dataset_split as ds  # noqa: E402


def check_manifest(rows, verbose=True):
    problems = []
    # 1) 组键/MD5 不跨集
    grp_split, md5_split = {}, {}
    for r in rows:
        g, h, sp = r["group"], r["md5"], r["split"]
        if g in grp_split and grp_split[g] != sp:
            problems.append("组 %s 跨 %s/%s" % (g, grp_split[g], sp))
        grp_split[g] = sp
        if h in md5_split and md5_split[h] != sp:
            problems.append("MD5 %s 跨 %s/%s" % (h[:8], md5_split[h], sp))
        md5_split[h] = sp

    # 2) 文件仍在且 MD5 未变（母本被改动要报出来）
    drift = 0
    for r in rows:
        p = os.path.join(ROOT, r["src_path"])
        if not os.path.exists(p):
            problems.append("文件丢失: %s" % r["src_path"])
            continue
        if ds.md5_of(p) != r["md5"]:
            drift += 1
            if drift <= 5:
                problems.append("内容已变（MD5 不符）: %s" % r["src_path"])
    if drift:
        problems.append("共 %d 张图片内容与冻结清单不一致" % drift)

    if verbose:
        cnt = defaultdict(int)
        for r in rows:
            cnt[r["split"]] += 1
        print("清单自检：%s" % ", ".join("%s=%d" % (k, cnt[k]) for k in ("train", "val", "test") if cnt[k]))
    return problems


def check_guard(rows, guard_dir, split="test", deep=False, verbose=True):
    """检查指定 split 与守卫数据集之间是否存在像素级重复"""
    problems = []
    if not os.path.isdir(guard_dir):
        return ["守卫目录不存在: %s" % guard_dir]
    sub = [r for r in rows if r["split"] == split]
    guard_imgs = []
    for d, _, fs in os.walk(guard_dir):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                guard_imgs.append(os.path.join(d, f))
    if verbose:
        print("守卫集 %s：%d 张；待检 split=%s：%d 张" % (guard_dir, len(guard_imgs), split, len(sub)))

    gmd5 = {}
    for p in guard_imgs:
        gmd5.setdefault(ds.md5_of(p), p)
    n_md5 = 0
    for r in sub:
        if r["md5"] in gmd5:
            n_md5 += 1
            if n_md5 <= 5:
                problems.append("MD5 命中: %s == %s" % (r["src_path"], os.path.relpath(gmd5[r["md5"]], ROOT)))
    if n_md5:
        problems.append("%s 中 %d 张与守卫集字节相同" % (split, n_md5))
    elif verbose:
        print("  MD5 层：无重复")

    if deep and not n_md5:
        descs, vecs, names = [], [], []
        for r in sub:
            g = ds.imread_cn(os.path.join(ROOT, r["src_path"]))
            descs.append(ds.desc_of(g))
            vecs.append(ds.mae_vec_of(g))
            names.append(r["src_path"])
        D = np.stack(descs)
        V = np.stack(vecs)
        n_dupe = 0
        for p in guard_imgs:
            g = ds.imread_cn(p)
            if g is None:
                continue
            d = ds.desc_of(g)
            sim = D @ d
            for k in np.where(sim >= 0.999)[0]:
                if g.shape != (ds._MAE_SIZE, ds._MAE_SIZE):
                    gv = ds.mae_vec_of(g)
                else:
                    gv = g.reshape(-1)
                mae = float(np.abs(V[k].astype(np.int16) - gv.astype(np.int16)).mean())
                if mae <= 1.5:
                    n_dupe += 1
                    if n_dupe <= 5:
                        problems.append("像素级重复: %s <-> %s (cos=%.6f MAE=%.3f)" % (
                            names[k], os.path.relpath(p, ROOT), sim[k], mae))
                    break
        if n_dupe:
            problems.append("%s 中 %d 张与守卫集像素级重复" % (split, n_dupe))
        elif verbose:
            print("  像素层：无重复（cos>=0.999 且 MAE<=1.5）")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--guard", default="", help="守卫数据集目录（如某个训练集）")
    ap.add_argument("--split", default="test")
    ap.add_argument("--deep", action="store_true", help="MD5 无命中时再做像素级复核（慢）")
    args = ap.parse_args()

    rows = ds.load_manifest(args.manifest)
    problems = check_manifest(rows)
    if args.guard:
        problems += check_guard(rows, args.guard, split=args.split, deep=args.deep)

    if problems:
        print("\n[校验未通过] %d 项：" % len(problems))
        for p in problems[:20]:
            print("  -", p)
        sys.exit(1)
    print("\n[校验通过] 冻结清单可用，未发现跨集泄漏或内容漂移")


if __name__ == "__main__":
    main()
