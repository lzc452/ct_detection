# -*- coding: utf-8 -*-
"""训练/考核数据卫生全面复查（2026-11-07）
1. 考卷 vs V4 train/val（像素级，复查既有结论）
2. 问题电芯图片汇总（未来V5素材）vs 考卷 → 重建数据集前必须排除
3. V4 train vs val 近似重复（64x64缩略哈希，抓"连续切片"级泄漏）
"""
import os, hashlib, json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = r"D:\CT Detection"
DS = os.path.join(ROOT, "datasets")
HUB = os.path.join(ROOT, "问题电芯图片汇总")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def full_hash(p):
    with Image.open(p) as im:
        return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()

def quick_hash(p, size=64):
    with Image.open(p) as im:
        return hashlib.md5(im.convert("RGB").resize((size, size)).tobytes()).hexdigest()

def iter_images(root, recursive=True):
    out = []
    if recursive:
        for dp, _, fns in os.walk(root):
            for f in fns:
                if f.lower().endswith(IMG_EXT):
                    out.append(os.path.join(dp, f))
    else:
        for f in os.listdir(root):
            if f.lower().endswith(IMG_EXT):
                out.append(os.path.join(root, f))
    return sorted(out)

def hash_set(paths, fn, tag, step=1000):
    m = {}
    for i, p in enumerate(paths):
        try:
            m[p] = fn(p)
        except Exception:
            pass
        if (i + 1) % step == 0:
            print(f"  {tag} {i+1}/{len(paths)}", flush=True)
    return m

report = {}

# ── ① 考卷 vs V4 train/val（像素级复查） ──
print("① 考卷 vs V4 train/val", flush=True)
kj_paths = iter_images(os.path.join(DS, "考卷_清洗版", "images"), False)
kj_full = hash_set(kj_paths, full_hash, "考卷")
for split in ("train", "val"):
    sp = iter_images(os.path.join(DS, "CT_DATASET_FINAL_V4", split, "images"), False)
    sp_full = hash_set(sp, full_hash, f"V4-{split}")
    inv = {}
    for p, h in sp_full.items():
        inv.setdefault(h, []).append(p)
    hits = sorted(os.path.basename(p) for p, h in kj_full.items() if h in inv)
    report[f"考卷_vs_V4_{split}"] = hits
    print(f"  考卷 vs V4 {split}: {len(hits)} 张像素级重复", flush=True)

# ── ② 汇总（V5素材）vs 考卷 ──
print("② 问题电芯图片汇总 vs 考卷", flush=True)
hub_paths = iter_images(HUB, True)
print(f"  汇总共 {len(hub_paths)} 张", flush=True)
hub_full = hash_set(hub_paths, full_hash, "汇总")
inv = {}
for p, h in hub_full.items():
    inv.setdefault(h, []).append(p)
hits2 = {}
for p, h in kj_full.items():
    if h in inv:
        hits2[os.path.basename(p)] = [os.path.relpath(x, HUB) for x in inv[h]]
report["考卷_vs_汇总"] = hits2
print(f"  考卷 vs 汇总: {len(hits2)} 张重复（V5若从汇总建集，这些必须排除出训练侧）", flush=True)

# ── ③ V4 train vs val 近似重复（连续切片级） ──
print("③ V4 train vs val 近似重复扫描", flush=True)
tr_q = hash_set(iter_images(os.path.join(DS, "CT_DATASET_FINAL_V4", "train", "images"), False), quick_hash, "train64")
va_q = hash_set(iter_images(os.path.join(DS, "CT_DATASET_FINAL_V4", "val", "images"), False), quick_hash, "val64")
va_inv = {}
for p, h in va_q.items():
    va_inv.setdefault(h, []).append(os.path.basename(p))
near = {}
for p, h in tr_q.items():
    if h in va_inv:
        near[os.path.basename(p)] = va_inv[h]
report["V4_train_val_近似"] = near
print(f"  近似重复（缩略哈希一致）: {len(near)} 对", flush=True)
for k, v in list(near.items())[:20]:
    print(f"   train/{k} ~~ val/{','.join(v)}", flush=True)

with open(os.path.join(DS, "overlap_report_v2.json"), "w", encoding="utf-8") as fp:
    json.dump(report, fp, ensure_ascii=False, indent=1)
print("\n报告: datasets/overlap_report_v2.json", flush=True)
