# -*- coding: utf-8 -*-
"""检查 考卷 vs V4_AUG（最新训练实际用的数据集）重叠
策略：先文件名，再缩略图快哈希初筛，候选再全像素验证
同时检查 AUG train vs val 内部互串（同原图被增广后分属两边）
"""
import os, hashlib, json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = r"D:\CT Detection\datasets"
KA0 = os.path.join(ROOT, "考卷_清洗版", "images")
AUG_TR = os.path.join(ROOT, "CT_DATASET_FINAL_V4_AUG", "train", "images")
AUG_VA = os.path.join(ROOT, "CT_DATASET_FINAL_V4_AUG", "val", "images")

def quick_hash(p, size=64):
    with Image.open(p) as im:
        im = im.convert("RGB").resize((size, size))
        return hashlib.md5(im.tobytes()).hexdigest()

def full_hash(p):
    with Image.open(p) as im:
        im = im.convert("RGB")
        return hashlib.md5(im.tobytes()).hexdigest()

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
def list_images(d):
    allf = sorted(os.listdir(d))
    imgs = [f for f in allf if os.path.splitext(f)[1].lower() in IMG_EXT]
    others = [f for f in allf if f not in imgs]
    if others:
        print(f"  [注意] {d} 里有 {len(others)} 个非图片文件: {others[:10]}", flush=True)
    return imgs

# 1. 考卷快哈希
kj = list_images(KA0)
kj_q = {}
for f in kj:
    kj_q[f] = quick_hash(os.path.join(KA0, f))
kj_full = {}
print(f"考卷 {len(kj)} 张已哈希", flush=True)

# 2. AUG train/val 逐张快哈希比对
report = {"AUG_train_exact": [], "AUG_val_exact": [], "AUG_train_val_cross": {}}
aug_tr_files = list_images(AUG_TR)
aug_va_files = list_images(AUG_VA)

# 先建 val 快哈希表（用于 train/val 互串检查）
val_q = {}
for i, f in enumerate(aug_va_files):
    val_q[f] = quick_hash(os.path.join(AUG_VA, f))
    if (i+1) % 400 == 0: print(f"  AUG val hash {i+1}/{len(aug_va_files)}", flush=True)
val_q_inv = {}
for f, h in val_q.items():
    val_q_inv.setdefault(h, []).append(f)

# 考卷 vs AUG val：全像素验证同名命中
for f in kj:
    for vf, vh in val_q.items():
        pass  # 下面统一做

kj_q_inv = {}
for f, h in kj_q.items():
    kj_q_inv.setdefault(h, []).append(f)

# AUG val 里与考卷快哈希相同的 → 全像素验证
for vf, vh in val_q.items():
    if vh in kj_q_inv:
        for kf in kj_q_inv[vh]:
            if full_hash(os.path.join(KA0, kf)) == full_hash(os.path.join(AUG_VA, vf)):
                report["AUG_val_exact"].append((kf, vf))
print(f"AUG val vs 考卷 精确重复: {len(report['AUG_val_exact'])}", flush=True)

# AUG train：快哈希比对考卷 + 与 val 互串
tr_val_cross = 0
cross_examples = []
for i, f in enumerate(aug_tr_files):
    h = quick_hash(os.path.join(AUG_TR, f))
    if h in kj_q_inv:
        for kf in kj_q_inv[h]:
            if full_hash(os.path.join(KA0, kf)) == full_hash(os.path.join(AUG_TR, f)):
                report["AUG_train_exact"].append((kf, f))
    if h in val_q_inv:
        tr_val_cross += 1
        if len(cross_examples) < 20:
            cross_examples.append((f, val_q_inv[h]))
    if (i+1) % 2000 == 0: print(f"  AUG train {i+1}/{len(aug_tr_files)} 考卷命中{len(report['AUG_train_exact'])} val互串{tr_val_cross}", flush=True)

report["AUG_train_val_cross_count"] = tr_val_cross
report["AUG_train_val_cross_examples"] = cross_examples

print(f"\nAUG train vs 考卷 精确重复: {len(report['AUG_train_exact'])}")
for kf, tf in report["AUG_train_exact"][:30]:
    print(f"   考卷/{kf} == AUG_train/{tf}")
print(f"AUG train vs AUG val 互串（快哈希，含增广同源）: {tr_val_cross}")
for a, b in cross_examples[:10]:
    print(f"   train/{a} ~~ val/{','.join(b)}")

with open(os.path.join(ROOT, "overlap_report_aug.json"), "w", encoding="utf-8") as fp:
    json.dump(report, fp, ensure_ascii=False, indent=1)
print("\n报告已存 datasets/overlap_report_aug.json")
