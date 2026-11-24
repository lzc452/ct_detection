# -*- coding: utf-8 -*-
"""检查训练/验证/考卷之间的图片重叠（数据污染）
比对方式：像素级 MD5（打开图片解码后哈希，防改名/改扩展名）+ 文件名比对
"""
import os, sys, hashlib, json
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = r"D:\CT Detection\datasets"

SETS = {
    "V4_train":  os.path.join(ROOT, "CT_DATASET_FINAL_V4", "train", "images"),
    "V4_val":    os.path.join(ROOT, "CT_DATASET_FINAL_V4", "val", "images"),
    "考卷":      os.path.join(ROOT, "考卷_清洗版", "images"),
}

def hash_dir(d):
    out = {}
    files = sorted(os.listdir(d))
    for i, f in enumerate(files):
        p = os.path.join(d, f)
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                h = hashlib.md5(im.tobytes()).hexdigest()
        except Exception as e:
            h = "ERR:" + str(e)
        out[f] = h
        if (i + 1) % 500 == 0:
            print(f"  {d}: {i+1}/{len(files)}", flush=True)
    return out

hashes = {}
for name, d in SETS.items():
    print(f"hashing {name} ...", flush=True)
    hashes[name] = hash_dir(d)
    print(f"  done {name}: {len(hashes[name])}", flush=True)

def overlap(a, b):
    ha, hb = hashes[a], hashes[b]
    inv = {}
    for f, h in hb.items():
        inv.setdefault(h, []).append(f)
    hits = []
    for f, h in ha.items():
        if h in inv and not h.startswith("ERR:"):
            hits.append((f, inv[h]))
    return hits

pairs = [("考卷", "V4_train"), ("考卷", "V4_val"), ("V4_train", "V4_val")]

print("\n===== 文件名重叠（同文件名） =====")
for a, b in pairs:
    common = set(hashes[a]) & set(hashes[b])
    print(f"{a} vs {b}: {len(common)} 个同名文件")

print("\n===== 像素级重叠（内容相同，防改名） =====")
report = {}
for a, b in pairs:
    hits = overlap(a, b)
    report[f"{a}_vs_{b}"] = hits
    print(f"{a} vs {b}: {len(hits)} 张内容重复")
    for f, ms in hits[:30]:
        print(f"   {a}/{f}  ==  {b}/{','.join(ms)}")

with open(os.path.join(ROOT, "overlap_report.json"), "w", encoding="utf-8") as fp:
    json.dump(report, fp, ensure_ascii=False, indent=1)
print("\n报告已存 datasets/overlap_report.json")
