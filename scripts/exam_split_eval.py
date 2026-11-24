# -*- coding: utf-8 -*-
"""损失量化：把考卷拆成 [66张训练集见过] 和 [130张仅val见过] 两半分别评测 V4
回答一个问题：污染到底把分数抬高了多少
"""
import os, sys, shutil, json, hashlib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = r"D:\CT Detection"
DS = os.path.join(ROOT, "datasets")
TMP = os.path.join(DS, "_exam_split_eval")
MODEL = os.path.join(ROOT, "models", "启用", "20261019_V4", "best.pt")
if not os.path.exists(MODEL):
    MODEL = os.path.join(ROOT, "models", "启用", "20261019_V4", "weights", "best.pt")
print("model:", MODEL, os.path.exists(MODEL))

def full_hash(p):
    with Image.open(p) as im:
        return hashlib.md5(im.convert("RGB").tobytes()).hexdigest()

# 考卷清单
KJ_IMG = os.path.join(DS, "考卷_清洗版", "images")
KJ_LBL = os.path.join(DS, "考卷_清洗版", "labels")

# 训练集哈希（带缓存）
HCACHE = os.path.join(DS, "_train_hashes.json")
tr_dir = os.path.join(DS, "CT_DATASET_FINAL_V4", "train", "images")
if os.path.exists(HCACHE):
    tr_hashes = set(json.load(open(HCACHE)))
    print(f"训练集哈希读缓存 {len(tr_hashes)}", flush=True)
else:
    tr_hashes = set()
    for f in os.listdir(tr_dir):
        if f.lower().endswith((".jpg", ".png")):
            tr_hashes.add(full_hash(os.path.join(tr_dir, f)))
    json.dump(list(tr_hashes), open(HCACHE, "w"))
    print(f"训练集 {len(tr_hashes)} 张哈希完毕", flush=True)

# 考卷分组
seen_train, seen_val = [], []
for f in sorted(os.listdir(KJ_IMG)):
    if not f.lower().endswith((".jpg", ".png")): continue
    h = full_hash(os.path.join(KJ_IMG, f))
    (seen_train if h in tr_hashes else seen_val).append(f)
print(f"考卷拆分: 训练见过 {len(seen_train)} 张 / 仅val见过 {len(seen_val)} 张", flush=True)

# 搭两个临时数据集
def build(name, files):
    d = os.path.join(TMP, name)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(d, sub), exist_ok=True)
    for f in files:
        shutil.copy2(os.path.join(KJ_IMG, f), os.path.join(d, "images", f))
        lb = os.path.splitext(f)[0] + ".txt"
        src = os.path.join(KJ_LBL, lb)
        dst = os.path.join(d, "labels", lb)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            open(dst, "w").close()
    yaml = os.path.join(d, "data.yaml")
    with open(yaml, "w", encoding="utf-8") as fp:
        fp.write(f"path: {d}\ntrain: images\nval: images\nnames:\n  0: 极片损伤\n  1: 暗斑\n  2: 亮斑\n")
    return yaml

y1 = build("seen_train", seen_train)
y2 = build("seen_val", seen_val)

from ultralytics import YOLO
m = YOLO(MODEL)
out = {}
only = sys.argv[1] if len(sys.argv) > 1 else None
for tag, y in (("seen_train", y1), ("seen_val", y2)):
    if only and tag != only:
        continue
    print(f"\n===== 评测 {tag} =====", flush=True)
    r = m.val(data=y, imgsz=1920, batch=8, split="val", verbose=False, plots=False, device=0, half=True, workers=0)
    per_class = {}
    for i, c in enumerate(r.names.values()):
        per_class[c] = {
            "precision": round(float(r.box.p[i]), 4),
            "recall": round(float(r.box.r[i]), 4),
            "map50": round(float(r.box.ap50[i]), 4),
        }
    out[tag] = {
        "map50": round(float(r.box.map50), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "per_class": per_class,
    }
    print(json.dumps(out[tag], ensure_ascii=False, indent=1), flush=True)

RES = os.path.join(DS, "exam_split_eval.json")
old = json.load(open(RES, encoding="utf-8")) if os.path.exists(RES) else {}
old.update(out)
with open(RES, "w", encoding="utf-8") as fp:
    json.dump(old, fp, ensure_ascii=False, indent=1)
print("\n完成, 结果存 datasets/exam_split_eval.json")
