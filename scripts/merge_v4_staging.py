# -*- coding: utf-8 -*-
# 待微调_V4捡漏 -> 合并回母本；负样本转正；双类别图并集合并
from pathlib import Path
import json, shutil

ROOT = Path(r"D:\CT Detection")
STAGE = ROOT / "待微调_V4捡漏"
MASTER = ROOT / "问题电芯图片汇总"
PREFIX_DIR = {"0": "0-极片损伤", "1": "1-暗斑", "2": "2-亮斑", "neg": "负样本"}
SUB2CLS = {"损伤_补标": ("0", "0-极片损伤"), "暗斑_补标": ("1", "1-暗斑"), "亮斑_补标": ("2", "2-亮斑")}

def box_key(sh):
    xs = [round(p[0], 1) for p in sh["points"]]; ys = [round(p[1], 1) for p in sh["points"]]
    return (sh["label"], min(xs), min(ys), max(xs), max(ys))

def next_id(folder, prefix):
    nums = [int(p.stem.split("_")[1]) for p in (MASTER / folder).glob(f"{prefix}_*.jpg")
            if p.stem.split("_")[1].isdigit()]
    return max(nums) + 1 if nums else 1

log = []
for sub in sorted(STAGE.iterdir()):
    if not sub.is_dir(): continue
    for jp in sorted(sub.glob("*.json")):
        stem = jp.stem
        doc = json.loads(jp.read_text(encoding="utf-8"))
        if not doc["shapes"]:
            log.append(f"[空标注跳过] {sub.name}/{stem}"); continue
        prefix = stem.split("_")[0]
        if prefix == "neg":
            cls_id, dst_folder = SUB2CLS[sub.name]
            new_stem = f"{cls_id}_{next_id(dst_folder, cls_id):04d}"
            doc["imagePath"] = new_stem + ".jpg"
            shutil.copy2(sub / (stem + ".jpg"), MASTER / dst_folder / (new_stem + ".jpg"))
            (MASTER / dst_folder / (new_stem + ".json")).write_text(
                json.dumps(doc, ensure_ascii=False), encoding="utf-8")
            old_img = MASTER / "负样本" / (stem[4:] + ".jpg")
            if old_img.exists(): old_img.unlink()
            log.append(f"[转正] 负样本/{stem[4:]} -> {dst_folder}/{new_stem}")
        else:
            folder = PREFIX_DIR[prefix]
            dst_json = MASTER / folder / (stem + ".json")
            dst_img = MASTER / folder / (stem + ".jpg")
            if dst_json.exists():
                old = json.loads(dst_json.read_text(encoding="utf-8"))
                keys = {box_key(s) for s in old["shapes"]}
                added = 0
                for s in doc["shapes"]:
                    if box_key(s) not in keys:
                        old["shapes"].append(s); added += 1
                old["imagePath"] = stem + ".jpg"
                dst_json.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
                log.append(f"[合并] {folder}/{stem} 总{len(old['shapes'])}框(新增{added})")
            else:
                doc["imagePath"] = stem + ".jpg"
                dst_json.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
                if not dst_img.exists():
                    shutil.copy2(sub / (stem + ".jpg"), dst_img)
                log.append(f"[新建] {folder}/{stem} {len(doc['shapes'])}框")

# 对照表追加转正记录
with open(MASTER / "重命名对照表.csv", "a", encoding="utf-8-sig") as f:
    for l in log:
        if l.startswith("[转正]"):
            src = l.split("] ")[1].split(" -> ")[0]
            dst = l.split(" -> ")[1]
            f.write(f"\n{dst.split('/')[0]},{src.split('/')[1]}.jpg,{dst.split('/')[1]}.jpg")

for l in log: print(l)
