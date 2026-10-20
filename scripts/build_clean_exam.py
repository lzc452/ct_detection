# -*- coding: utf-8 -*-
# 考卷清洗版：V2统一考卷 - 剔除已从母本删除的噪声图，标签按当前母本json重新生成
from pathlib import Path
import json, shutil

ROOT = Path(r"D:\CT Detection")
SRC = ROOT / "datasets" / "CT_DATASET_FINAL_V2_AUG" / "val"
OUT = ROOT / "datasets" / "考卷_清洗版"
MASTER = ROOT / "问题电芯图片汇总"
POS = {"0": "0-极片损伤", "1": "1-暗斑", "2": "2-亮斑"}

(OUT / "images").mkdir(parents=True, exist_ok=True)
(OUT / "labels").mkdir(parents=True, exist_ok=True)

dropped, kept, fixed = [], 0, 0
for img in sorted(SRC.joinpath("images").glob("*.jpg")):
    name = img.name
    stem = img.stem
    master_name = name[4:] if name.startswith("neg_") else name  # neg_9_xxx -> 9_xxx
    prefix = master_name.split("_")[0]
    if prefix in POS:
        mdir = MASTER / POS[prefix]
        mimg, mjson = mdir / master_name, mdir / (Path(master_name).stem + ".json")
        if not mimg.exists():
            dropped.append(name); continue
        doc = json.loads(mjson.read_text(encoding="utf-8"))
        W, H = doc["imageWidth"], doc["imageHeight"]
        lines = []
        for sh in doc["shapes"]:
            xs = [p[0] for p in sh["points"]]; ys = [p[1] for p in sh["points"]]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            lines.append(f"{sh['label']} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
        old_lab = (SRC / "labels" / (stem + ".txt")).read_text().strip() if (SRC / "labels" / (stem + ".txt")).exists() else ""
        if old_lab != "\n".join(lines): fixed += 1
        (OUT / "labels" / (stem + ".txt")).write_text("\n".join(lines), encoding="utf-8")
        shutil.copy2(img, OUT / "images" / name)
        kept += 1
    else:  # 负样本
        mimg = MASTER / "负样本" / master_name
        if not mimg.exists():
            dropped.append(name); continue
        (OUT / "labels" / (stem + ".txt")).write_text("", encoding="utf-8")
        shutil.copy2(img, OUT / "images" / name)
        kept += 1

(OUT / "data.yaml").write_text(
    "# 统一考卷清洗版：剔除噪声图，标签以当前母本为准\n"
    f"path: {OUT}\ntrain: images\nval: images\n\nnc: 3\nnames:\n  0: damage\n  1: dark\n  2: light\n",
    encoding="utf-8")
print(f"保留 {kept} 张，剔除 {len(dropped)} 张: {dropped}")
print(f"其中标签被母本修正的: {fixed} 张")
