# -*- coding: utf-8 -*-
# V4错题本复核结论执行：A删样本 B修标注 C打包捡漏待微调
from pathlib import Path
import csv, json, shutil, datetime

ROOT = Path(r"D:\CT Detection")
MASTER = ROOT / "问题电芯图片汇总"
OUT = ROOT / "复核_V4"
STAGE = ROOT / "待微调_V4捡漏"
VAL_IMG = ROOT / "datasets/CT_DATASET_FINAL_V4_AUG/val/images"
PREFIX_DIR = {"0": "0-极片损伤", "1": "1-暗斑", "2": "2-亮斑", "9": "负样本", "neg": "负样本"}
CLS_ID = {"损伤": "0", "暗斑": "1", "亮斑": "2"}
CLS_EN = {"损伤": "damage", "暗斑": "dark", "亮斑": "light"}

DELETE = {"0-极片损伤": [], "1-暗斑": [], "2-亮斑": ["2_0244", "2_0417", "2_0426", "2_0351"],
          "负样本": ["9_1334"]}

# ---------- A. 删除 ----------
today = datetime.date.today().isoformat()
del_log = [f"# 删除记录 {today}（V4错题本复核）"]
for folder, stems in DELETE.items():
    for s in stems:
        hit = False
        for ext in (".jpg", ".json"):
            p = MASTER / folder / (s + ext)
            if p.exists(): p.unlink(); hit = True
        if hit: del_log.append(f"{folder}/{s}"); print(f"[删除] {folder}/{s}")

# ---------- B. 修 1_0441：删掉误标的暗斑框 ----------
# 从错题清单找到该框坐标
target_box = None
with open(OUT / "错题清单.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["图片"] == "1_0441.jpg" and r["类型"] == "漏检" and r["类别"] == "暗斑":
            target_box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
jp = MASTER / "1-暗斑" / "1_0441.json"
doc = json.loads(jp.read_text(encoding="utf-8"))
def box_of(sh):
    xs = [p[0] for p in sh["points"]]; ys = [p[1] for p in sh["points"]]
    return [min(xs), min(ys), max(xs), max(ys)]
if target_box:
    before = len(doc["shapes"])
    doc["shapes"] = [sh for sh in doc["shapes"]
                     if not (sh["label"] == "1" and
                             all(abs(a-b) < 3 for a, b in zip(box_of(sh), target_box)))]
    print(f"[修框] 1_0441: {before} -> {len(doc['shapes'])} 框")
if not doc["shapes"]:
    # 无框 -> 转负样本
    neg_dir = MASTER / "负样本"
    nums = [int(p.stem.split("_")[1]) for p in neg_dir.glob("9_*.jpg") if p.stem.split("_")[1].isdigit()]
    new_stem = f"9_{max(nums)+1:04d}"
    shutil.move(str(MASTER / "1-暗斑" / "1_0441.jpg"), str(neg_dir / (new_stem + ".jpg")))
    jp.unlink()
    del_log.append(f"1-暗斑/1_0441（误标暗斑框删除后无框，转负样本 {new_stem}）")
    print(f"[转负样本] 1_0441 -> 负样本/{new_stem}")
else:
    jp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

with open(MASTER / "删除记录.txt", "a", encoding="utf-8") as f:
    f.write("\n" + "\n".join(del_log) + "\n")

# ---------- C. 打包捡漏 ----------
EXCLUDE = {("亮斑", "1_1229.jpg"), ("亮斑", "2_0351.jpg"), ("亮斑", "neg_9_1334.jpg")}
fp = {}
with open(OUT / "错题清单.csv", encoding="utf-8-sig") as f:
    for r in csv.DictReader(f):
        if r["类型"] != "误报": continue
        key = (r["类别"], r["图片"])
        if key in EXCLUDE: continue
        fp.setdefault(key, []).append([float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])])

manifest = []
for (cls, name), boxes in sorted(fp.items()):
    sub = f"{cls}_补标"
    dst = STAGE / sub
    dst.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem
    prefix = stem.split("_")[0]
    master_stem = stem[4:] if prefix == "neg" else stem
    mjson = MASTER / PREFIX_DIR[prefix] / (master_stem + ".json")
    shapes, iw, ih = [], None, None
    if mjson.exists():
        old = json.loads(mjson.read_text(encoding="utf-8"))
        shapes = old.get("shapes", [])
        iw, ih = old.get("imageWidth"), old.get("imageHeight")
    if iw is None:
        import numpy as np, cv2
        im = cv2.imdecode(np.fromfile(str(VAL_IMG / name), dtype=np.uint8), cv2.IMREAD_COLOR)
        ih, iw = im.shape[:2]
    for x1, y1, x2, y2 in boxes:
        shapes.append({"label": CLS_ID[cls], "score": None,
                       "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                       "group_id": None, "description": CLS_EN[cls], "difficult": False,
                       "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []})
    doc2 = {"version": "4.0.6", "flags": {}, "checked": False, "shapes": shapes,
            "imagePath": name, "imageData": None, "imageHeight": ih, "imageWidth": iw}
    (dst / (stem + ".json")).write_text(json.dumps(doc2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(VAL_IMG / name, dst / name)
    manifest.append(f"[{sub}] {name} 新增{len(boxes)}框(原有{len(shapes)-len(boxes)}框)")

(STAGE / "清单.txt").write_text(
    "V4捡漏待微调：模型预标注已写入json，请在X-AnyLabeling中核对微调，改完告诉我合并\n\n"
    + "\n".join(manifest), encoding="utf-8")
print(f"\n[打包] {len(manifest)} 张 -> {STAGE}")
for m in manifest: print(" ", m)
