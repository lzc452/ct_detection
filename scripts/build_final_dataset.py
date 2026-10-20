# -*- coding: utf-8 -*-
"""
终版数据集构建（以上级母本"问题电芯图片汇总"为唯一来源）
- 正样本：汇总下 0-极片损伤 / 1-暗斑 / 2-亮斑 三个文件夹，JSON(X-AnyLabeling) 转 YOLO
- 负样本：汇总\\负样本，空标注
- 按内容哈希去重；正样本按类别分层、负样本按编号跨组抽样，85/15 切 train/val
- 输出：datasets\\CT_DATASET_FINAL（已存在则拒绝覆盖）
用法：runtime\\python.exe scripts\\build_final_dataset.py
"""
import os, re, json, glob, random, hashlib, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "问题电芯图片汇总")
OUT = os.path.join(ROOT, "datasets", sys.argv[1] if len(sys.argv) > 1 else "CT_DATASET_FINAL")
POS_DIRS = ["0-极片损伤", "1-暗斑", "2-亮斑"]
VAL_RATIO = 0.15
SEED = 42

def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()

def json_to_yolo(jpath):
    """X-AnyLabeling JSON -> (W, H, [(cls, xc, yc, w, h), ...])；无框返回空表"""
    d = json.load(open(jpath, encoding="utf-8"))
    W, H = d["imageWidth"], d["imageHeight"]
    boxes = []
    for s in d.get("shapes", []):
        try:
            cls = int(s["label"])
        except (ValueError, KeyError):
            continue
        pts = s.get("points", [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x1, x2 = max(0.0, min(xs)), min(float(W), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(H), max(ys))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append((cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H,
                      (x2 - x1) / W, (y2 - y1) / H))
    return W, H, boxes

def main():
    if os.path.exists(OUT):
        print("已存在 %s，如需重建请先手动删除。" % OUT); sys.exit(1)
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            os.makedirs(os.path.join(OUT, split, kind), exist_ok=True)

    seen = set()          # 全局内容去重
    stats = {"pos_train": 0, "pos_val": 0, "neg_train": 0, "neg_val": 0,
             "dup": 0, "nolabel": 0}
    cls_boxes = {"train": [0, 0, 0], "val": [0, 0, 0]}
    rng = random.Random(SEED)

    # ---- 正样本：按类别文件夹分层切分 ----
    for folder in POS_DIRS:
        items = []
        for jf in sorted(glob.glob(os.path.join(SRC, folder, "*.json"))):
            stem = os.path.splitext(os.path.basename(jf))[0]
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = os.path.join(SRC, folder, stem + ext)
                if os.path.exists(p):
                    img = p; break
            if img is None:
                continue
            h = md5(img)
            if h in seen:
                stats["dup"] += 1; continue
            seen.add(h)
            items.append((stem, img, jf))
        k = max(1, int(round(len(items) * VAL_RATIO)))
        val_idx = set(rng.sample(range(len(items)), k))
        for i, (stem, img, jf) in enumerate(items):
            split = "val" if i in val_idx else "train"
            W, H, boxes = json_to_yolo(jf)
            if not boxes:
                stats["nolabel"] += 1
            safe = stem.replace(" ", "_").replace("(", "").replace(")", "")
            dst_img = os.path.join(OUT, split, "images", safe + ".jpg")
            os.rename  # noqa
            with open(img, "rb") as fi, open(dst_img, "wb") as fo:
                fo.write(fi.read())
            with open(os.path.join(OUT, split, "labels", safe + ".txt"), "w") as f:
                for cls, xc, yc, w, h in boxes:
                    f.write("%d %.6f %.6f %.6f %.6f\n" % (cls, xc, yc, w, h))
                    cls_boxes[split][cls] += 1
            stats["pos_" + split] += 1
        print("%s: %d 张（验证集 %d 张）" % (folder, len(items), k))

    # ---- 负样本：按编号排序后跨组等距抽样进验证集 ----
    negs = []
    for f in sorted(os.listdir(os.path.join(SRC, "负样本"))):
        if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue
        p = os.path.join(SRC, "负样本", f)
        h = md5(p)
        if h in seen:
            stats["dup"] += 1; continue
        seen.add(h)
        m = re.search(r"(\d+)", f)
        negs.append((int(m.group(1)) if m else 0, f, p))
    negs.sort()
    step = max(2, round(1 / VAL_RATIO))   # 约每7张抽1张
    for idx, (num, fname, p) in enumerate(negs):
        split = "val" if idx % step == 0 else "train"
        safe = "neg_" + os.path.splitext(fname)[0].replace(" ", "_").replace("(", "").replace(")", "")
        with open(p, "rb") as fi, open(os.path.join(OUT, split, "images", safe + ".jpg"), "wb") as fo:
            fo.write(fi.read())
        open(os.path.join(OUT, split, "labels", safe + ".txt"), "w").close()
        stats["neg_" + split] += 1

    # ---- data.yaml ----
    with open(os.path.join(OUT, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# 终版数据集（母本：问题电芯图片汇总，全部人工标注）\n")
        f.write("path: %s\n" % OUT.replace("\\", "/"))
        f.write("train: train/images\nval: val/images\n\nnc: 3\n")
        f.write("names:\n  0: damage\n  1: dark\n  2: light\n")

    print("\n===== CT_DATASET_FINAL 构建完成 =====")
    print("train: 正%d + 负%d = %d 张 | val: 正%d + 负%d = %d 张" % (
        stats["pos_train"], stats["neg_train"], stats["pos_train"] + stats["neg_train"],
        stats["pos_val"], stats["neg_val"], stats["pos_val"] + stats["neg_val"]))
    print("框数 train: 损伤%d 暗斑%d 亮斑%d | val: 损伤%d 暗斑%d 亮斑%d" % (
        *cls_boxes["train"], *cls_boxes["val"]))
    print("去重跳过 %d 张，无框标注 %d 张" % (stats["dup"], stats["nolabel"]))
    print("输出: %s" % OUT)

if __name__ == "__main__":
    main()
