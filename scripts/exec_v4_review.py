# -*- coding: utf-8 -*-
"""
V4 错题本复核结论执行：A 删样本 / B 修标注 / C 打包捡漏待微调。

【安全约定 —— 2026-10-21 改造】
1. 默认 dry-run：只打印操作清单（同时写 _backup/plans/plan_exec_v4_review_<批次>.csv），
   不加 --apply 不会改动任何文件。
2. 写前备份：任何 unlink / 覆盖写 / 移动之前，先把目标复制到
   _backup/<批次>/<原相对路径>/，并写 journal.json。误执行可整批回滚：
       runtime\\python.exe scripts\\review_rollback.py --batch <批次> --apply
3. 幂等：批次号 + 操作编号写进 json 的 ct_meta.op_id；同一批次重复 commit 会跳过。
4. ROOT 由脚本位置推导（不再硬编码 D:\\CT Detection —— 旧工程与本工程是两套树）。

用法：
  # 看清单（不改文件）
  runtime\\python.exe scripts\\exec_v4_review.py
  # 确认后执行
  runtime\\python.exe scripts\\exec_v4_review.py --apply
  # 回滚
  runtime\\python.exe scripts\\exec_v4_review.py --rollback 20261021_213000 --apply
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_safe import Plan, base_parser, rollback   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MASTER = ROOT / "问题电芯图片汇总"
OUT = ROOT / "复核_V4"
STAGE = ROOT / "待微调_V4捡漏"
VAL_IMG = ROOT / "datasets" / "CT_DATASET_FINAL_V4_AUG" / "val" / "images"

PREFIX_DIR = {"0": "0-极片损伤", "1": "1-暗斑", "2": "2-亮斑", "9": "负样本", "neg": "负样本"}
CLS_ID = {"损伤": "0", "暗斑": "1", "亮斑": "2"}
CLS_EN = {"损伤": "damage", "暗斑": "dark", "亮斑": "light"}

DELETE = {"0-极片损伤": [], "1-暗斑": [],
          "2-亮斑": ["2_0244", "2_0417", "2_0426", "2_0351"],
          "负样本": ["9_1334"]}

FIX_STEM = "1_0441"
FIX_FOLDER = "1-暗斑"


# ---------- A. 删除 ----------
def plan_delete(plan):
    n = 0
    hits = []
    for folder, stems in DELETE.items():
        for s in stems:
            hit = False
            for ext in (".jpg", ".json"):
                p = MASTER / folder / (s + ext)
                if plan.delete(p, "A 错题删除 %s/%s" % (folder, s)):
                    n += 1
                    hit = True
            if hit:
                hits.append("%s/%s" % (folder, s))
    if hits:
        # 审计痕迹：删除记录.txt 仍然追加（人工可查），只是内容改由清单驱动
        plan.append_text(MASTER / "删除记录.txt",
                         "\n# 删除记录（复核批次 %s）\n%s\n"
                         % (plan.batch, "\n".join(hits)),
                         "A 追加删除记录")
    print("[A 删除] 计划删除 %d 个文件（%d 组）" % (n, len(hits)))


# ---------- B. 修 1_0441：删掉误标的暗斑框 ----------
def box_of(sh):
    xs = [p[0] for p in sh["points"]]
    ys = [p[1] for p in sh["points"]]
    return [min(xs), min(ys), max(xs), max(ys)]


def plan_fix(plan):
    csv_path = OUT / "错题清单.csv"
    if not csv_path.exists():
        plan.note("[B 跳过] 缺少 %s（复核结论清单）" % csv_path)
        return
    target_box = None
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["图片"] == FIX_STEM + ".jpg" and r["类型"] == "漏检" and r["类别"] == "暗斑":
                target_box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]

    jp = MASTER / FIX_FOLDER / (FIX_STEM + ".json")
    img = MASTER / FIX_FOLDER / (FIX_STEM + ".jpg")
    if not jp.exists():
        plan.note("[B 跳过] %s 不存在（可能已修过或已删除）" % jp)
        return
    doc = json.loads(jp.read_text(encoding="utf-8"))

    if target_box is None:
        plan.note("[B 跳过] 错题清单里没找到 %s 的暗斑漏检框，不做改动" % FIX_STEM)
        return

    before = len(doc.get("shapes", []))
    doc["shapes"] = [sh for sh in doc.get("shapes", [])
                     if not (sh.get("label") == "1"
                             and all(abs(a - b) < 3 for a, b in zip(box_of(sh), target_box)))]
    print("[B 修框] %s: %d -> %d 框" % (FIX_STEM, before, len(doc["shapes"])))

    if doc["shapes"]:
        plan.write_json(jp, doc, "B 修框 %s（删 1 个误标暗斑框）" % FIX_STEM)
        return

    # 删完没有框 → 转负样本（图片改名移入 负样本/，旧 json 删除）
    neg_dir = MASTER / "负样本"
    nums = []
    for p in neg_dir.glob("9_*.jpg"):
        parts = p.stem.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            nums.append(int(parts[1]))
    new_stem = "9_%04d" % ((max(nums) + 1) if nums else 1)
    dst_img = neg_dir / (new_stem + ".jpg")
    if dst_img.exists():
        raise SystemExit("[B 中止] 目标已存在，拒绝覆盖：%s" % dst_img)
    if img.exists():
        plan.move(img, dst_img, "B 转负样本 %s -> 负样本/%s" % (FIX_STEM, new_stem))
    plan.delete(jp, "B 转负样本后删除旧 json")
    plan.append_text(MASTER / "删除记录.txt",
                     "\n# 转负样本（复核批次 %s）\n%s/%s（误标暗斑框删除后无框，转负样本 %s）\n"
                     % (plan.batch, FIX_FOLDER, FIX_STEM, new_stem),
                     "B 追加删除记录")
    print("[B 转负样本] %s -> 负样本/%s" % (FIX_STEM, new_stem))


# ---------- C. 打包捡漏 ----------
EXCLUDE = {("亮斑", "1_1229.jpg"), ("亮斑", "2_0351.jpg"), ("亮斑", "neg_9_1334.jpg")}


def plan_stage(plan):
    csv_path = OUT / "错题清单.csv"
    if not csv_path.exists():
        plan.note("[C 跳过] 缺少 %s" % csv_path)
        return
    fp = {}
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["类型"] != "误报":
                continue
            key = (r["类别"], r["图片"])
            if key in EXCLUDE:
                continue
            fp.setdefault(key, []).append(
                [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])])

    manifest = []
    for (cls, name), boxes in sorted(fp.items()):
        sub = "%s_补标" % cls
        dst = STAGE / sub
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
            import cv2
            import numpy as np
            im = cv2.imdecode(np.fromfile(str(VAL_IMG / name), dtype=np.uint8),
                              cv2.IMREAD_COLOR)
            if im is None:
                plan.note("[C 跳过] 读图失败：%s" % (VAL_IMG / name))
                continue
            ih, iw = im.shape[:2]
        for x1, y1, x2, y2 in boxes:
            shapes.append({"label": CLS_ID[cls], "score": None,
                           "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           "group_id": None, "description": CLS_EN[cls], "difficult": False,
                           "shape_type": "rectangle", "flags": {}, "attributes": {},
                           "kie_linking": []})
        doc2 = {"version": "4.0.6", "flags": {}, "checked": False, "shapes": shapes,
                "imagePath": name, "imageData": None, "imageHeight": ih, "imageWidth": iw}
        plan.write_json(dst / (stem + ".json"), doc2, "C 捡漏标注 %s/%s" % (sub, name))
        plan.copy(VAL_IMG / name, dst / name, "C 捡漏拷贝原图 %s" % name)
        manifest.append("[%s] %s 新增%d框(原有%d框)"
                        % (sub, name, len(boxes), len(shapes) - len(boxes)))

    if manifest:
        plan.write_text(STAGE / "清单.txt",
                        "V4捡漏待微调：模型预标注已写入json，请在X-AnyLabeling中核对微调，"
                        "改完告诉我合并\n\n" + "\n".join(manifest) + "\n",
                        "C 捡漏清单")
    print("[C 打包] %d 张 -> %s" % (len(manifest), STAGE))


def main():
    ap = base_parser(
        "V4 错题本复核结论执行（默认 dry-run）",
        epilog="示例：\n"
               "  runtime\\python.exe scripts\\exec_v4_review.py            # 只出清单\n"
               "  runtime\\python.exe scripts\\exec_v4_review.py --apply    # 落盘（自动备份）\n"
               "  runtime\\python.exe scripts\\review_rollback.py --batch <批次> --apply  # 回滚")
    args = ap.parse_args()

    if args.rollback:
        rollback(ROOT, args.rollback, backup_dir=args.backup_dir, apply=args.apply)
        return
    if not MASTER.is_dir():
        raise SystemExit("[中止] 母本目录不存在：%s（是否在错误的工作区？）" % MASTER)

    plan = Plan(args, label="exec_v4_review", root=ROOT)
    print("[根目录] %s  [批次] %s" % (ROOT, plan.batch))
    plan_delete(plan)
    plan_fix(plan)
    plan_stage(plan)
    if not plan.emit():
        return
    plan.commit()


if __name__ == "__main__":
    main()
