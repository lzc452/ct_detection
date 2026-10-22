# -*- coding: utf-8 -*-
"""
按冻结清单落地数据集（#10 的落地环节）

原则：
- 只读母本；图片用 **硬链接**（同卷 NTFS，零额外空间），绝不复制、绝不移动原始文件
- 训练数据完全由 `split_manifest.csv` 决定 —— 训练集与"冻结清单"逐字节一致，可复核
- 输出目录默认 `datasets_v5/`（worktree 内、已 gitignore），不写共享 datasets/

用法：
  runtime\\python.exe scripts\\materialize_dataset.py --name CT_DATASET_V5_GROUPED
  runtime\\python.exe scripts\\materialize_dataset.py --splits train,val --verify
"""
import os
import sys
import shutil
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))

import dataset_split as ds  # noqa: E402


def link_or_copy(src, dst):
    if os.path.exists(dst):
        return "exists"
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--name", default="CT_DATASET_V5_GROUPED")
    ap.add_argument("--root", default="datasets_v5", help="落地根目录（默认 worktree 内）")
    ap.add_argument("--splits", default="train,val", help="要落地的 split；test 留给评估用")
    ap.add_argument("--force", action="store_true", help="已存在时先删除重建")
    args = ap.parse_args()

    rows = ds.load_manifest(args.manifest)
    wants = [s.strip() for s in args.splits.split(",") if s.strip()]
    out = os.path.join(ROOT, args.root, args.name)
    if os.path.isdir(out) and args.force:
        shutil.rmtree(out)
    if os.path.isdir(out) and not args.force:
        print("已存在 %s；如需重建加 --force" % out)
        return 1

    total = {"link": 0, "copy": 0, "exists": 0, "missing": 0}
    for sp in wants:
        img_dir = os.path.join(out, sp, "images")
        lab_dir = os.path.join(out, sp, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)
        n = 0
        for r in rows:
            if r["split"] != sp:
                continue
            src = os.path.join(ROOT, r["src_path"])
            if not os.path.exists(src):
                total["missing"] += 1
                continue
            total[link_or_copy(src, os.path.join(img_dir, r["name"] + ".jpg"))] += 1
            lines = []
            if r["label_json"]:
                _, _, boxes = ds.json_to_yolo(os.path.join(ROOT, r["label_json"]))
                for cls, xc, yc, bw, bh in boxes:
                    lines.append("%d %.6f %.6f %.6f %.6f" % (cls, xc, yc, bw, bh))
            with open(os.path.join(lab_dir, r["name"] + ".txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            n += 1
        print("  %-5s %4d 张 -> %s" % (sp, n, os.path.relpath(img_dir, ROOT)))

    yaml_path = os.path.join(out, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# 由 scripts/materialize_dataset.py 按冻结清单生成（图片为硬链接，母本未改动）\n")
        f.write("# 冻结清单：%s\n" % args.manifest.replace("\\", "/"))
        f.write("path: %s\n" % out.replace("\\", "/"))
        f.write("train: train/images\nval: val/images\n\n")
        f.write("nc: 3\nnames:\n  0: damage\n  1: dark\n  2: light\n")

    print("\n落地完成：%s" % out)
    print("  硬链接 %d 张 | 复制 %d 张 | 跳过已存在 %d | 缺失 %d" % (
        total["link"], total["copy"], total["exists"], total["missing"]))
    print("  data.yaml: %s" % yaml_path)
    if total["missing"]:
        print("  [警告] 有文件缺失，训练前请先跑 verify_split.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
