# -*- coding: utf-8 -*-
"""待微调_V4捡漏 → 合并回母本；负样本转正；双类别图并集合并。

【安全约定 —— 2026-10-21 改造】
1. 默认 dry-run：只打印"会转正几张 / 会合并几个 json / 会新建几个"，不动任何文件。
2. 写前备份：覆盖母本 json、覆盖图片之前先备份到 _backup/<批次>/，可整批回滚。
3. 幂等：每处理完一个 staging 文件就在 待微调_V4捡漏/.merged/ 落一个标记，
   重跑自动跳过；加 --force 才重做。
4. 编号不再撞车：同一批次里的多次"转正"共享一个自增计数器（旧版按磁盘 max+1
   逐个算，一批里两张负样本会得到同一个新编号，互相覆盖图片）。
5. ROOT 由脚本位置推导（旧版硬编码 D:\\CT Detection）。

用法：
  runtime\\python.exe scripts\\merge_v4_staging.py            # 只出清单
  runtime\\python.exe scripts\\merge_v4_staging.py --apply    # 落盘（自动备份）
  runtime\\python.exe scripts\\merge_v4_staging.py --rollback <批次> --apply   # 回滚
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_safe import Plan, base_parser, rollback   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PREFIX_DIR = {"0": "0-极片损伤", "1": "1-暗斑", "2": "2-亮斑", "neg": "负样本"}
SUB2CLS = {"损伤_补标": ("0", "0-极片损伤"),
           "暗斑_补标": ("1", "1-暗斑"),
           "亮斑_补标": ("2", "2-亮斑")}


def box_key(sh):
    xs = [round(p[0], 1) for p in sh["points"]]
    ys = [round(p[1], 1) for p in sh["points"]]
    return (sh["label"], min(xs), min(ys), max(xs), max(ys))


def existing_ids(master, folder, prefix):
    nums = []
    d = master / folder
    if not d.is_dir():
        return nums
    for p in d.glob("%s_*.jpg" % prefix):
        parts = p.stem.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return nums


def main():
    ap = base_parser("待微调_V4捡漏 合并回母本（默认 dry-run）",
                     epilog="示例：\n"
                            "  runtime\\python.exe scripts\\merge_v4_staging.py            # 只出清单\n"
                            "  runtime\\python.exe scripts\\merge_v4_staging.py --apply    # 落盘")
    args = ap.parse_args()

    root = Path(args.root).resolve() if args.root else ROOT
    if args.rollback:
        rollback(root, args.rollback, backup_dir=args.backup_dir, apply=args.apply)
        return
    STAGE = root / "待微调_V4捡漏"
    MASTER = root / "问题电芯图片汇总"
    if not STAGE.is_dir():
        raise SystemExit('[中止] staging 目录不存在：%s（先跑 exec_v4_review.py --apply 生成）' % STAGE)
    if not MASTER.is_dir():
        raise SystemExit('[中止] 母本目录不存在：%s' % MASTER)

    plan = Plan(args, label="merge_v4_staging", root=root)
    marker_dir = STAGE / ".merged"
    alloc = {}          # folder -> 下一个可用编号（本批次内自增，避免撞号）
    rename_rows = []
    n_conv = n_merge = n_new = n_skip = n_done_marker = 0

    def alloc_id(folder, prefix):
        if folder not in alloc:
            nums = existing_ids(MASTER, folder, prefix)
            alloc[folder] = (max(nums) + 1) if nums else 1
        v = alloc[folder]
        alloc[folder] = v + 1
        return v

    for sub in sorted(STAGE.iterdir()):
        if not sub.is_dir() or sub.name.startswith('.'):
            continue
        for jp in sorted(sub.glob("*.json")):
            stem = jp.stem
            marker = marker_dir / ("%s__%s.json" % (sub.name, stem))
            if marker.exists() and not args.force:
                n_done_marker += 1
                continue
            try:
                doc = json.loads(jp.read_text(encoding="utf-8"))
            except Exception as e:
                plan.note('[跳过] json 读取失败 %s: %s' % (jp, e))
                continue
            if not doc.get("shapes"):
                n_skip += 1
                plan.note('[空标注跳过] %s/%s' % (sub.name, stem))
                continue
            prefix = stem.split("_")[0]

            if prefix == "neg":
                if sub.name not in SUB2CLS:
                    plan.note('[跳过] 未知子目录 %s（负样本转正需要 损伤/暗斑/亮斑_补标）' % sub.name)
                    continue
                cls_id, dst_folder = SUB2CLS[sub.name]
                new_stem = "%s_%04d" % (cls_id, alloc_id(dst_folder, cls_id))
                doc["imagePath"] = new_stem + ".jpg"
                plan.copy(sub / (stem + ".jpg"), MASTER / dst_folder / (new_stem + ".jpg"),
                          "[转正] %s/%s -> %s/%s.jpg" % (sub.name, stem, dst_folder, new_stem))
                plan.write_json(MASTER / dst_folder / (new_stem + ".json"), doc,
                                "[转正] 写标注 %s/%s.json" % (dst_folder, new_stem))
                old_img = MASTER / "负样本" / (stem[4:] + ".jpg")
                plan.delete(old_img, "[转正] 删除负样本原图 %s" % old_img.name)
                rename_rows.append("%s,%s.jpg,%s.jpg" % (dst_folder, stem[4:], new_stem))
                n_conv += 1
            else:
                folder = PREFIX_DIR.get(prefix)
                if folder is None:
                    plan.note('[跳过] 无法识别前缀 %r（%s）' % (prefix, stem))
                    continue
                dst_json = MASTER / folder / (stem + ".json")
                dst_img = MASTER / folder / (stem + ".jpg")
                if dst_json.exists():
                    old = json.loads(dst_json.read_text(encoding="utf-8"))
                    keys = {box_key(s) for s in old.get("shapes", [])}
                    added = 0
                    for s in doc["shapes"]:
                        if box_key(s) not in keys:
                            old["shapes"].append(s)
                            keys.add(box_key(s))
                            added += 1
                    old["imagePath"] = stem + ".jpg"
                    plan.write_json(dst_json, old,
                                    "[合并] %s/%s 共%d框(新增%d)" % (folder, stem,
                                                                   len(old["shapes"]), added))
                    n_merge += 1
                else:
                    doc["imagePath"] = stem + ".jpg"
                    plan.write_json(dst_json, doc, "[新建] %s/%s %d框"
                                    % (folder, stem, len(doc["shapes"])))
                    if not dst_img.exists():
                        plan.copy(sub / (stem + ".jpg"), dst_img, "[新建] 拷贝原图 %s" % stem)
                    n_new += 1

            # 幂等标记：这一项处理完就不会再被重跑（--force 除外）
            plan.write_text(marker, json.dumps({
                "sub": sub.name, "stem": stem,
                "time": "%s" % plan.batch,
            }, ensure_ascii=False), "标记已合并 %s/%s" % (sub.name, stem))

    if rename_rows:
        plan.append_text(MASTER / "重命名对照表.csv",
                         "".join("\n" + r for r in rename_rows) + "\n",
                         "追加转正对照表 %d 行" % len(rename_rows))

    print("[统计] 转正 %d / 合并 %d / 新建 %d / 空标注跳过 %d / 已处理过跳过 %d"
          % (n_conv, n_merge, n_new, n_skip, n_done_marker))
    if not plan.emit():
        return
    plan.commit()


if __name__ == "__main__":
    main()
