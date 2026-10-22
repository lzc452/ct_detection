# -*- coding: utf-8 -*-
"""
量化"数据泄漏把指标吹高了多少"（#10/#11 的可信性核查）

做法（纯只读母本，只写 reports/ 与 .cache/）：
  1. 把冻结 test 的 676 张按"是否出现在 V4 训练集里"拆成两个子集：
       seen   = 与 V4/train 字节相同(md5) 或 像素近重复(cos>=0.999 且 MAE<=1.5)
       unseen = 其余（V4 训练时真正没见过的图）
  2. 用**同一套** layer1 (COCO mAP) / layer2+3 (工作点) 分别在两个子集上评估
  3. 若 V4 在 seen 上明显高于 unseen、而 V5 在两个子集上持平，
     说明 V4 的高分主要来自"背题"，而不是泛化能力

用法：
  runtime\\python.exe scripts\\eval_seen_unseen.py
  runtime\\python.exe scripts\\eval_seen_unseen.py --weights A.pt=V4 --weights B.pt=V5
"""
import os
import sys
import json
import argparse
import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import dataset_split as ds  # noqa: E402
import eval_report as er  # noqa: E402

OUT = os.path.join(ROOT, "reports", "eval")


def build_train_index(train_dir, verbose=True):
    """返回 (md5 集合, 描述子矩阵, mae 向量矩阵)"""
    files = [f for f in sorted(os.listdir(train_dir))
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    md5s, descs, maes = set(), [], []
    for f in files:
        p = os.path.join(train_dir, f)
        md5s.add(ds.md5_of(p))
        img = ds.imread_cn(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        descs.append(ds.desc_of(img))
        maes.append(ds.mae_vec_of(img))
    if verbose:
        print("V4/train 建立索引：%d 张（md5/描述子）" % len(files))
    return md5s, np.stack(descs).astype(np.float32), np.stack(maes).astype(np.float32)


def classify(rows, train_md5, train_desc, train_mae, cos_th, mae_th, verbose=True):
    """给每张 test 图打 seen/unseen 标签"""
    seen, unseen = [], []
    n_md5 = n_dupe = 0
    for i, r in enumerate(rows):
        p = os.path.join(ROOT, r["src_path"])
        if r["md5"] in train_md5:
            seen.append(r)
            n_md5 += 1
            continue
        img = ds.imread_cn(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            unseen.append(r)
            continue
        d = ds.desc_of(img)
        cos = train_desc @ d
        cand = np.where(cos >= cos_th)[0]
        hit = False
        if cand.size:
            m = ds.mae_vec_of(img)
            for j in cand:
                mae = float(np.abs(train_mae[j] - m).mean())
                if mae <= mae_th:
                    hit = True
                    break
        if hit:
            seen.append(r)
            n_dupe += 1
        else:
            unseen.append(r)
    if verbose:
        print("test 拆分：seen %d（md5 命中 %d + 近重复命中 %d）| unseen %d"
              % (len(seen), n_md5, n_dupe, len(unseen)))
    return seen, unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--train-dir", default="datasets/CT_DATASET_FINAL_V4/train/images",
                    help="参照用的旧训练集（判断 test 是否被它污染）")
    ap.add_argument("--weights", action="append", default=None,
                    help="形如 path.pt=标签；可多次传入")
    ap.add_argument("--imgsz", type=int, default=1920)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--cos", type=float, default=0.999)
    ap.add_argument("--mae", type=float, default=1.5)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    ws = args.weights or [
        "models/启用/20261019_V4/best.pt=V4",
        "runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.pt=V5",
    ]
    pairs = []
    for s in ws:
        p, _, tag = s.rpartition("=")
        pairs.append((p or s, tag or os.path.basename(os.path.dirname(p))))

    rows = [r for r in ds.load_manifest(args.manifest) if r["split"] == args.split]
    print("冻结 %s：%d 张" % (args.split, len(rows)))

    train_md5, train_desc, train_mae = build_train_index(
        os.path.join(ROOT, args.train_dir))
    seen, unseen = classify(rows, train_md5, train_desc, train_mae,
                            args.cos, args.mae)

    result = {"schema": "seen_unseen/v1",
              "generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "manifest": args.manifest,
              "reference_train_dir": args.train_dir,
              "cos_th": args.cos, "mae_th": args.mae,
              "n_test": len(rows), "n_seen": len(seen), "n_unseen": len(unseen),
              "subsets": {}}

    for name, sub in (("seen", seen), ("unseen", unseen)):
        if not sub:
            continue
        yaml_path = er.prepare_eval_dataset(sub, "seenunseen_%s" % name)
        for wp, tag in pairs:
            print("\n===== %s on %s (%d 张) =====" % (tag, name, len(sub)))
            l1 = er.layer1_coco(wp, yaml_path, args.imgsz, args.device)
            l23 = er.layer23_workpoint(wp, sub, args.imgsz, args.conf, args.iou,
                                       args.device, verbose=False)
            result["subsets"].setdefault(name, {})[tag] = {
                "n_images": len(sub),
                "weights": wp, "weights_md5": er.md5_file(wp),
                "mAP50": l1["mAP50"], "mAP50_95": l1["mAP50_95"],
                "per_class": {c: {"ap50": v["ap50"], "ap": v["ap"],
                                  "p": v["p"], "r": v["r"]}
                              for c, v in l1["per_class"].items()},
                "workpoint": {
                    "per_class": {c: {"p": d["precision"], "r": d["recall"],
                                      "tp": d["tp"], "fp": d["fp"], "fn": d["fn"]}
                                  for c, d in l23["per_class"].items()},
                    "fp_per_1000img": l23["fp_per_1000img"],
                    "group_miss_rate": l23["group_level"]["miss_rate"],
                },
            }

    out_json = os.path.join(OUT, "seen_unseen_%s.json" % args.split)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 控制台对照
    print("\n" + "=" * 78)
    print("泄漏影响力（同一冻结 test，按\"V4 是否训练时见过\"拆分）")
    print("=" * 78)
    print("%-6s %-8s %6s %8s %10s %10s" % ("模型", "子集", "张数", "mAP50", "mAP50-95", "损伤R@0.5"))
    for name in ("seen", "unseen"):
        for _, tag in pairs:
            v = result["subsets"].get(name, {}).get(tag)
            if not v:
                continue
            dmg = v["workpoint"]["per_class"].get("damage", {})
            print("%-6s %-8s %6d %8.4f %10.4f %10.3f"
                  % (tag, name, v["n_images"], v["mAP50"], v["mAP50_95"],
                     dmg.get("r", float("nan"))))
    print("\n报告：%s" % out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
