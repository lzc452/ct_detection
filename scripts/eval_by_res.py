# -*- coding: utf-8 -*-
"""按原生分辨率分组复算工作点指标，用来定位「精度掉在哪一组」。

背景：ultralytics 验证阶段用 `LetterBox(scaleup=False)`，不倒小图放大；训练与推理
（predict）都放大。ROI 裁剪让低分辨率组变得更小（319x484），如果只按整体看指标，
会把「测量口径造成的低分」误判成「模型变差」。本脚本用与产线一致的 predict 口径
（scaleup=True）按组复算，把两组分开看。

用法：
    ./runtime/python.exe scripts/eval_by_res.py ^
        --model V5=全图@1920:runs/detect/CT_pro/V5_GROUPED_1021_80ep/weights/best.pt:0 ^
        --model V51_ROI=ROI@1536:runs/detect/CT_pro/V51_ROI_1536/weights/best.pt:1

    --model 格式：名称=口径@画布:权重路径:是否ROI(1/0)
产出：
    reports/eval/by_resolution.json
    reports/eval/figs/eval_by_res.png
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "core"))

CLS = ["damage", "dark", "light"]
CN = {"damage": "极片损伤", "dark": "暗斑", "light": "亮斑"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=SPEC@SZ:WEIGHTS:ROI")
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="0")
    ap.add_argument("--roi-images", default="",
                    help="ROI 口径用的已裁剪图目录；默认 .cache/eval_<split>_roi/images")
    args = ap.parse_args()

    import eval_report as E
    rows = [r for r in E.ds.load_manifest(args.manifest) if r["split"] == args.split]
    if not rows:
        raise SystemExit("split=%s 无数据" % args.split)

    # 按 size 分组：2129x1198 为高分辨率，其余算低分辨率
    groups = {}
    for r in rows:
        hi = r["size"] == "2129x1198"
        groups.setdefault("高分辨率 2129x1198" if hi else "低分辨率 ~1052x58x", []).append(r)
    print("测试集分组：" + " | ".join("%s %d 张" % (k, len(v)) for k, v in groups.items()))

    out = {"split": args.split, "conf": args.conf, "iou": args.iou, "models": []}
    for spec in args.model:
        name, wpath, roi = spec.rsplit(":", 2)
        label, sz = name.rsplit("=", 1)
        sz = int(sz.split("@")[-1])
        roi = bool(int(roi))
        entry = {"name": label, "spec": "%s@%d" % ("ROI" if roi else "全图", sz),
                 "weights": wpath, "groups": {}}
        print("\n===== %s (%s) =====" % (label, entry["spec"]))
        for gname, grows in groups.items():
            roi_dir = None
            if roi:
                if args.roi_images:
                    roi_dir = args.roi_images if os.path.isabs(args.roi_images) \
                        else os.path.join(ROOT, args.roi_images)
                else:
                    roi_dir = os.path.join(ROOT, ".cache", "eval_%s_roi" % args.split, "images")
                if not os.path.isdir(roi_dir):
                    raise SystemExit("缺少 ROI 评估集缓存：%s" % roi_dir)
            res = E.layer23_workpoint(wpath, grows, sz, args.conf, args.iou,
                                      args.device, chunk=32, verbose=False,
                                      roi_dir=roi_dir)
            pc = res["per_class"]
            entry["groups"][gname] = {
                "n_images": len(grows),
                "fp_per_1000img": res["fp_per_1000img"],
                "group_miss": res["group_level"]["miss_rate"],
                "n_defect_groups": res["group_level"]["n_defect_groups"],
                "per_class": {c: {k: pc[c][k] for k in ("tp", "fp", "fn", "precision", "recall")}
                              for c in CLS},
            }
            print("  %-20s FP/千张 %6.1f | 组级漏检 %5.1f%% | " % (
                gname, res["fp_per_1000img"], 100 * res["group_level"]["miss_rate"]) +
                " ".join("%s P%.3f/R%.3f" % (CN[c], pc[c]["precision"], pc[c]["recall"])
                         for c in CLS))
        out["models"].append(entry)

    p = os.path.join(ROOT, "reports", "eval", "by_resolution.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已写出 %s" % os.path.relpath(p, ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
