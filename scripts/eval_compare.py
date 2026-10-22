# -*- coding: utf-8 -*-
"""
把多个 eval_report.json 合并成一张对照表（固定口径，禁止手抄数字）

用法：
  runtime\\python.exe scripts\\eval_compare.py                      # 默认汇总 reports/eval/*.json
  runtime\\python.exe scripts\\eval_compare.py --out reports/eval/compare.md --split test
"""
import os
import sys
import json
import glob
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)


def fmt(v, n=3):
    return "-" if v is None else ("%.*f" % (n, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="reports/eval/*.json")
    ap.add_argument("--split", default="")
    ap.add_argument("--out", default="reports/eval/compare.md")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
    reps = []
    for f in files:
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if d.get("schema") != "ct-eval-report/1":
            continue
        if args.split and d.get("split") != args.split:
            continue
        reps.append((f, d))
    if not reps:
        print("没有可汇总的报告"); return

    rep0 = reps[0][1]
    lines = []
    lines.append("# 评估对照表（自动生成，勿手改）\n")
    lines.append("- 测试集：`%s` split=%s，%d 张" % (rep0["manifest"], rep0["split"], rep0["n_images"]))
    lines.append("- 清单 md5：`%s`（同一份才是同一张考卷）" % rep0["manifest_md5"])
    lines.append("- 工作点 conf=%s / IoU=%s" % (
        rep0["layer2_3_workpoint"]["workpoint"]["conf"],
        rep0["layer2_3_workpoint"]["workpoint"]["iou"]))

    def spec_of(d):
        """推理口径：是否 ROI 裁剪 + 实际生效的方形画布"""
        sz = d.get("imgsz_layer1") or d.get("imgsz") or "?"
        return "%s@%s" % ("ROI" if d.get("roi") else "全图", sz)

    specs = sorted({spec_of(d) for _, d in reps})
    if len(specs) > 1:
        lines.append("- **⚠ 本表含 %d 种推理口径（%s），不同口径的行不可直接比较**——"
                     "ROI 行只裁了含缺陷的区域、画布也不同，mAP 绝对值只在同口径内可比。"
                     % (len(specs), " / ".join(specs)))
    else:
        lines.append("- 推理口径：%s（全表一致）" % specs[0])
    lines.append("")
    lines.append("| 报告 | 口径 | mAP50 | mAP50-95 | 损伤 AP50/AP | 暗斑 AP50/AP | 亮斑 AP50/AP | 损伤 P/R | 暗斑 P/R | 亮斑 P/R | FP/千张 | 组级漏检率 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")

    for f, d in reps:
        l1 = d.get("layer1_coco", {})
        l23 = d.get("layer2_3_workpoint", {})
        pc = l1.get("per_class", {})
        wp = l23.get("per_class", {})
        gl = l23.get("group_level", {})

        def ap(c):
            x = pc.get(c, {})
            return "%s / %s" % (fmt(x.get("ap50")), fmt(x.get("ap")))

        def pr(c):
            x = wp.get(c, {})
            return "%s / %s" % (fmt(x.get("precision")), fmt(x.get("recall")))

        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            os.path.basename(f), spec_of(d),
            fmt(l1.get("mAP50")), fmt(l1.get("mAP50_95")),
            ap("damage"), ap("dark"), ap("light"),
            pr("damage"), pr("dark"), pr("light"),
            fmt(l23.get("fp_per_1000img"), 1),
            "%s (%d/%d)" % (fmt(gl.get("miss_rate")), gl.get("n_missed_groups", 0), gl.get("n_defect_groups", 0)),
        ))

    lines.append("")
    lines.append("> 注：组级漏检率 = 有缺陷组中「至少一个缺陷未被检出」的比例；电芯级（有电芯号的子集）样本过少，不作结论。")
    lines.append("> 注：宽松口径（IoU≥0.1 或中心互落）只作业务参考，禁止与 IoU0.5 口径并列比较。")

    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    print("\n已写出 %s" % args.out)


if __name__ == "__main__":
    main()
