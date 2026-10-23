# -*- coding: utf-8 -*-
"""
torchmetrics 对拍：用第三方实现重算本项目的层1 mAP，验证自研评估链路没有口径漂移。

为什么值得做：本项目层2/3 是自研匹配器，层1 借 ultralytics.val。两者都跑在"自己人"写的代码上，
一旦出现 L-023 那类静默错位（参数传错、坐标系不一致），指标全错但不报错。
第三方实现是唯一能独立复现的尺子。

用法：
  references\\.venv\\Scripts\\python.exe scripts\oss_map_crosscheck.py \\\\
      --preds reports/oss_bench/preds_yolo_lowconf.json
"""
import os
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
import sys
import json
import argparse

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "core"))
import dataset_split as ds  # noqa: E402

CLS = ["damage", "dark", "light"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="reports/oss_bench/preds_yolo_lowconf.json")
    ap.add_argument("--manifest", default="reports/splits/v5_split_manifest.csv")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ref-json", default="reports/eval/V5_80ep_v5test.json")
    args = ap.parse_args()

    import torch
    from torchmetrics.detection import MeanAveragePrecision

    with open(args.preds, encoding="utf-8") as f:
        preds_by_name = json.load(f)
    rows = [r for r in ds.load_manifest(args.manifest) if r["split"] == args.split]

    p_list, t_list, n_gt, n_pred = [], [], 0, 0
    for r in rows:
        W, H = [int(x) for x in r["size"].split("x")]
        gb, gl = [], []
        if r["label_json"]:
            _, _, boxes = ds.json_to_yolo(os.path.join(ROOT, r["label_json"]))
            for cls, xc, yc, bw, bh in boxes:
                gb.append([(xc - bw / 2) * W, (yc - bh / 2) * H,
                           (xc + bw / 2) * W, (yc + bh / 2) * H])
                gl.append(int(cls))
        t_list.append({"boxes": torch.tensor(gb, dtype=torch.float32).reshape(-1, 4),
                       "labels": torch.tensor(gl, dtype=torch.long)})
        n_gt += len(gb)

        pb, ps, pl = [], [], []
        for p in preds_by_name.get(r["name"], []):
            pb.append(p["xyxy"])
            ps.append(p["conf"])
            pl.append(p["cls"])
        p_list.append({"boxes": torch.tensor(pb, dtype=torch.float32).reshape(-1, 4),
                       "scores": torch.tensor(ps, dtype=torch.float32),
                       "labels": torch.tensor(pl, dtype=torch.long)})
        n_pred += len(pb)

    # 显式指定后端：Windows 上 pycocotools 需要 MSVC 编译，faster-coco-eval 是纯 Python 实现
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox",
                                 class_metrics=True, extended_summary=False,
                                 backend="faster_coco_eval")
    metric.update(p_list, t_list)
    res = metric.compute()

    out = {
        "n_images": len(rows), "n_gt": n_gt, "n_pred_boxes": n_pred,
        "torchmetrics": {
            "mAP50_95": round(float(res["map"]), 4),
            "mAP50": round(float(res["map_50"]), 4),
            "mAP75": round(float(res["map_75"]), 4),
            "per_class_ap50_95": [round(float(v), 4) for v in np.atleast_1d(res["map_per_class"])],
        },
    }
    if os.path.isfile(args.ref_json):
        with open(args.ref_json, encoding="utf-8") as f:
            ref = json.load(f)["layer1_coco"]
        out["project_layer1"] = {
            "mAP50_95": round(ref["mAP50_95"], 4), "mAP50": round(ref["mAP50"], 4),
            "mAP75": round(ref["mAP75"], 4),
            "per_class_ap50_95": [round(ref["per_class"][c]["ap"], 4) for c in CLS],
        }
        out["delta"] = {
            "mAP50_95": round(out["torchmetrics"]["mAP50_95"] - out["project_layer1"]["mAP50_95"], 4),
            "mAP50": round(out["torchmetrics"]["mAP50"] - out["project_layer1"]["mAP50"], 4),
        }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    p = os.path.join(ROOT, "reports", "oss_bench", "map_crosscheck.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写:", p)


if __name__ == "__main__":
    main()
