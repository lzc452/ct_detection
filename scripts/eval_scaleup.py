# -*- coding: utf-8 -*-
"""把验证管线的缩放口径改成与训练/推理一致，量化 `scaleup=False` 造成的测量偏差。

背景
----
`runtime/lib/site-packages/ultralytics/data/dataset.py:319` 在非训练分支里用

    LetterBox(new_shape=(imgsz, imgsz), scaleup=False)

而训练（`v8_transforms`）和推理（`predictor`）都是允许放大的。后果是：凡是
「原图尺寸 < 画布尺寸」的样本，验证时会被**卡在 1.0 倍**，而训练/推理会放大。

这对不同模型的影响极不对称：
- V5 全图 @1920：高分辨率组 2129x1198 → 0.902 倍（缩小，不受影响）
  低分辨率组 1052x582 → 需要 1.83 倍 → **被卡住**（占 11% 样本）
- ROI 模型 @1536：647x991 → 需要 1.55 倍 → **被卡住**（占 89% 样本）
  低分辨率 ROI 319x484 → 需要 3.17 倍 → 被卡住

也就是说，用默认 `model.val` 比 ROI 模型 vs 全图模型，是拿「按 1.0 倍评的 ROI
模型」去比「基本按 0.9 倍评的全图模型」，而产线两条都用 `predict`（会放大）。
本脚本只改这一处测量口径，把两边都拉回训练/推理的尺度，给出可比数字。

注意
----
1. 只 monkeypatch 运行时对象，**不修改 site-packages**，不动训练产物。
2. 输出同时给 `val_default`（现状）与 `val_scaleup`（修正后），差值即偏差。
3. 该修正只影响 mAP 类指标；工作点 P/R/FP/组级漏检走 `predict`，本来就正确。

用法
----
    python scripts/eval_scaleup.py --model "V5=1920:.cache/eval_V5_80ep_v5test/data.yaml:runs/.../best.pt"
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "reports", "eval", "val_scaleup.json")


def _patched_init(self, *a, **kw):
    """模块级函数（可被 multiprocessing pickle）；强制 scaleup。"""
    kw["scaleup"] = _FORCE_SCALEUP[0]
    _ORIG_INIT[0](self, *a, **kw)


_ORIG_INIT = [None]
_FORCE_SCALEUP = [True]


def _patch_scaleup(enable):
    """把 ultralytics 验证分支的 LetterBox.scaleup 强制设为 enable。

    只替换 `LetterBox.__init__`，不新建子类——Windows 的 spawn 模式要给 worker
    进程 pickle 数据集，动态生成的局部类 pick 不了（会报 Can't pickle local object）。
    """
    import ultralytics.data.dataset as ud

    if _ORIG_INIT[0] is None:
        _ORIG_INIT[0] = ud.LetterBox.__init__
    _FORCE_SCALEUP[0] = bool(enable)
    ud.LetterBox.__init__ = _patched_init
    return _ORIG_INIT[0]


def run_one(name, imgsz, data, weights, device):
    from ultralytics import YOLO

    out = {"name": name, "imgsz": int(imgsz), "data": data, "weights": weights}
    for mode, flag in (("val_default", False), ("val_scaleup", True)):
        import ultralytics.data.dataset as ud

        orig = _patch_scaleup(flag)
        try:
            m = YOLO(weights)
            r = m.val(data=data, imgsz=int(imgsz), device=device,
                      verbose=False, plots=False, workers=0)
            names = r.names
            out[mode] = {
                "mAP50": round(float(r.box.map50), 4),
                "mAP50_95": round(float(r.box.map), 4),
                "per_class": {
                    names[i]: {"AP50": round(float(r.box.ap50[i]), 4),
                               "AP50_95": round(float(r.box.ap[i]), 4)}
                    for i in range(len(names))
                },
            }
        finally:
            ud.LetterBox.__init__ = orig
        print("  %-14s %-13s mAP50 %.4f  mAP50-95 %.4f" % (
            name, mode, out[mode]["mAP50"], out[mode]["mAP50_95"]))
    d = out["val_default"]["mAP50_95"]
    u = out["val_scaleup"]["mAP50_95"]
    out["delta_mAP50_95"] = round(u - d, 4)
    out["delta_pct"] = round(100.0 * (u - d) / d, 1) if d else None
    print("  %-14s 修正增益   ΔmAP50-95 %+.4f (%+.1f%%)" % (name, u - d, out["delta_pct"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    metavar="NAME=IMGSZ=DATA=WEIGHTS")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    res = []
    for s in args.model:
        parts = s.split("=", 3)
        if len(parts) != 4:
            raise SystemExit("--model 需写成 NAME=IMGSZ=DATA=WEIGHTS，收到 %r" % s)
        name, imgsz, data, weights = parts
        print("[%s]" % name)
        res.append(run_one(name, imgsz, data, weights, args.device))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"schema": "ct-val-scaleup-v1", "items": res}, f,
                  ensure_ascii=False, indent=2)
    print("\n-> %s" % os.path.relpath(args.out, ROOT))


if __name__ == "__main__":
    main()
