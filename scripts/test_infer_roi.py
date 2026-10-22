# -*- coding: utf-8 -*-
"""
推理链路改造自测（不启 Flask 服务，直接调 web/app.py 里的封装函数）

覆盖：
1. 旧模型（无 _train_meta.json / roi=false）—— 必须与改造前行为一致：全图推理、框坐标不变
2. ROI 模型（roi=true）—— 裁剪后推理，框坐标必须映射回原图，且画图是原图尺寸
3. ROI 有效性检查能拦住"裁剪到空白区"的情况

用法：
  set CUDA_VISIBLE_DEVICES=  (CPU，避免抢占训练显存)
  runtime\\python.exe scripts/test_infer_roi.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # 强制 CPU：训练正在用 GPU
os.environ.setdefault("PROCESSOR_ARCHITECTURE", os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "web"))
sys.path.insert(0, os.path.join(ROOT, "core"))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import dataset_split as ds  # noqa: E402
import roi as R  # noqa: E402

import app as W  # noqa: E402  (会加载 models/启用 的默认模型)


def read_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def main():
    rows = ds.load_manifest("reports/splits/v5_split_manifest.csv")
    # 必须挑一张**真的能检出框**的图，否则坐标映射的断言等于没跑
    cands = [r for r in rows if r["split"] == "test" and int(r["n_boxes"]) > 0][:40]
    if not cands:
        print("找不到测试图")
        return 1
    r = img = res0 = None
    for c in cands:
        _im = read_cn(os.path.join(ROOT, c["src_path"]))
        if _im is None:
            continue
        _rr = W.current_model(_im, imgsz=1920, half=False, device=W.DEVICE,
                              max_det=300, verbose=False, conf=0.05)[0]
        if _rr.boxes is not None and len(_rr.boxes) > 0:
            r, img = c, _im
            print("选用可检出图 %s（conf=0.05 下 %d 个框）" % (c["name"], len(_rr.boxes)))
            break
    if img is None:
        print("前 40 张测试图在 conf=0.05 下均无检出，无法验证坐标映射")
        return 1
    H, W0 = img.shape[:2]
    print("测试图 %s %dx%d，标注 %s 框" % (r["name"], W0, H, r["n_boxes"]))

    mid = W.current_model_id
    info = W.MODEL_REGISTRY[mid]
    spec = info.get("spec") or W._load_model_spec(info["path"])
    print("当前模型 %s -> spec=%s" % (mid, spec))

    # ---------- 1) 旧口径（roi=False） ----------
    info["spec"] = {"imgsz": 1920, "roi": False, "source": "test"}
    res1, draw1 = W.infer_one(W.current_model, img, 0.05)
    b1 = res1.boxes.xyxy.cpu().numpy() if res1.boxes is not None and len(res1.boxes) else np.zeros((0, 4))
    print("\n[1] roi=False: 框 %d 个 | draw %s | 原图 %sx%s"
          % (len(b1), draw1.shape[:2], H, W0))
    assert draw1.shape[:2] == (H, W0), "非 ROI 模式画布尺寸应与原图一致"
    if len(b1):
        assert b1[:, 0].max() <= W0 + 1 and b1[:, 1].max() <= H + 1, "框坐标应落在原图内"

    # ---------- 2) ROI 口径 ----------
    info["spec"] = {"imgsz": 1536, "roi": True, "source": "test"}
    res2, draw2 = W.infer_one(W.current_model, img, 0.01)
    b2 = res2.boxes.xyxy.cpu().numpy() if res2.boxes is not None and len(res2.boxes) else np.zeros((0, 4))
    x0, y0, x1, y1 = R.roi_box(W0, H)
    print("[2] roi=True : 框 %d 个 | draw %s | ROI=(%d,%d,%d,%d)" % (len(b2), draw2.shape[:2], x0, y0, x1, y1))
    assert draw2.shape[:2] == (H, W0), "ROI 模式画布仍应是原图尺寸（框要能贴回去）"
    if len(b2):
        # 映射回原图后，框应落在 ROI 内（允许几像素误差）
        assert b2[:, 0].min() >= x0 - 5 and b2[:, 2].max() <= x1 + 5, \
            "映射后的框应落在 ROI 范围内：%s vs ROI x[%d,%d]" % (b2[:, [0, 2]], x0, x1)
        assert b2[:, 1].min() >= y0 - 5 and b2[:, 3].max() <= y1 + 5, \
            "映射后的框应落在 ROI 范围内：%s vs ROI y[%d,%d]" % (b2[:, [1, 3]], y0, y1)
        print("    映射后框范围 x[%.0f, %.0f] y[%.0f, %.0f] —— 全部落在 ROI 内"
              % (b2[:, 0].min(), b2[:, 2].max(), b2[:, 1].min(), b2[:, 3].max()))

    # ---------- 3) 有效性检查 ----------
    blank = np.full((400, 300, 3), 114, dtype=np.uint8)
    ok_blank = R.roi_looks_valid(blank)
    roi_img, _ = R.crop_roi(img)
    ok_real = R.roi_looks_valid(roi_img)
    print("\n[3] 有效性检查：空白区=%s（应为 False）真实 ROI=%s（应为 True）" % (ok_blank, ok_real))
    assert not ok_blank and ok_real, "有效性检查判据失效"

    # ---------- 4) 批量接口一致性 ----------
    res_b, draw_b = W.infer_batch(W.current_model, [img, img], 0.05)
    nb = [len(x.boxes) if x.boxes is not None else 0 for x in res_b]
    print("[4] infer_batch: 两张图框数 %s | draw 尺寸 %s" % (nb, [d.shape[:2] for d in draw_b]))
    assert len(res_b) == 2 and all(d.shape[:2] == (H, W0) for d in draw_b)

    # ---------- 5) 坐标映射纯逻辑测试（不依赖模型是否有检出） ----------
    b = np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32)
    box = (683, 141, 1330, 1132)
    m = R.map_boxes_from_roi(b, box)
    assert m[0].tolist() == [693.0, 161.0, 733.0, 201.0], m
    back = R.map_boxes_to_roi(m, box)
    assert back[0].tolist() == b[0].tolist(), back
    print("[5] 坐标映射往返一致：%s -> %s -> %s" % (b[0].tolist(), m[0].tolist(), back[0].tolist()))

    class _FB:
        def __init__(self, d): self.data = d
        def __len__(self): return len(self.data)

    class _FR:
        def __init__(self, d): self.boxes = _FB(d)

    fr = _FR(np.array([[10.0, 20.0, 50.0, 60.0]], dtype=np.float32))
    W._restore_boxes(fr, box)
    assert fr.boxes.data[0].tolist() == [693.0, 161.0, 733.0, 201.0], fr.boxes.data
    print("    _restore_boxes 就地映射正确")

    # 恢复原始 spec，避免污染进程状态
    info["spec"] = spec
    print("\n全部断言通过 ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
