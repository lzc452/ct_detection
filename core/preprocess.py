#coding:utf-8
# -*- coding: utf-8 -*-
"""
预处理核心：分辨率规划 / 缩放策略 / letterbox

背景（2026-10-21 实测，见 docs/preprocess_analysis_20261021.md）：
- 本数据集 **全部是 16:9 横向图**，只有两种原生尺寸：2129x1198(68%) 与 1052x580~585(32%)
- 用 imgsz=1920 的**方形**画布，内容只占 1920x1080 → **画布填充 56%，44% 算力烧在填充区**
- 高分辨率组在 1920 方形下被缩到 0.902；低分辨率组被**上采样 1.825 倍**（插值，无信息增量）
- 缺陷对比度极低（暗斑/亮斑中位仅 3.2 个灰度级），局部噪声 std≈4，
  可检出性靠"多像素平均"（SNR 13~19），所以**任何降低有效像素数的操作都在直接砍 SNR**

结论：画布应**按 16:9 取矩形**，缩放比应**尽量贴近 1.0**，下采样一律用 INTER_AREA。

用法（作为库）：
    from core.preprocess import plan_letterbox, letterbox, resolve_imgsz
    h, w = resolve_imgsz("1088x1920")          # -> (1088, 1920)
    meta = plan_letterbox(W, H, (h, w))
    canvas, meta = letterbox(img, (h, w))

自检：
    runtime\\python.exe core\\preprocess.py --selfcheck
"""
import os
import sys
import json
import argparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 画布高度/宽度都必须是 stride(32) 的整数倍，否则 ultralytics 会自动向上取整
STRIDE = 32
DEFAULT_FILL = 114          # 与 ultralytics 默认 letterbox 填灰一致
NATIVE_ASPECT = 16.0 / 9.0  # 本数据集的固定长宽比


# ============================ IO ============================
def read_cn(path, gray=False):
    """读中文路径图片（OpenCV 原生 imread 在中文路径下会静默失败）"""
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flag)


def write_cn(path, img, quality=95):
    """写中文路径图片"""
    ext = os.path.splitext(path)[1] or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(path)
    return ok


# ============================ 画布规划 ============================
def _ceil32(v):
    return int(np.ceil(float(v) / STRIDE) * STRIDE)


def _round32(v):
    return int(round(float(v) / STRIDE) * STRIDE)


def suggest_canvas(target_width=2112, aspect=NATIVE_ASPECT):
    """按数据集长宽比给出最小浪费的矩形画布 (h, w)，两者都是 32 的倍数。

    例：target_width=1920 -> (1088, 1920)；target_width=2112 -> (1184, 2112)
    """
    w = _ceil32(target_width)
    h = max(STRIDE, _round32(w / aspect))
    return h, w


def resolve_imgsz(spec, default=1920, for_what="train"):
    """把 '2112' / '1184x2112' / 'auto' 统一成 ultralytics 能吃的 imgsz。

    **重要约束（ultralytics 8.4.146 源码确认，2026-10-21 实测）**：
      - trainer.py:340   check_imgsz(..., max_dim=1)  -> 训练 imgsz **只能是 int（方形）**
      - validator.py:141 check_imgsz(..., max_dim=1)  -> val()  imgsz **只能是 int（方形）**
      - predictor.py:274 check_imgsz(..., min_dim=2)  -> predict() 支持 [h, w] 矩形
    传 tuple 给训练/验证会被静默降级成 max(imgsz)（并打 WARNING），不会报错但会让人误以为生效。
    另外 Mosaic.__init__ 里 `-imgsz // 2` 对 tuple 会直接 TypeError。

    因此：
      for_what="train"   -> 永远返回 int（方形画布）
      for_what="predict" -> 允许返回 (h, w) 矩形画布
    """
    if spec is None or spec == "":
        return default
    if isinstance(spec, (int, tuple, list)):
        v = spec
    else:
        s = str(spec).strip().lower()
        if s == "auto":
            v = suggest_canvas(2112)
        elif "x" in s:
            a, b = s.split("x")
            v = (_ceil32(int(a)), _ceil32(int(b)))
        else:
            v = int(s)

    if for_what == "train":
        # 训练/验证只能方形：取长边（与 ultralytics 的降级行为保持一致，但这里是显式的）
        if isinstance(v, (tuple, list)):
            return int(max(v))
        return int(v)
    # 推理允许矩形
    if isinstance(v, (tuple, list)):
        return (int(v[0]), int(v[1]))
    return int(v)


def train_canvas(spec):
    """训练/验证实际生效的方形画布边长（int）"""
    return resolve_imgsz(spec, for_what="train")


def predict_canvas(spec):
    """推理实际生效的画布（int 或 (h, w)），矩形可省 32% 算力"""
    return resolve_imgsz(spec, for_what="predict")


def plan_letterbox(W, H, canvas, fill=DEFAULT_FILL):
    """计算一张 WxH 图放进 canvas=(ch,cw) 时的缩放率、摆放位置与浪费比例。

    返回 dict：
      scale     实际缩放率（<1 下采样；>1 上采样=插值出来的假分辨率）
      new_w/h   缩放后内容尺寸
      pad_x/y   左右/上下填充像素
      fill_ratio 内容占画布比例（越接近 1 越好）
      kernel    建议的插值方式
    """
    ch, cw = (canvas, canvas) if isinstance(canvas, int) else canvas
    scale = min(float(cw) / W, float(ch) / H)
    nw, nh = int(round(W * scale)), int(round(H * scale))
    nw, nh = min(nw, cw), min(nh, ch)
    return {
        "scale": round(scale, 6),
        "new_w": nw, "new_h": nh,
        "pad_x": cw - nw, "pad_y": ch - nh,
        "canvas": (ch, cw),
        "fill_ratio": round(nw * nh / float(ch * cw), 4),
        "upscale": scale > 1.0001,
        "kernel": resize_kernel(scale),
        "fill": fill,
    }


def resize_kernel(scale):
    """缩放比决定插值核：
    - 缩小(scale<1)：INTER_AREA（区域平均=抗混叠，不会把噪声采成假纹理）
    - 放大(scale>1)：INTER_CUBIC（比 LINEAR 略锐，且不会像 LANCZOS 那样过冲）
    - 1:1：不重采样
    """
    if scale < 0.999:
        return cv2.INTER_AREA
    if scale > 1.001:
        return cv2.INTER_CUBIC
    return -1


def letterbox(img, canvas, fill=DEFAULT_FILL, pad_value=True):
    """把图按 plan_letterbox 的结果缩放并居中填充到 canvas，返回 (canvas_img, meta)"""
    H, W = img.shape[:2]
    meta = plan_letterbox(W, H, canvas, fill)
    if meta["kernel"] == -1:
        resized = img
    else:
        resized = cv2.resize(img, (meta["new_w"], meta["new_h"]),
                            interpolation=meta["kernel"])
    ch, cw = meta["canvas"]
    out = np.full((ch, cw, img.shape[2]) if img.ndim == 3 else (ch, cw),
                  fill, dtype=img.dtype)
    x0 = (cw - meta["new_w"]) // 2
    y0 = (ch - meta["new_h"]) // 2
    out[y0:y0 + meta["new_h"], x0:x0 + meta["new_w"]] = resized
    meta["offset"] = (x0, y0)
    return out, meta


def unletterbox_boxes(boxes, meta):
    """把 canvas 坐标的框还原到原图坐标（推理后用）"""
    x0, y0 = meta["offset"]
    s = meta["scale"]
    b = np.asarray(boxes, dtype=np.float32).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - x0) / s
    b[:, [1, 3]] = (b[:, [1, 3]] - y0) / s
    return b


# ============================ 自检 ============================
def selfcheck():
    """用冻结清单打印真实数据的画布利用率对照"""
    sys.path.insert(0, os.path.join(ROOT, "core"))
    import dataset_split as ds

    rows = ds.load_manifest(os.path.join(ROOT, "reports", "splits", "v5_split_manifest.csv"))
    sizes = {}
    for r in rows:
        sizes[r["size"]] = sizes.get(r["size"], 0) + 1
    canvases = [("1920方形", 1920), ("1088x1920矩形", (1088, 1920)), ("1184x2112原生1:1", (1184, 2112))]
    print("%-14s %-18s %8s %10s %10s" % ("原生尺寸", "画布", "张数", "缩放率", "填充率"))
    for size, n in sorted(sizes.items(), key=lambda x: -x[1]):
        W, H = (int(x) for x in size.split("x"))
        for tag, cv_ in canvases:
            m = plan_letterbox(W, H, cv_)
            print("%-14s %-18s %8d %10.3f %9.1f%%" % (size, tag, n, m["scale"], 100 * m["fill_ratio"]))
    print("\n按 16:9 建议画布：")
    for w in (1920, 2112, 2560):
        h, ww = suggest_canvas(w)
        print("  目标宽 %4d -> imgsz=(%d, %d)  画布 %.2f MP" % (w, h, ww, h * ww / 1e6))
    print("\n缩放/插值策略：scale<1 -> INTER_AREA；scale>1 -> INTER_CUBIC；1:1 -> 不重采样")
    print("说明：本数据集的 1052x582 组在目标宽 1920/2112 下都是**上采样**（1.83x / 2.01x），")
    print("      上采样不产生信息，只会把 10px 的亮斑插值成 20px 的模糊斑。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
    else:
        print(__doc__)
