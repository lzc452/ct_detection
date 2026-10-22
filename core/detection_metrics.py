# -*- coding: utf-8 -*-
"""检测工作点的唯一匹配实现：同类、按置信度降序、最大IoU一对一。
legacy-first仅用于复算旧预注册报告；本模块不依赖模型或GPU。
"""
import math

MATCHING_VERSION = "max-iou-v2"
LEGACY_MATCHING = "legacy-first"


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_predictions(predictions, ground_truth, threshold=0.5, mode=MATCHING_VERSION):
    """返回已配预测索引与已配GT索引；输入列表不改写。

    v2取最大IoU；legacy保留旧报告先遇先配。v2置信度/IoU同分时用几何坐标稳定排序。
    """
    if mode not in (MATCHING_VERSION, LEGACY_MATCHING):
        raise ValueError("未知匹配版本: " + str(mode))
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("匹配IoU必须在(0, 1]内")
    used_p, used_g = set(), set()
    if mode == LEGACY_MATCHING:
        order = sorted(range(len(predictions)), key=lambda i: -predictions[i]["conf"])
    else:
        order = sorted(range(len(predictions)), key=lambda i: (
            -predictions[i]["conf"], predictions[i]["cls"], tuple(predictions[i]["xyxy"])))
    for pi in order:
        pred = predictions[pi]
        candidates = [(gi, iou_xyxy(pred["xyxy"], gt["xyxy"]))
                      for gi, gt in enumerate(ground_truth)
                      if gi not in used_g and gt["cls"] == pred["cls"]]
        candidates = [(gi, overlap) for gi, overlap in candidates if overlap >= threshold]
        if not candidates:
            continue
        if mode == LEGACY_MATCHING:
            gi = candidates[0][0]
        else:
            gi = min(candidates, key=lambda item: (
                -item[1], tuple(ground_truth[item[0]]["xyxy"]), item[0]))[0]
        used_p.add(pi)
        used_g.add(gi)
    return used_p, used_g
