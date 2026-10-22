# -*- coding: utf-8 -*-
"""检测计分回归：真实两个入口必须计数一致，重叠同类框不再因GT顺序漏配。
用法：runtime/python.exe -B scripts/test_detection_metrics.py
仅用合成框和假模型，不读取生产图/权重，不使用GPU，不生成评测缓存。
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from detection_metrics import match_predictions, iou_xyxy, MATCHING_VERSION, LEGACY_MATCHING
from eval_report import layer23_workpoint
from sweep_workpoint import score_workpoints


def gt(box, cls=0):
    return dict(cls=cls, xyxy=box)


def pred(box, conf=0.9, cls=0):
    return dict(cls=cls, conf=conf, xyxy=box)


GT = [gt([0, 0, 10, 10]), gt([4, 0, 14, 10])]
PREDS = [pred([0, 0, 10, 10], 0.8), pred([2, 0, 14, 10], 0.9)]


class Array:
    def __init__(self, value):
        self.value = np.asarray(value)
    def cpu(self):
        return self
    def numpy(self):
        return self.value


class Boxes:
    xyxy = Array([p["xyxy"] for p in PREDS])
    cls = Array([p["cls"] for p in PREDS])
    conf = Array([p["conf"] for p in PREDS])
    def __len__(self):
        return len(PREDS)


class Model:
    def __init__(self, *args):
        pass
    def predict(self, feeds, **kw):
        return [SimpleNamespace(boxes=Boxes()) for _ in feeds]


class MatchingTests(unittest.TestCase):
    def test_overlap_counterexample(self):
        self.assertEqual(match_predictions(PREDS, GT), ({0, 1}, {0, 1}))
        self.assertEqual(len(match_predictions(PREDS, GT, mode=LEGACY_MATCHING)[0]), 1)
        self.assertEqual(match_predictions(PREDS[::-1], GT[::-1]), ({0, 1}, {0, 1}))

    def test_confidence_priority_and_class(self):
        predictions = [pred([0, 0, 10, 10], 0.2), pred([0, 0, 10, 10], 0.9),
                       pred([0, 0, 10, 10], 0.99, 1)]
        self.assertEqual(match_predictions(predictions, GT[:1]), ({1}, {0}))

    def test_empty_and_threshold(self):
        self.assertEqual(match_predictions([], GT), (set(), set()))
        self.assertEqual(match_predictions(PREDS, []), (set(), set()))
        self.assertEqual(match_predictions([pred([0, 0, 10, 10])], GT[:1], 1), ({0}, {0}))
        for threshold in (0, -0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                match_predictions(PREDS, GT, threshold)

    def test_real_entrypoints_share_counts(self):
        row = dict(name="synthetic", size="50x50", src_path="not-read.jpg", label_json="not-read.json",
                   cell_id="", group="g_test")
        normalized = [(g["cls"], (g["xyxy"][0] + g["xyxy"][2]) / 100,
                       (g["xyxy"][1] + g["xyxy"][3]) / 100,
                       (g["xyxy"][2] - g["xyxy"][0]) / 50,
                       (g["xyxy"][3] - g["xyxy"][1]) / 50) for g in GT]
        for mode, expected_tp in ((MATCHING_VERSION, 2), (LEGACY_MATCHING, 1)):
            with self.subTest(mode=mode), mock.patch("ultralytics.YOLO", Model), mock.patch("eval_report.validate_annotation"), mock.patch(
                    "eval_report.ds.json_to_yolo", return_value=(50, 50, normalized)):
                report = layer23_workpoint("fake", [row], 1920, 0.25, 0.5, "cpu", verbose=False, matching=mode)
                point = score_workpoints([row], [GT], [PREDS], [0.25], matching=mode)[0]
                counts = report["per_class"]["damage"]
                self.assertEqual(counts["tp"], expected_tp)
                for key in ("tp", "fp", "fn"):
                    self.assertEqual(point[key], counts[key])
                self.assertEqual(point["group_miss"], report["group_level"]["miss_rate"])
                self.assertEqual(point["fp_per_1000img"], report["fp_per_1000img"])
                self.assertEqual(report["workpoint"]["matching_version"], mode)


    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            score_workpoints([{}], [], [], [0.25])


if __name__ == "__main__":
    unittest.main(verbosity=2)
