# -*- coding: utf-8 -*-
"""评测缓存回归：用两张以内合成图验证ROI参数/标签指纹与异常输入拒绝。
用法：runtime/python.exe -B scripts/test_eval_cache.py
所有图片与缓存均在独立临时沙盘；不访问生产目录，不加载权重或使用GPU。
"""
import os
import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
import cv2
import numpy as np
import roi
from eval_report import prepare_eval_dataset


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ct_cache_")
        self.base = Path(self.temp.name) / "中文沙盘"
        self.base.mkdir()
        self.image = self.base / "source.jpg"
        self.label = self.base / "source.json"
        rng = np.random.default_rng(7)
        ok, data = cv2.imencode(".jpg", rng.integers(0, 255, (80, 128, 3), dtype=np.uint8))
        self.assertTrue(ok)
        data.tofile(str(self.image))
        self.annotation = dict(imageWidth=128, imageHeight=80,
                               shapes=[dict(label="0", points=[[50, 25], [60, 35]])])
        self.write_label()
        self.row = dict(name="sample", src_path=str(self.image), label_json=str(self.label),
                        size="128x80", n_boxes="1", md5=hashlib.md5(self.image.read_bytes()).hexdigest())
        self.patch = mock.patch("eval_report.CACHE", str(self.base / "cache"))
        self.patch.start()
        roi.set_margin_k(0)

    def tearDown(self):
        roi.set_margin_k(0)
        self.patch.stop()
        self.temp.cleanup()

    def write_label(self):
        self.label.write_text(json.dumps(self.annotation), encoding="utf-8")

    def prepare(self):
        return prepare_eval_dataset([self.row], "test", verbose=False, roi=True)

    def test_identical_reuse(self):
        first = self.prepare()
        self.assertEqual(first, self.prepare())
        self.assertTrue(Path(first).parent.joinpath("cache_contract.json").is_file())

    def test_margin_changes_key(self):
        first = self.prepare()
        roi.set_margin_k(1.5)
        second = self.prepare()
        self.assertNotEqual(first, second)
        a = cv2.imdecode(np.fromfile(str(Path(first).parent / "images/sample.jpg"), np.uint8), cv2.IMREAD_COLOR)
        b = cv2.imdecode(np.fromfile(str(Path(second).parent / "images/sample.jpg"), np.uint8), cv2.IMREAD_COLOR)
        self.assertNotEqual(a.shape, b.shape)

    def test_label_content_changes_key_without_count_change(self):
        first = self.prepare()
        self.annotation["shapes"][0]["points"][0][0] += 1
        self.write_label()
        second = self.prepare()
        self.assertNotEqual(first, second)
        self.assertNotEqual((Path(first).parent / "labels/sample.txt").read_text(),
                            (Path(second).parent / "labels/sample.txt").read_text())

    def test_corrupt_cache_rejected(self):
        first = self.prepare()
        (Path(first).parent / "labels/sample.txt").write_text("wrong")
        with self.assertRaisesRegex(ValueError, "缓存内容"):
            self.prepare()

    def test_missing_annotation_rejected(self):
        self.label.unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()

    def test_source_drift_rejected(self):
        self.image.write_bytes(self.image.read_bytes() + b"drift")
        with self.assertRaisesRegex(ValueError, "冻结清单"):
            self.prepare()

    def test_invalid_class_and_dropped_gt_rejected(self):
        self.annotation["shapes"][0]["label"] = "3"
        self.write_label()
        with self.assertRaisesRegex(ValueError, "未知类别"):
            self.prepare()
        self.annotation["shapes"][0] = dict(label="0", points=[[0, 0], [10, 10]])
        self.write_label()
        with self.assertRaisesRegex(ValueError, "丢弃"):
            self.prepare()


if __name__ == "__main__":
    unittest.main(verbosity=2)
