# -*- coding: utf-8 -*-
"""权重规格登记沙盘回归；只读假权重，不初始化YOLO/GPU。
用法：runtime/python.exe -B scripts/test_model_spec.py
"""
import os
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
import sys
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from model_spec import load_model_spec


class ModelSpecTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ct_meta_')
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        self.weight = self.folder / 'best.pt'
        self.weight.write_bytes(b'synthetic')
        self.meta = self.folder / '_train_meta.json'
        self.good = dict(hyperparams=dict(imgsz=1920), roi=False, roi_k=0.0,
                         state='completed', best_pt_sha256=hashlib.sha256(b'synthetic').hexdigest())

    def write(self, data):
        self.meta.write_text(json.dumps(data), encoding='utf8')

    def test_unknown_weight_without_meta_is_rejected(self):
        with self.assertRaises(ValueError):
            load_model_spec(self.weight)

    def test_legacy_compatibility_is_bound_to_hash_not_filename(self):
        with mock.patch('model_spec.LEGACY_FULL_MODELS', {self.good['best_pt_sha256']: 'known-old'}):
            spec = load_model_spec(self.weight)
            self.assertEqual((spec.imgsz, spec.roi), (1920, False))
            self.assertIn('known-old', spec.source)
            self.weight.write_bytes(b'replaced under same name')
            with self.assertRaises(ValueError):
                load_model_spec(self.weight)

    def test_meta_sha256_and_historical_md5_binding(self):
        self.write(self.good)
        self.assertEqual(load_model_spec(self.weight).imgsz, 1920)
        old = dict(self.good)
        old.pop('best_pt_sha256')
        old.pop('roi_k')  # 全图不裁剪，历史meta缺k按0；ROI则不允许猜测。
        old['best_pt_md5'] = hashlib.md5(b'synthetic').hexdigest()
        self.write(old)
        self.assertFalse(load_model_spec(self.weight).roi)
        (self.folder/'weights').mkdir()
        moved = self.folder/'weights/best.pt'
        moved.write_bytes(b'synthetic')
        self.assertEqual(load_model_spec(moved).imgsz, 1920)

    def test_bad_state_hash_classes_or_missing_margin_are_rejected(self):
        for change in [dict(state='failed'), dict(state='running'), dict(best_pt_sha256='0'*64),
                       dict(roi_k=None), dict(roi_k=float('nan')), dict(roi_k=1.5),
                       dict(roi='false'), dict(imgsz=0), dict(classes={'0': 'wrong'})]:
            data = dict(self.good)
            if 'imgsz' in change:
                data['hyperparams'] = dict(imgsz=change['imgsz'])
            else:
                data.update(change)
            self.write(data)
            with self.subTest(change=change), self.assertRaises(ValueError):
                load_model_spec(self.weight)

    def test_corrupt_meta_does_not_fall_back_even_for_known_legacy(self):
        with mock.patch('model_spec.LEGACY_FULL_MODELS', {self.good['best_pt_sha256']: 'known-old'}):
            self.meta.write_text('{bad', encoding='utf8')
            with self.assertRaises(ValueError):
                load_model_spec(self.weight)


if __name__ == '__main__':
    unittest.main(verbosity=2)
