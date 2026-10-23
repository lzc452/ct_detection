# -*- coding: utf-8 -*-
"""BGR输入、中文输出和保存失败台账回归（合成图片，不占GPU）。
用法：runtime/python.exe -B scripts/test_image_contract.py
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock
import numpy as np
import torch
from ultralytics.engine.predictor import BasePredictor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'core'))
from image_io import read_bgr_upload, read_bgr_disk
from test_web_model_state import ModelStateTests, Model, png_bytes


class ImageContractTests(unittest.TestCase):
    setUp = ModelStateTests.setUp
    records = ModelStateTests.records
    task = ModelStateTests.task

    def red(self):
        rgb = np.zeros((64, 96, 3), np.uint8)
        rgb[:] = [255, 0, 0]
        return rgb

    def test_actual_ultralytics_preprocessor_requires_bgr(self):
        predictor = BasePredictor.__new__(BasePredictor)
        predictor.device = torch.device('cpu')
        predictor.model = SimpleNamespace(fp16=False)
        predictor.pre_transform = lambda images: images
        rgb = self.red()
        wrong = predictor.preprocess([rgb])[0, :, 0, 0].tolist()
        bgr = read_bgr_upload(png_bytes(rgb))
        corrected = predictor.preprocess([bgr])[0, :, 0, 0].tolist()
        self.assertEqual(wrong, [0., 0., 1.])
        self.assertEqual(corrected, [1., 0., 0.])

    def test_upload_disk_and_saved_color_agree_in_chinese_directory(self):
        rgb = self.red()
        path = self.base / '中文图.png'
        path.write_bytes(png_bytes(rgb).getvalue())
        bgr = read_bgr_disk(path)
        np.testing.assert_array_equal(read_bgr_upload(png_bytes(rgb)), bgr)
        self.web.RESULTS_FOLDER = str(self.base / '中文结果')
        Path(self.web.RESULTS_FOLDER).mkdir()
        for name in (self.web.save_annotated_image(bgr, '色彩.png'),
                     self.web.save_original_image(bgr)):
            self.assertTrue(name)
            pixel = read_bgr_disk(Path(self.web.RESULTS_FOLDER) / name)[32, 48]
            self.assertGreater(int(pixel[2]), 250)
            self.assertLess(int(pixel[0]), 5)

    def test_both_upload_routes_pass_bgr_to_model(self):
        original = Model.__call__
        seen = []
        def capture(model, source, **kw):
            seen.extend([im.copy() for im in source])
            return original(model, source, **kw)
        for route in ('image', 'batch'):
            with mock.patch.object(Model, '__call__', capture):
                response = self.client.post('/api/detect/'+route, data={
                    'file' if route == 'image' else 'files': (png_bytes(self.red()), 'red.png'),
                    'batch_id': route}, content_type='multipart/form-data')
            self.assertTrue(response.get_json()['success'], response.get_json())
            self.assertEqual(seen[-1][0, 0].tolist(), [0, 0, 255])

    def test_encoder_false_does_not_return_nonexistent_filename(self):
        with mock.patch.object(self.web.cv2, 'imencode', return_value=(False, None)):
            self.assertIsNone(self.web.save_annotated_image(self.red(), 'a.png'))
            self.assertIsNone(self.web.save_original_image(self.red()))

    def test_upload_save_failures_are_failed_without_detection_rows(self):
        for route in ('image', 'batch'):
            for helper in ('save_annotated_image', 'save_original_image'):
                batch = route + helper
                with mock.patch.object(self.web, helper, return_value=None):
                    response = self.client.post('/api/detect/'+route, data={
                        'file' if route == 'image' else 'files': (png_bytes(self.red()), 'red.png'),
                        'batch_id': batch}, content_type='multipart/form-data')
                payload = response.get_json()
                status = payload['status'] if route == 'image' else payload['results'][0]['status']
                self.assertEqual(status, 'failed', payload)
                records = self.records(batch)
                self.assertEqual(len(records), 1)
                self.assertEqual(records[0]['状态'], 'failed')
                self.assertFalse([r for r in self.web.detection_results if r['批次ID'] == batch])
                self.assertFalse([r for r in self.web.problem_images if r['批次ID'] == batch])

    def test_worker_saves_fail_without_defect_success_or_detection_rows(self):
        path = self.base / 'red.png'
        path.write_bytes(png_bytes(self.red()).getvalue())
        for helper in ('save_annotated_image', 'backup_original_file'):
            self.task(helper, 1, helper)
            with mock.patch.object(self.web, helper, return_value=None):
                self.web._detect_worker(helper, str(self.base), [(str(path), 'red.png')],
                                        .3, 0, helper, self.web.snapshot_model())
            task = self.web.detect_tasks[helper]
            self.assertEqual((task['errors'], task['defect_imgs']), (1, 0))
            self.assertEqual(self.records(helper)[0]['状态'], 'failed')
            self.assertFalse([r for r in self.web.detection_results if r['批次ID'] == helper])

    def test_legacy_color_contract_cannot_continue_same_upload_batch(self):
        old = self.web._model_contract(self.web.snapshot_model())
        old.pop('image_contract')
        self.web.batch_info['legacy-color'] = dict(model_contract=old)
        for route in ('image', 'batch'):
            response = self.client.post('/api/detect/'+route, data={
                'file' if route == 'image' else 'files': (png_bytes(self.blank), 'a.png'),
                'batch_id': 'legacy-color'}, content_type='multipart/form-data')
            self.assertEqual(response.status_code, 409)
        self.assertFalse(self.records('legacy-color'))

    def test_legacy_color_contract_cannot_resume_path_task(self):
        folder = self.base / 'input'
        folder.mkdir()
        (folder / 'a.png').write_bytes(png_bytes(self.blank).getvalue())
        old = self.web._model_contract(self.web.snapshot_model())
        old.pop('image_contract')
        jobs = [(str(folder / 'a.png'), folder.name+'/a.png')]
        prev = dict(root=str(folder), conf=.3, done=0, status='stopped', batch_id='old',
                    model_contract=old, jobs_fingerprint=self.web._jobs_fingerprint(jobs))
        with mock.patch.object(self.web, '_interrupted_task', return_value=prev), \
             mock.patch.object(self.web.threading, 'Thread') as worker:
            response = self.client.post('/api/detect/start', json={
                'path': str(folder), 'conf': .3, 'resume': True})
            self.assertEqual(response.status_code, 409)
            worker.assert_not_called()


if __name__ == '__main__':
    unittest.main(defaultTest='ImageContractTests', verbosity=2)
