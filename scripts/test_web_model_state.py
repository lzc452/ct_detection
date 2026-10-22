# -*- coding: utf-8 -*-
"""Web模型快照/并发/无效ROI故障注入回归，仅用沙盘假权重和合成图片。
用法：runtime/python.exe -B scripts/test_web_model_state.py
不访问生产图片、不启用新权重、不占GPU；HTTP自测使用独立随机端口。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
import sys
import io
import json
import hashlib
import importlib.util
import tempfile
import threading
import unittest
import urllib.request
from dataclasses import replace, FrozenInstanceError
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'core'))
from test_web_ledger import FakeTensor, FakeYOLO, install_fake_ultralytics


class Boxes:
    def __init__(self, positive):
        self.data = np.array([[10., 10., 20., 20., .9, 1.]]) if positive else np.zeros((0, 6))

    def __len__(self):
        return len(self.data)

    @property
    def xyxy(self):
        return FakeTensor(self.data[:, :4])

    @property
    def cls(self):
        return FakeTensor(self.data[:, 5])

    @property
    def conf(self):
        return FakeTensor(self.data[:, 4])


class Result:
    def __init__(self, img):
        self.img = img
        self.boxes = Boxes(float(img.mean()) > 10)

    def plot(self):
        return self.img.copy()


class Model(FakeYOLO):
    calls = []

    def __call__(self, source, **kw):
        inputs = source if isinstance(source, list) else [source]
        self.calls.append(dict(path=self.path, shapes=[im.shape for im in inputs], **kw))
        return [Result(im) for im in inputs]


def png_bytes(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format='PNG')
    buf.seek(0)
    return buf


class ModelStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ct_模型沙盘_')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        models = self.base / 'models'
        for folder, k in [('legacy', None), ('roi0', 0.), ('roi15', 1.5)]:
            dst = models / folder
            dst.mkdir(parents=True)
            (dst / 'best.pt').write_bytes(folder.encode())
            (dst / '_train_meta.json').write_text(json.dumps({
                'hyperparams': {'imgsz': 1920 if k is None else 1536}, 'roi': k is not None,
                'roi_k': k or 0.0, 'best_pt_sha256': hashlib.sha256(folder.encode()).hexdigest()}), encoding='utf8')
        self.env = mock.patch.dict(os.environ, {
            'CT_MODEL_DIR': str(models), 'CT_RESULTS_DIR': str(self.base / 'results'),
            'CT_NO_BROWSER': '1'})
        self.env.start()
        self.addCleanup(self.env.stop)
        install_fake_ultralytics()
        sys.modules['ultralytics'].YOLO = Model
        Model.calls.clear()
        name = 'web_state_sandbox'
        spec = importlib.util.spec_from_file_location(name, ROOT / 'web/app.py')
        self.web = importlib.util.module_from_spec(spec)
        sys.modules[name] = self.web
        self.addCleanup(sys.modules.pop, name, None)
        with mock.patch('device_utils.pick_device', return_value='cpu'):
            spec.loader.exec_module(self.web)
        self.ids = {Path(v['path']).parent.name: key for key, v in self.web.MODEL_REGISTRY.items()}
        self.web.load_model(self.ids['legacy'], warmup=False)
        Model.calls.clear()
        self.client = self.web.app.test_client()
        self.img = np.random.default_rng(23).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        self.blank = np.zeros_like(self.img)

    def roi_bundle(self, key='roi0'):
        return self.web.load_model(self.ids[key], warmup=False)

    def records(self, batch):
        return [r for r in self.web.file_records if r['批次ID'] == batch]

    def task(self, key, n, batch):
        self.web.detect_tasks[key] = dict(status='running', done=0, total=n,
            defect_imgs=0, defects=0, errors=0, speed=0., eta_s=-1,
            current='', message='', items=[], stop=False, batch_id=batch)

    def test_frozen_snapshot_survives_switch_and_registry_mutation(self):
        old = self.roi_bundle()
        with self.assertRaises(FrozenInstanceError):
            old.spec.roi_k = 1.5
        self.web.MODEL_REGISTRY[old.model_id]['spec'] = {'imgsz': 640, 'roi_k': 99}
        self.roi_bundle('roi15')
        res, draw = self.web.infer_one(old, self.img, .3)
        box = self.web.roi_mod.roi_box(320, 240)
        self.assertEqual(Model.calls[-1]['imgsz'], 1536)
        self.assertEqual(Model.calls[-1]['path'], old.model.path)
        self.assertEqual(Model.calls[-1]['shapes'], [(box[3]-box[1], box[2]-box[0], 3)])
        np.testing.assert_array_equal(res.boxes.data[0, :4],
                                      [10+box[0], 10+box[1], 20+box[0], 20+box[1]])
        self.assertEqual(draw.shape, self.img.shape)

    def test_upload_routes_keep_snapshot_when_switch_occurs_during_decode(self):
        for route in ('image', 'batch'):
            with self.subTest(route=route):
                old = self.web.load_model(self.ids['legacy'], warmup=False)
                original_open = Image.open
                def switch_then_open(*args, **kwargs):
                    self.roi_bundle()
                    return original_open(*args, **kwargs)
                data = {'batch_id': route}
                data['file' if route == 'image' else 'files'] = (png_bytes(self.blank), 'blank.png')
                with mock.patch.object(self.web.Image, 'open', side_effect=switch_then_open):
                    response = self.client.post('/api/detect/'+route, data=data,
                                                content_type='multipart/form-data')
                self.assertEqual(response.status_code, 200, response.get_json())
                payload = response.get_json()
                status = payload['status'] if route == 'image' else payload['results'][0]['status']
                self.assertEqual(status, 'ok_clean')  # 全图旧快照，不可误走新ROI判失败
                record = self.records(route)[0]
                self.assertEqual(record['模型'], old.model_id)
                self.assertEqual(record['模型SHA256'], old.model_sha256)
                self.assertEqual(record['推理规格']['imgsz'], 1920)
                self.assertEqual(Model.calls[-1]['path'], old.model.path)

    def test_upload_chunks_reject_model_change_in_same_batch(self):
        for route in ('image', 'batch'):
            with self.subTest(route=route):
                self.web.load_model(self.ids['legacy'], warmup=False)
                def post():
                    data = {'batch_id': 'chunks_'+route}
                    data['file' if route == 'image' else 'files'] = (png_bytes(self.blank), 'a.png')
                    return self.client.post('/api/detect/'+route, data=data,
                                            content_type='multipart/form-data')
                self.assertEqual(post().status_code, 200)
                self.roi_bundle()
                Model.calls.clear()
                self.assertEqual(post().status_code, 409)
                self.assertEqual(Model.calls, [])
                self.assertEqual(len(self.records('chunks_'+route)), 1)

    def test_resume_checks_model_threshold_and_file_manifest(self):
        folder = self.base / 'input'
        folder.mkdir()
        for name in ('a.png', 'b.png'):
            Image.fromarray(self.img).save(folder / name)
        with mock.patch.object(self.web.threading, 'Thread') as worker:
            first = self.client.post('/api/detect/start', json={'path': str(folder), 'conf': .3}).get_json()
            self.assertTrue(first['success'], first)
            task = self.web.detect_tasks[first['task_id']]
            task['status'] = 'stopped'
            prev = dict(root=str(folder), conf=.3, done=1, status='stopped',
                        batch_id=first['batch_id'], model_contract=self.web._model_contract(self.web.snapshot_model()),
                        jobs_fingerprint=task['jobs_fingerprint'])
            with mock.patch.object(self.web, '_interrupted_task', return_value=prev):
                response = self.client.post('/api/detect/start', json={
                    'path': str(folder), 'conf': .3, 'resume': True})
                resumed = response.get_json()
                self.assertTrue(resumed['success'], resumed)
                self.assertEqual(resumed['skipped'], 1)
                args = worker.call_args.kwargs['args']
                self.assertEqual(len(args[2]), 1)
                self.assertEqual(args[-1].model_sha256, prev['model_contract']['model_sha256'])
                self.web.detect_tasks[resumed['task_id']]['status'] = 'stopped'
                for delta in ({'conf': .4}, {'model': 'roi0'}, {'files': True}):
                    with self.subTest(delta=delta):
                        self.web.load_model(self.ids['legacy'], warmup=False)
                        if 'model' in delta:
                            self.roi_bundle()
                        if 'files' in delta:
                            (folder / 'a.png').write_bytes(b'changed')
                        response = self.client.post('/api/detect/start', json={
                            'path': str(folder), 'conf': delta.get('conf', .3), 'resume': True})
                        self.assertEqual(response.status_code, 409, response.get_json())
                old = dict(prev)
                old.pop('model_contract')
                with mock.patch.object(self.web, '_interrupted_task', return_value=old):
                    self.assertEqual(self.client.post('/api/detect/start', json={
                        'path': str(folder), 'resume': True}).status_code, 409)

    def test_concurrent_scan_submission_creates_only_one_worker(self):
        folder = self.base / 'input'
        folder.mkdir()
        Image.fromarray(self.img).save(folder / 'a.png')
        entered, release, started2 = [threading.Event() for _ in range(3)]
        original_scan = self.web.scan_structure
        original_thread = threading.Thread
        responses, errors = [], []
        def scan(root):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('submission release missing')
            return original_scan(root)
        def submit(second=False):
            try:
                if second:
                    started2.set()
                with self.web.app.test_client() as client:
                    responses.append(client.post('/api/detect/start', json={
                        'path': str(folder)}).get_json())
            except Exception as exc:
                errors.append(exc)
        threads = [original_thread(target=submit), original_thread(target=submit, args=(True,))]
        with mock.patch.object(self.web, 'scan_structure', side_effect=scan) as scanning, \
             mock.patch.object(self.web.threading, 'Thread') as worker:
            try:
                threads[0].start()
                self.assertTrue(entered.wait(3))
                threads[1].start()
                self.assertTrue(started2.wait(3))
                self.assertEqual(scanning.call_count, 1)
            finally:
                release.set()
                for thread in threads:
                    if thread.ident:
                        thread.join(5)
            self.assertEqual(worker.call_count, 1)
        self.assertFalse(errors, errors)
        self.assertEqual(sorted(r['success'] for r in responses), [False, True])

    def test_failed_warmup_keeps_old_model_available(self):
        old = self.web.snapshot_model()
        class BadWarmup(Model):
            def __call__(self, *args, **kw):
                raise RuntimeError('injected warmup failure')
        with mock.patch.object(self.web, 'YOLO', BadWarmup):
            response = self.client.post('/api/switch_model', json={'model_id': self.ids['roi0']})
        self.assertEqual(response.status_code, 500)
        self.assertIs(self.web.snapshot_model(), old)
        result = self.web.infer_one(old, self.blank, .3)[0]
        self.assertEqual(len(result.boxes), 0)

    def test_bad_metadata_never_falls_back_to_clean_default(self):
        path = Path(self.web.MODEL_REGISTRY[self.ids['roi0']]['path']).with_name('_train_meta.json')
        old = self.web.snapshot_model()
        cases = ['{bad', '{}', json.dumps({'roi': True, 'hyperparams': {'imgsz': 1536}}),
                 json.dumps({'roi': 'false', 'hyperparams': {'imgsz': 1920}}),
                 json.dumps({'roi': True, 'roi_k': float('nan'), 'hyperparams': {'imgsz': 1536}}),
                 json.dumps({'roi': False, 'hyperparams': {'imgsz': 1537}})]
        for text in cases:
            with self.subTest(meta=text):
                path.write_text(text, encoding='utf8')
                with self.assertRaises((ValueError, TypeError)):
                    self.web.load_model(self.ids['roi0'])
                self.assertIs(self.web.snapshot_model(), old)

    def test_forward_and_switch_warmup_are_serialized(self):
        entered, release, started2, switching, constructed = [threading.Event() for _ in range(5)]
        guard = threading.Lock()
        active = [0, 0]
        errors = []
        class BlockingModel(Model):
            def __init__(self, *args, **kw):
                super().__init__(*args, **kw)
                constructed.set()
            def __call__(self, source, **kw):
                with guard:
                    active[0] += 1
                    active[1] = max(active)
                try:
                    if threading.current_thread().name == 'first':
                        entered.set()
                        if not release.wait(5):
                            raise TimeoutError('test did not release')
                    return super().__call__(source, **kw)
                finally:
                    with guard:
                        active[0] -= 1
        old = replace(self.web.snapshot_model(), model=BlockingModel('old'))
        constructed.clear()
        def first():
            self.web._predict_inputs(old, [self.img], .3)
        def second():
            started2.set()
            self.web._predict_inputs(old, [self.img], .3)
        def switch():
            switching.set()
            self.web.load_model(self.ids['roi0'])
        def checked(fn):
            try:
                fn()
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=checked, args=(fn,), name=name)
                   for fn, name in [(first, 'first'), (second, 'second'), (switch, 'switch')]]
        with mock.patch.object(self.web, 'YOLO', BlockingModel):
            try:
                threads[0].start()
                self.assertTrue(entered.wait(3))
                threads[1].start()
                threads[2].start()
                self.assertTrue(started2.wait(3) and switching.wait(3))
                self.assertFalse(constructed.wait(.1), '推理结束前不应构造/预热新模型')
            finally:
                release.set()
                for thread in threads:
                    if thread.ident:
                        thread.join(5)
        self.assertFalse(errors, errors)
        self.assertTrue(all(not t.is_alive() for t in threads))
        self.assertEqual(active[1], 1)
        self.assertEqual(self.web.snapshot_model().model_id, self.ids['roi0'])

    def test_two_roi_margins_cannot_interleave_set_and_crop(self):
        bundles = [self.roi_bundle(), self.roi_bundle('roi15')]
        entered, release, started2, second_crop = [threading.Event() for _ in range(4)]
        original_crop = self.web.roi_mod.crop_roi
        outputs, errors = {}, []
        def crop(img):
            if threading.current_thread().name == 'k0':
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('crop release missing')
            else:
                second_crop.set()
            return original_crop(img)
        def run(index):
            try:
                if index:
                    started2.set()
                outputs[index] = self.web._prep_infer(self.img, bundles[index])[1]
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=run, args=(i,), name='k0' if i == 0 else 'k15')
                   for i in range(2)]
        with mock.patch.object(self.web.roi_mod, 'crop_roi', side_effect=crop):
            try:
                threads[0].start()
                self.assertTrue(entered.wait(3))
                threads[1].start()
                self.assertTrue(started2.wait(3))
                self.assertFalse(second_crop.wait(.1))
            finally:
                release.set()
                for thread in threads:
                    if thread.ident:
                        thread.join(5)
        self.assertFalse(errors, errors)
        self.assertEqual(outputs[0], self.web.roi_mod.roi_box(320, 240))
        self.assertEqual(outputs[1], self.web.roi_mod.roi_box_padded(320, 240, 1.5))

    def test_invalid_roi_single_is_failed_with_no_forward(self):
        bundle = self.roi_bundle()
        Model.calls.clear()
        response = self.client.post('/api/detect/image', data={'batch_id': 'bad_roi',
            'file': (png_bytes(self.blank), '空白.png')}, content_type='multipart/form-data')
        self.assertEqual(response.get_json()['status'], 'failed')
        records = self.records('bad_roi')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['状态'], 'failed')
        self.assertEqual(records[0]['模型'], bundle.model_id)
        self.assertIn('ROI', records[0]['备注'])
        self.assertEqual(Model.calls, [])

    def test_invalid_roi_batch_does_not_poison_neighbors(self):
        self.roi_bundle()
        Model.calls.clear()
        response = self.client.post('/api/detect/batch', data={'batch_id': 'mixed', 'files': [
            (png_bytes(self.img), 'a.png'), (png_bytes(self.blank), '空白.png'),
            (png_bytes(self.img), 'c.png')]}, content_type='multipart/form-data').get_json()
        self.assertEqual([r['status'] for r in response['results']], ['ok_defect', 'failed', 'ok_defect'])
        self.assertEqual([r['index'] for r in response['results']], [0, 1, 2])
        self.assertEqual((response['ok_defect'], response['ok_clean'], response['failed']), (2, 0, 1))
        self.assertEqual(len(Model.calls), 1)
        self.assertEqual(len(Model.calls[0]['shapes']), 2)
        self.assertEqual(len(self.records('mixed')), 3)

    def test_invalid_roi_worker_retains_task_snapshot_and_ledger(self):
        old = self.roi_bundle()
        self.web.load_model(self.ids['legacy'], warmup=False)
        jobs = []
        for name, img in [('a.png', self.img), ('空白.png', self.blank), ('c.png', self.img)]:
            path = self.base / name
            Image.fromarray(img).save(path)
            jobs.append((str(path), name))
        self.task('t', 3, 'worker')
        self.web._detect_worker('t', str(self.base), jobs, .3, 0, 'worker', old)
        task = self.web.detect_tasks['t']
        self.assertEqual((task['done'], task['errors'], task['defect_imgs']), (3, 1, 2))
        self.assertEqual(task['model_id'], old.model_id)
        self.assertEqual(sorted(r['状态'] for r in self.records('worker')), ['failed', 'ok_defect', 'ok_defect'])
        self.assertTrue(all(r['模型'] == old.model_id for r in self.records('worker')))
        progress = json.loads(Path(self.web.TASK_FILE).read_text(encoding='utf8'))
        self.assertEqual(progress['model_contract'], self.web._model_contract(old))
        self.assertEqual(progress['jobs_fingerprint'], self.web._jobs_fingerprint(jobs))

    def test_invalid_roi_video_counts_failed_frames(self):
        bundle = self.roi_bundle()
        Model.calls.clear()
        cap = mock.Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {self.web.cv2.CAP_PROP_FPS: 25,
            self.web.cv2.CAP_PROP_FRAME_WIDTH: 320, self.web.cv2.CAP_PROP_FRAME_HEIGHT: 240,
            self.web.cv2.CAP_PROP_FRAME_COUNT: 2}[prop]
        cap.read.side_effect = [(True, self.blank.copy()), (True, self.blank.copy()), (False, None)]
        writer = mock.Mock()
        writer.isOpened.return_value = True
        self.web.video_tasks['v'] = {}
        path = self.base / 'synthetic.mp4'
        path.write_bytes(b'fake')
        with mock.patch.object(self.web.cv2, 'VideoCapture', return_value=cap), \
             mock.patch.object(self.web.cv2, 'VideoWriter', return_value=writer):
            self.web.process_video_task('v', str(path), .3, 'synthetic.mp4', bundle)
        task = self.web.video_tasks['v']
        self.assertEqual(task['status'], 'done_with_errors')
        self.assertEqual((task['frames'], task['frames_failed'], task['frames_defect']), (2, 2, 0))
        self.assertEqual(Model.calls, [])
        self.assertEqual(writer.write.call_count, 2)

    def test_result_count_mismatch_is_failed_for_upload_and_worker(self):
        old = self.web.snapshot_model()
        short = replace(old, model=mock.Mock(return_value=[]))
        self.web.current_bundle = short
        one = self.client.post('/api/detect/image', data={'batch_id': 'count_one',
            'file': (png_bytes(self.img), 'a.png')}, content_type='multipart/form-data').get_json()
        self.assertEqual(one['status'], 'failed')
        batch = self.client.post('/api/detect/batch', data={'batch_id': 'count_batch',
            'files': [(png_bytes(self.img), 'a.png')]}, content_type='multipart/form-data').get_json()
        self.assertEqual(batch['failed'], 1)
        self.task('t', 1, 'count_worker')
        path = self.base / 'source.png'
        Image.fromarray(self.img).save(path)
        self.web._detect_worker('t', str(self.base), [(str(path), 'a.png')], .3, 0, 'count_worker', short)
        self.assertEqual(self.web.detect_tasks['t']['errors'], 1)
        self.assertEqual(self.records('count_worker')[0]['状态'], 'failed')

    def test_state_file_replacement_failure_preserves_previous_json(self):
        self.web.save_state()
        before = Path(self.web.STATE_FILE).read_bytes()
        with mock.patch.object(self.web.os, 'replace', side_effect=PermissionError('injected')):
            self.web.save_state()
        self.assertEqual(Path(self.web.STATE_FILE).read_bytes(), before)
        self.assertIsInstance(json.loads(before), dict)
        self.assertFalse(list(Path(self.web.RESULTS_FOLDER).glob('*.tmp')))

    def test_partial_concurrent_state_write_is_not_visible_to_readers(self):
        self.web.save_state()
        before = Path(self.web.STATE_FILE).read_bytes()
        entered, release, started2, wrote2 = [threading.Event() for _ in range(4)]
        original_dump = json.dump
        errors = []
        def dump(obj, file, **kwargs):
            if threading.current_thread().name == 'writer1':
                file.write('{partial')
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('state release missing')
                file.seek(0)
                file.truncate()
            else:
                wrote2.set()
            return original_dump(obj, file, **kwargs)
        def write(second=False):
            try:
                if second:
                    started2.set()
                self.web.save_state()
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=write, name='writer1'),
                   threading.Thread(target=write, args=(True,), name='writer2')]
        with mock.patch.object(self.web.json, 'dump', side_effect=dump):
            try:
                threads[0].start()
                self.assertTrue(entered.wait(3))
                threads[1].start()
                self.assertTrue(started2.wait(3))
                self.assertFalse(wrote2.wait(.1))
                self.assertEqual(Path(self.web.STATE_FILE).read_bytes(), before)
            finally:
                release.set()
                for thread in threads:
                    if thread.ident:
                        thread.join(5)
        self.assertFalse(errors, errors)
        self.assertTrue(wrote2.is_set())
        self.assertIsInstance(json.loads(Path(self.web.STATE_FILE).read_bytes()), dict)

    def test_coordinate_restore_failure_is_not_suppressed(self):
        bundle = self.roi_bundle()
        result = Result(self.img)
        del result.boxes.data
        # 让len(boxes)有效，但实际data不可写，模拟不兼容的结果结构。
        class BadBoxes:
            def __len__(self):
                return 1
        result.boxes = BadBoxes()
        with self.assertRaises(AttributeError):
            self.web._restore_boxes(result, (10, 20, 30, 40))
        with mock.patch.object(self.web, '_restore_boxes', side_effect=ValueError('injected mapping failure')):
            response = self.client.post('/api/detect/image', data={'batch_id': 'mapping',
                'file': (png_bytes(self.img), 'a.png')}, content_type='multipart/form-data')
        self.assertEqual(response.get_json()['status'], 'failed')
        self.assertEqual(len(self.records('mapping')), 1)
        self.assertEqual(self.records('mapping')[0]['模型'], bundle.model_id)

    def test_isolated_http_server(self):
        from werkzeug.serving import make_server
        server = make_server('127.0.0.1', 0, self.web.app)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = 'http://127.0.0.1:%d' % server.server_port
            with urllib.request.urlopen(base+'/', timeout=5) as response:
                self.assertEqual(response.status, 200)
            with urllib.request.urlopen(base+'/api/models', timeout=5) as response:
                payload = json.load(response)
            self.assertEqual(payload['current'], self.web.snapshot_model().model_id)
            self.assertEqual(sum(m['active'] for m in payload['models']), 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
