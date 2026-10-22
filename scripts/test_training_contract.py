# -*- coding: utf-8 -*-
"""训练/物化门槛沙盘回归：合成图片、假训练器；不训练真实模型，不改生产数据。
用法：runtime/python.exe -B scripts/test_training_contract.py
"""
import os
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import sys
import csv
import io
import json
import copy
import hashlib
import logging
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'core'))
sys.path.insert(0, str(ROOT / 'scripts'))
from path_safety import dataset_output, run_output, safe_name, within
from dataset_contract import inspect_dataset, check_yolo
from materialize import build_dataset
from eval_cache import file_sha256
import training_contract as tc
import train_run as tr
import train as interactive


class Sandbox(unittest.TestCase):
    def setUp(self):
        cache = ROOT / '.cache'
        cache.mkdir(exist_ok=True)
        self.base = Path(tempfile.mkdtemp(prefix='训练沙盘_', dir=cache))
        self.addCleanup(self.cleanup)
        self.rows = []
        self.source = self.base / '母本'
        self.source.mkdir()
        for index, split in enumerate(['train', 'val', 'train']):
            positive = index != 2
            img = np.random.default_rng(index).integers(0, 256, (240, 320, 3), dtype=np.uint8)
            image_path = self.source / f'{index}.jpg'
            ok, encoded = cv2.imencode('.jpg', img)
            self.assertTrue(ok)
            encoded.tofile(str(image_path))
            label_path = self.source / f'{index}.json'
            if positive:
                label_path.write_text(json.dumps(dict(imageWidth=320, imageHeight=240,
                    shapes=[dict(label='1', points=[[130, 100], [150, 120]])])), encoding='utf8')
            self.rows.append(dict(name=f'样本{index}', src_path=str(image_path),
                label_json=str(label_path) if positive else '', split=split, group=f'G{index}',
                md5=hashlib.md5(image_path.read_bytes()).hexdigest(), n_boxes='1' if positive else '0',
                is_neg='0' if positive else '1', size='320x240'))
        self.manifest = self.base / 'manifest.csv'
        self.write_manifest()

    def cleanup(self):
        # 递归删除前先解析并核对绝对目标，仅清理本测试独占沙盘。
        target = self.base.resolve()
        if target.parent != (ROOT / '.cache').resolve() or not target.name.startswith('训练沙盘_'):
            raise RuntimeError('拒绝清理非沙盘路径')
        shutil.rmtree(target)

    def write_manifest(self):
        with self.manifest.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def build(self, name='derived', **kwargs):
        return build_dataset(self.manifest, self.base, name, **kwargs)

    def command(self, *args):
        return subprocess.run([sys.executable, '-B', '-X', 'utf8', *map(str, args)],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf8', timeout=30)


class DatasetTests(Sandbox):
    def test_full_materialization_is_hardlinked_and_frozen(self):
        originals = {p: p.read_bytes() for p in self.source.iterdir()}
        result = self.build()
        out = Path(result['output'])
        report = inspect_dataset(out / 'data.yaml', 1920)
        self.assertEqual(report['stats']['train'], dict(images=2, boxes=1, negative=1))
        source = Path(self.rows[0]['src_path'])
        linked = out / 'train/images/样本0.jpg'
        self.assertEqual(source.stat().st_ino, linked.stat().st_ino)
        self.assertGreaterEqual(source.stat().st_nlink, 2)
        self.assertTrue(all(p.read_bytes() == before for p, before in originals.items()))
        self.assertIsNone(report['contract']['sources'][2]['label_sha256'])

    def test_roi_transform_and_verify_only_do_not_touch_sources(self):
        before = {p: p.read_bytes() for p in self.source.iterdir()}
        result = self.build(name='readonly', roi=True, roi_k=1.5, verify_only=True)
        self.assertFalse(Path(result['output']).exists())
        result = self.build(name='roi', roi=True, roi_k=1.5)
        report = inspect_dataset(Path(result['output']) / 'data.yaml', 1536)
        self.assertEqual(report['contract']['roi_k'], 1.5)
        self.assertEqual(report['stats']['val']['boxes'], 1)
        self.assertTrue(all(p.read_bytes() == value for p, value in before.items()))

    def test_force_preserves_existing_output_and_protected_paths(self):
        out = self.base / 'old'
        out.mkdir()
        sentinel = out / '用户文件.txt'
        sentinel.write_text('keep', encoding='utf8')
        with self.assertRaises(FileExistsError):
            self.build(name='old', force=True)
        self.assertEqual(sentinel.read_text(encoding='utf8'), 'keep')
        for root in ['runtime', 'datasets', 'models/启用', '问题电芯图片汇总', ROOT.parent]:
            with self.subTest(root=root), self.assertRaises(ValueError):
                dataset_output(root, 'test')
        for name in ['../x', r'..\x', 'C:\\x', 'bad:stream', 'CON', 'nul.txt', 'a.', 'a ', '']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_name(name)
        with self.assertRaises(ValueError):
            run_output('../outside')

    def test_junction_escape_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory(prefix='ct_external_') as target:
            sentinel = Path(target) / 'keep.txt'
            sentinel.write_text('keep', encoding='utf8')
            link = self.base / 'junction'
            result = subprocess.run(['cmd.exe', '/c', 'mklink', '/J', str(link), target],
                                    capture_output=True)
            if result.returncode:
                self.skipTest('本机不允许创建沙盘junction')
            try:
                with self.assertRaises(ValueError):
                    dataset_output(link, 'escaped')
                self.assertEqual(sentinel.read_text(encoding='utf8'), 'keep')
            finally:
                os.rmdir(link)  # 只移除junction本身，不递归触及外部目标。

    def test_missing_positive_label_and_unconfirmed_negative_are_rejected(self):
        self.rows[0]['label_json'] = ''
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.build()
        self.assertFalse((self.base / 'derived/data_contract.json').exists())
        self.rows[0]['n_boxes'], self.rows[0]['is_neg'] = '0', '0'
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.build(name='unreviewed')

    def test_bad_json_class_and_coordinates_are_rejected(self):
        label = Path(self.rows[0]['label_json'])
        original = json.loads(label.read_text(encoding='utf8'))
        cases = [None, 'unknown', float('nan'), -1]
        for i, value in enumerate(cases):
            with self.subTest(value=value):
                data = copy.deepcopy(original)
                if value is None:
                    label.write_text('{bad', encoding='utf8')
                else:
                    if value == 'unknown':
                        data['shapes'][0]['label'] = '9'
                    else:
                        data['shapes'][0]['points'][0][0] = value
                    label.write_text(json.dumps(data), encoding='utf8')
                with self.assertRaises(ValueError):
                    self.build(name=f'bad{i}')
                self.assertFalse((self.base / f'bad{i}/data_contract.json').exists())

    def test_image_md5_dimension_and_missing_image_are_rejected(self):
        self.rows[0]['md5'] = '0'*32
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.build()
        self.rows[0]['md5'] = hashlib.md5(Path(self.rows[0]['src_path']).read_bytes()).hexdigest()
        self.rows[0]['size'] = '321x240'
        self.write_manifest()
        with self.assertRaises(ValueError):
            self.build(name='badsize')
        self.rows[0]['size'] = '320x240'
        self.rows[0]['src_path'] = str(self.source / 'missing.jpg')
        self.write_manifest()
        with self.assertRaises(OSError):
            self.build(name='missing')

    def test_cross_split_groups_hashes_and_case_collisions_are_rejected(self):
        for key, value in [('group', self.rows[0]['group']), ('md5', self.rows[0]['md5']),
                           ('name', self.rows[0]['name'])]:
            with self.subTest(key=key):
                before = self.rows[1][key]
                self.rows[1][key] = value
                self.write_manifest()
                with self.assertRaises(ValueError):
                    self.build(name='bad_'+key)
                self.rows[1][key] = before
        self.write_manifest()

    def test_hardlink_failure_does_not_copy_and_bad_encoding_never_completes(self):
        before = Path(self.rows[0]['src_path']).read_bytes()
        with mock.patch('materialize.os.link', side_effect=OSError('cross volume')):
            with self.assertRaises(OSError):
                self.build()
        self.assertFalse((self.base / 'derived/data.yaml').exists())
        self.assertEqual(Path(self.rows[0]['src_path']).read_bytes(), before)
        with mock.patch('materialize.cv2.imencode', return_value=(False, None)):
            with self.assertRaises(ValueError):
                self.build(name='encode', roi=True)
        self.assertFalse((self.base / 'encode/data_contract.json').exists())

    def test_invalid_roi_and_dropped_gt_never_complete(self):
        with mock.patch('materialize.R.roi_looks_valid', return_value=False):
            with self.assertRaises(ValueError):
                self.build(roi=True)
        label = Path(self.rows[0]['label_json'])
        data = json.loads(label.read_text(encoding='utf8'))
        data['shapes'][0]['points'] = [[5, 5], [15, 15]]
        label.write_text(json.dumps(data), encoding='utf8')
        with self.assertRaises(ValueError):
            self.build(name='dropped', roi=True)
        self.assertFalse((self.base / 'dropped/data.yaml').exists())

    def test_preflight_rejects_missing_label_drift_and_wrong_classes(self):
        out = Path(self.build()['output'])
        lp = out / 'train/labels/样本0.txt'
        original = lp.read_bytes()
        lp.unlink()
        with self.assertRaises(ValueError):
            inspect_dataset(out / 'data.yaml')
        lp.write_bytes(original + b'\n1 nan .5 .1 .1')
        with self.assertRaises(ValueError):
            inspect_dataset(out / 'data.yaml')
        lp.write_bytes(original)
        data = yaml.safe_load((out / 'data.yaml').read_text(encoding='utf8'))
        data['names'][0] = 'wrong'
        (out / 'data.yaml').write_text(yaml.safe_dump(data), encoding='utf8')
        with self.assertRaises(ValueError):
            inspect_dataset(out / 'data.yaml')

    def test_yolo_quality_gate(self):
        for text in ['9 .5 .5 .1 .1', '0 .5 .5 0 .1', '1 nan .5 .1 .1',
                     '1 .9 .5 .5 .1', '1 .5 .5 .1 .1\n1 .5 .5 .1 .1']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                check_yolo(text)
        self.assertEqual(check_yolo(''), [])

    def test_actual_materialize_and_kb_cli_use_same_gate(self):
        result = self.command('scripts/materialize_dataset.py', '--manifest', self.manifest,
                              '--root', self.base, '--name', 'cli')
        self.assertEqual(result.returncode, 0, result.stderr)
        data = self.base / 'cli/data.yaml'
        result = self.command('scripts/kb.py', 'preflight', '--data', data, '--imgsz', 1920)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        (data.parent / 'val/labels/样本1.txt').unlink()
        result = self.command('scripts/kb.py', 'preflight', '--data', data, '--imgsz', 1920)
        self.assertNotEqual(result.returncode, 0)


class TrainingTests(Sandbox):
    def setUp(self):
        super().setUp()
        self.data = Path(self.build()['output']) / 'data.yaml'
        self.model = self.base / 'fake.pt'
        self.model.write_bytes(b'fake-local-weights')
        self.baseline = self.base / 'args.yaml'
        self.baseline.write_text('{}', encoding='utf8')
        self.findings = self.base / 'FINDINGS.md'
        self.findings.write_text('', encoding='utf8')
        self.output = self.base / 'runs'
        self.path_patch = mock.patch.object(tr, 'run_output', side_effect=lambda name: self.output / safe_name(name))
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)
        check = lambda f, n, h: tc.check_preregistration(f, n, h, self.findings)
        self.reg_patch = mock.patch.object(tr, 'check_preregistration', side_effect=check)
        self.reg_patch.start()
        self.addCleanup(self.reg_patch.stop)
        self.options = ['--data', str(self.data), '--model', str(self.model), '--name', 'sandbox_run',
                        '--baseline-args', str(self.baseline), '--no-archive']
        with mock.patch.object(tr, 'compare_config', return_value={}):
            initial = tr.build_plan(tr.parser().parse_args(self.options))
        base = dict(initial['payload']['train_kwargs'], model=str(self.model))
        self.baseline.write_text(yaml.safe_dump(base), encoding='utf8')

    def plan(self, extra=(), registered=True):
        args = tr.parser().parse_args(self.options + list(extra) + ['--finding', 'F-999'])
        plan = tr.build_plan(args)
        if registered:
            self.findings.write_text('### F-999 沙盘假设\n- **状态**：待验\n'
                '- **关联 run**：`sandbox_run`\n- **证据**：plan_sha256='+plan['plan_sha256']+'\n', encoding='utf8')
            plan = tr.build_plan(args)
        return plan

    def factory(self, failure=None):
        testcase = self
        class FakeModel:
            def __init__(self, path):
                self.callbacks = {}
                meta = json.loads((testcase.output / 'sandbox_run/_train_meta.json').read_text(encoding='utf8'))
                testcase.assertEqual(meta['state'], 'starting')
                if failure == 'constructor':
                    raise RuntimeError('injected constructor')
            def add_callback(self, name, function):
                self.callbacks[name] = function
            def train(self, **kwargs):
                out = testcase.output / 'sandbox_run'
                self.trainer = SimpleNamespace(save_dir=out if failure != 'directory' else testcase.output / 'other',
                    args=SimpleNamespace(**kwargs), batch_size=3, accumulate=21, epoch=79,
                    best=out / 'weights/best.pt')
                self.callbacks['on_pretrain_routine_start'](self.trainer)
                testcase.assertEqual(json.loads((out/'_train_meta.json').read_text(encoding='utf8'))['state'], 'running')
                self.callbacks['on_train_start'](self.trainer)
                print('synthetic epoch log')
                if failure == 'training':
                    raise RuntimeError('injected training')
                self.trainer.best.parent.mkdir()
                self.trainer.best.write_bytes(b'sandbox-best')
        return FakeModel

    def execute(self, plan, failure=None):
        with mock.patch.object(tc, 'other_training_processes', return_value=[]):
            return tr.execute_plan(plan, model_factory=self.factory(failure))

    def test_default_plan_is_readonly_1920_and_both_entries_delegate(self):
        plan = self.plan(registered=False)
        self.assertFalse(plan['ready'])
        self.assertEqual(plan['payload']['train_kwargs']['imgsz'], 1920)
        self.assertFalse(self.output.exists())
        with mock.patch.object(interactive.train_run, 'main', return_value=0) as main:
            self.assertEqual(interactive.main(self.options + ['--dry-run']), 0)
        main.assert_called_once_with(self.options + ['--dry-run'])
        with mock.patch('builtins.input', return_value=''):
            self.assertEqual(interactive.ask_imgsz(), 1920)

    def test_two_changes_are_blocked_and_path_normalization_is_not_change(self):
        with self.assertRaises(ValueError):
            self.plan(['--mosaic', '0', '--degrees', '0'])
        baseline = yaml.safe_load(self.baseline.read_text(encoding='utf8'))
        baseline['model'] = baseline['model'].replace('/', '\\')
        self.baseline.write_text(yaml.safe_dump(baseline), encoding='utf8')
        self.assertEqual(self.plan()['differences'], {})

    def test_preregistration_must_bind_pending_run_and_exact_plan(self):
        plan = self.plan(registered=False)
        with self.assertRaises(ValueError):
            tr.execute_plan(plan, self.factory())
        registered = self.plan()
        text = self.findings.read_text(encoding='utf8')
        for changed in [text.replace('待验', '已证伪'), text.replace('sandbox_run', 'other'),
                        text.replace(registered['plan_sha256'], '0'*64)]:
            self.findings.write_text(changed, encoding='utf8')
            with self.assertRaises(ValueError):
                tr.verify_plan_inputs(registered)
        self.assertFalse(self.output.exists())

    def test_cutmix_is_forwarded_and_actual_batch_and_best_are_recorded(self):
        plan = self.plan(['--cutmix', '.2'])
        self.assertEqual(set(plan['differences']), {'cutmix'})
        stdout, stderr = sys.stdout, sys.stderr
        before = list(logging.getLogger().handlers)
        meta = self.execute(plan)
        self.assertEqual(meta['state'], 'completed')
        self.assertEqual(meta['hyperparams_effective']['cutmix'], .2)
        self.assertEqual((meta['actual_batch'], meta['accumulate']), (3, 21))
        self.assertEqual(meta['save_dir'], str((self.output/'sandbox_run').resolve()))
        self.assertEqual(meta['best_pt_sha256'], file_sha256(meta['best_pt']))
        from model_spec import load_model_spec
        spec = load_model_spec(meta['best_pt'])
        self.assertEqual((spec.imgsz, spec.roi, spec.roi_k), (1920, False, 0.0))
        self.assertIs(sys.stdout, stdout)
        self.assertIs(sys.stderr, stderr)
        self.assertEqual(logging.getLogger().handlers, before)
        self.assertFalse((self.output/'.ct_training.lock').exists())

    def test_constructor_and_training_failure_always_leave_meta_and_release_handles(self):
        for failure in ['constructor', 'training']:
            with self.subTest(failure=failure):
                plan = self.plan()
                before = list(logging.getLogger().handlers)
                stdout, stderr = sys.stdout, sys.stderr
                with self.assertRaises(RuntimeError):
                    self.execute(plan, failure)
                out = self.output / 'sandbox_run'
                meta = json.loads((out/'_train_meta.json').read_text(encoding='utf8'))
                self.assertEqual(meta['state'], 'failed')
                self.assertIn(failure, meta['error'])
                self.assertEqual(meta['hyperparams']['imgsz'], 1920)
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
                self.assertEqual(logging.getLogger().handlers, before)
                # Windows上日志必须已关闭，才能改名；沙盘产物保留，不删除。
                (out/'train.log').rename(out/'closed.log')
                out.rename(self.output / failure)

    def test_actual_directory_drift_is_failed_without_metadata_written_elsewhere(self):
        plan = self.plan()
        with self.assertRaises(RuntimeError):
            self.execute(plan, 'directory')
        meta = json.loads((self.output/'sandbox_run/_train_meta.json').read_text(encoding='utf8'))
        self.assertEqual(meta['state'], 'failed')
        self.assertFalse((self.output/'other/_train_meta.json').exists())

    def test_model_plan_and_label_drift_are_blocked_before_model_factory(self):
        plan = self.plan()
        changed = copy.deepcopy(plan)
        changed['payload']['train_kwargs']['lr0'] = .01
        with self.assertRaises(ValueError):
            tr.verify_plan_inputs(changed)
        self.model.write_bytes(b'changed weights')
        with self.assertRaises(ValueError):
            tr.verify_plan_inputs(plan)
        self.model.write_bytes(b'fake-local-weights')
        (self.data.parent/'val/labels/样本1.txt').write_text('', encoding='utf8')
        factory = mock.Mock()
        with self.assertRaises(subprocess.CalledProcessError):
            tr.execute_plan(plan, factory)
        factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_training_slot_blocks_existing_process_lock_and_releases_on_error(self):
        slot = self.base / 'slots'
        with mock.patch.object(tc, 'other_training_processes', return_value=[{'ProcessId': 123}]):
            with self.assertRaises(RuntimeError), tc.training_slot(slot):
                self.fail('不得进入训练')
        with mock.patch.object(tc, 'other_training_processes', return_value=[]):
            with self.assertRaises(RuntimeError):
                with tc.training_slot(slot):
                    with self.assertRaises(FileExistsError), tc.training_slot(slot):
                        self.fail('不得重入')
                    raise RuntimeError('injected')
            self.assertFalse((slot/'.ct_training.lock').exists())

    def test_roi_training_unknown_specs_and_unsafe_run_names_are_blocked(self):
        with self.assertRaises(ValueError):
            self.plan(['--roi'])
        for extra in [['--roi-k', '1.5'], ['--imgsz', '0'], ['--batch', 'nan'], ['--name', '../escape']]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                self.plan(extra)
        (self.data.parent/'data_contract.json').unlink()
        with self.assertRaises(OSError):
            self.plan()

    def test_inactive_detection_augmentation_cannot_launch_a_noop_experiment(self):
        for args in [['--erasing', '0'], ['--copy-paste', '.5']]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.plan(args)

    def test_archive_failure_is_distinct_from_training_failure(self):
        plan = self.plan()
        plan['no_archive'] = False
        original_run = subprocess.run
        def run(args, **kwargs):
            if any(str(a).endswith('archive_experiment.py') for a in args):
                raise subprocess.CalledProcessError(1, args)
            return original_run(args, **kwargs)
        with mock.patch.object(tr.subprocess, 'run', side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute(plan)
        meta = json.loads((self.output/'sandbox_run/_train_meta.json').read_text(encoding='utf8'))
        self.assertEqual(meta['state'], 'completed')
        self.assertEqual(meta['archive_status'], 'failed')
        self.assertTrue(Path(meta['best_pt']).is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)
