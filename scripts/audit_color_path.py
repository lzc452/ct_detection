# -*- coding: utf-8 -*-
"""阶段5补充：隔离通道交换影响，测真实路径worker的无检出分支。
用法：--freeze --out reports/system_audit/stage5_followup_plan.json；
然后两个独立进程 --plan <JSON> --half 0|1 --out <新JSON>。
只读取冻结清单；路径测试装系统审计钩子禁止任何原文件复制；不碰现有服务。
"""
import os
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
import sys
import io
import json
import time
import argparse
import tempfile
from pathlib import Path

from audit_precision_e2e import (ROOT, WEIGHTS, rows_and_contract, file_sha256,
    dump_new, load_web, prediction, stats, percentile, paired_predictions, bootstrap)

BASE_PLAN = ROOT / 'reports/system_audit/stage5_plan_v2.json'
FP32 = ROOT / 'reports/system_audit/stage5_fp32.json'
FP16 = ROOT / 'reports/system_audit/stage5_fp16.json'


def freeze(out):
    rows, _, fp = rows_and_contract()
    a, b = [json.loads(p.read_text(encoding='utf8')) for p in (FP32, FP16)]
    chosen = []
    for i, row in enumerate(rows):
        if (row['is_neg'] == '1' and row['n_boxes'] == '0' and
            all(not x['quality'][batch]['predictions'][i] for x in (a, b) for batch in ('1', '16'))):
            chosen.append(row['name'])
        if len(chosen) == 16:
            break
    if len(chosen) != 16:
        raise ValueError('没有足够无检出样本，不代替实际写盘函数')
    plan = dict(schema='ct-stage5-followup/1', created=time.strftime('%Y-%m-%d %H:%M:%S%z'),
        base_plan=str(BASE_PLAN), base_plan_sha256=file_sha256(BASE_PLAN), input_fingerprint=fp,
        source_sha256=file_sha256(__file__), baseline_sha256={str(p): file_sha256(p) for p in (FP32, FP16)},
        color=dict(n=676, half=False, batch=1, contract='cv2 BGR与同一数组交换通道的RGB；其它参数固定',
                   decoder='全676图同时比较PIL转BGR与cv2 BGR像素，若不同须披露解码混杂'),
        path=dict(names=chosen, batch=16, repeats=6, worker='web._detect_worker原函数',
            contract='清单预建后从路径读图到处理完成，包含原读图/推理/台账/状态JSON；无检出分支',
            selection='两精度两batch均无框的声明负例前16；条件性诊断，不代表全生产',
            precheck='实际重新组成batch16后如出现框直接失败；sys.audit禁止shutil.copyfile原文件复制',
            excludes='目录发现/问题图备份分支/外盘/并发队列/浏览器'),
        decision='报告颜色影响大小；不宣称修正颜色即改善泛化。路径仅给无检出分支，不据此宣称缺陷混合产线P95',
        closure='算力，作为F-034补充，不改变其判据')
    dump_new(out, plan)
    print(file_sha256(out))


def measure(plan_path, half, out):
    import cv2
    import numpy as np
    import torch
    from image_io import read_bgr_disk, read_bgr_upload
    from training_contract import other_training_processes
    plan = json.loads(Path(plan_path).read_text(encoding='utf8'))
    base = json.loads(BASE_PLAN.read_text(encoding='utf8'))
    rows, _, fp = rows_and_contract()
    if (file_sha256(BASE_PLAN) != plan['base_plan_sha256'] or fp != plan['input_fingerprint'] or
        file_sha256(__file__) != plan['source_sha256'] or
        any(file_sha256(p) != sha for p, sha in plan['baseline_sha256'].items()) or
        any(file_sha256(ROOT / p) != sha for p, sha in base['sources'].items()) or
        file_sha256(WEIGHTS) != base['weight_sha256']):
        raise ValueError('冻结输入或代码变化')
    if not torch.cuda.is_available() or other_training_processes():
        raise ValueError('GPU不可用或已有训练')
    forbidden = {str((ROOT / row['src_path']).resolve()).casefold() for row in rows}
    copies = []
    def guard(event, args):
        if event == 'shutil.copyfile' and str(Path(args[0]).resolve()).casefold() in forbidden:
            copies.append(str(args[0]))
            raise PermissionError('核验沙盘禁止复制原始生产图')
    sys.addaudithook(guard)
    result = dict(schema=plan['schema'], half=bool(half), plan_sha256=file_sha256(plan_path),
                  start=time.strftime('%Y-%m-%d %H:%M:%S%z'))
    with tempfile.TemporaryDirectory(prefix='ct_path_stage5_', dir=ROOT / '.cache') as tmp:
        sandbox = Path(tmp).resolve()
        if not sandbox.is_relative_to((ROOT / '.cache').resolve()):
            raise ValueError('沙盘越界')
        web = load_web(half, sandbox)
        bundle = web.snapshot_model()
        dtype = str(next(bundle.model.predictor.model.model.parameters()).dtype)
        if dtype != ('torch.float16' if half else 'torch.float32'):
            raise ValueError('实际dtype不符合要求')
        result['actual_dtype'] = dtype
        if not half:
            predictions, decoder = [], dict(n_nonidentical=0, max_abs_delta=0)
            for row in rows:
                bgr = read_bgr_disk(ROOT / row['src_path'])
                upload_bgr = read_bgr_upload(io.BytesIO((ROOT / row['src_path']).read_bytes()))
                delta = int(np.abs(bgr.astype(np.int16)-upload_bgr.astype(np.int16)).max())
                decoder['n_nonidentical'] += int(delta > 0)
                decoder['max_abs_delta'] = max(decoder['max_abs_delta'], delta)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                predictions.append(prediction(web._predict_inputs(bundle, [rgb], .30)[0]))
            baseline = json.loads(FP32.read_text(encoding='utf8'))['quality']['1']
            metrics = stats(rows, predictions)
            result['color'] = dict(rgb_predictions=predictions, rgb_metrics=metrics,
                bgr_metrics=baseline['metrics'], decoder_comparison=decoder,
                pairing=paired_predictions(baseline['predictions'], predictions),
                bootstrap_rgb_minus_bgr=bootstrap(baseline['metrics'], metrics))
        chosen = [next(r for r in rows if r['name'] == n) for n in plan['path']['names']]
        images = [read_bgr_disk(ROOT / r['src_path']) for r in chosen]
        if any(len(r.boxes) for r in web._predict_inputs(bundle, images, .30)):
            raise ValueError('重组后的batch16出现框，不能按无检出分支测速')
        jobs = [(str(ROOT / r['src_path']), r['name']+'.jpg') for r in chosen]
        wall, tasks = [], []
        for i in range(plan['path']['repeats']):
            key = 'clean-path-'+str(i)
            web.detect_tasks[key] = dict(status='running', done=0, total=16, defect_imgs=0,
                defects=0, errors=0, speed=0., eta_s=-1, current='', message='', items=[], stop=False, batch_id=key)
            torch.cuda.synchronize()
            start = time.perf_counter()
            web._detect_worker(key, str(ROOT), jobs, .30, 0, key, bundle)
            torch.cuda.synchronize()
            wall.append((time.perf_counter()-start)*1000)
            task = web.detect_tasks[key]
            tasks.append(dict(status=task['status'], done=task['done'], errors=task['errors'],
                              defect_imgs=task['defect_imgs']))
            if task['errors'] or task['done'] != 16 or task['defect_imgs'] or copies:
                raise ValueError('路径测量未走完整无检出分支')
        result['path'] = dict(timing_ms=percentile(wall), tasks=tasks,
            attempted_raw_copies=copies, ledger_records=len(web.file_records),
            sandbox_files=sorted(p.name for p in Path(web.RESULTS_FOLDER).iterdir()))
        result['end'] = time.strftime('%Y-%m-%d %H:%M:%S%z')
        if rows_and_contract()[2] != fp:
            raise ValueError('测量期间输入变化')
        dump_new(out, result)
        print(json.dumps(dict(out=out, dtype=dtype), ensure_ascii=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--plan')
    ap.add_argument('--half', type=int, choices=(0, 1))
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.freeze:
        freeze(args.out)
    elif args.plan and args.half is not None:
        measure(args.plan, args.half, args.out)
    else:
        ap.error('需--freeze或--plan和--half')
