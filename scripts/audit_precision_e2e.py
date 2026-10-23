# -*- coding: utf-8 -*-
"""阶段5真实GPU数值/时延/有界扰动核验；不训练、不发布、不重启5000。
用法：先 --mode freeze --plan reports/system_audit/stage5_plan.json；
再在独立进程分别 --mode measure --half 0|1 --plan ... --out <新JSON>；
最后 --mode compare --fp32 <JSON> --fp16 <JSON> --out <新JSON>。
只按清单精确访问，图片仅进内存，HTTP沙盘保存派生图；禁止拿此同集回归证明泛化。
"""
import os
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
import sys
import io
import csv
import json
import time
import hashlib
import argparse
import importlib.util
import tempfile
import threading
import urllib.request
from pathlib import Path
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / 'core'))
from eval_cache import build_contract, file_sha256
from model_spec import load_model_spec
from detection_metrics import match_predictions, iou_xyxy, MATCHING_VERSION

MANIFEST = ROOT / 'reports/splits/v5_split_manifest.csv'
WEIGHTS = ROOT / 'models/启用/20261019_V4/best.pt'
SOURCES = ['scripts/audit_precision_e2e.py', 'web/app.py', 'core/image_io.py',
           'core/model_spec.py', 'core/detection_metrics.py', 'core/eval_cache.py']


def dump_new(path, data):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT / 'reports'):
        raise ValueError('输出只允许reports下的新文件')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf8')).hexdigest()


def rows_and_contract():
    with MANIFEST.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == 'test']
    if len(rows) != 676 or len({r['name'] for r in rows}) != len(rows):
        raise ValueError('冻结test清单不再是676个唯一样本')
    for row in rows:
        if not row['label_json'] and not (row['is_neg'] == '1' and row['n_boxes'] == '0'):
            raise ValueError('无标注样本未显式声明负例')
    contract, fp = build_contract(rows, False)
    return rows, contract, fp


def selection(rows):
    chosen = []
    for cls in ('damage', 'dark', 'light'):
        for high in (False, True):
            row = next(r for r in rows if int(r['n_'+cls]) > 0 and
                       (int(r['size'].split('x')[0]) > 1500) == high)
            if row not in chosen:
                chosen.append(row)
    for row in rows:
        if int(row['n_damage']) and row not in chosen and len(chosen) < 8:
            chosen.append(row)
    return chosen + [r for r in rows if r['is_neg'] == '1'][:16-len(chosen)]


def freeze(path):
    rows, contract, fp = rows_and_contract()
    spec = load_model_spec(WEIGHTS)
    if spec.roi:
        raise ValueError('本次预注册仅限既有V4全图')
    import ultralytics
    plan = dict(schema='ct-stage5/1', created=time.strftime('%Y-%m-%d %H:%M:%S%z'),
        weights=str(WEIGHTS), weight_sha256=file_sha256(WEIGHTS), spec=asdict(spec),
        manifest_sha256=file_sha256(MANIFEST), input_fingerprint=fp, input_contract=contract,
        sources={p: file_sha256(ROOT / p) for p in SOURCES},
        ultralytics=ultralytics.__version__, selected=[r['name'] for r in selection(rows)],
        predict=dict(conf=.30, iou=.7, max_det=300, imgsz=spec.imgsz, rect=True, device=0),
        full_quality=dict(n=676, batch_sizes=[1, 16], decoder='cv2 BGR uint8', matching=MATCHING_VERSION,
                          match_iou=.5, precision_processes='independent'),
        perturbations=dict(gamma=.9, gaussian_sigma=2., gaussian_seed=20261023, jpeg_quality=85,
                           n=16, batch=1, meaning='诊断幅度，未由工艺确认为生产容差；不改几何/GT'),
        e2e=dict(route='/api/detect/image + /api/detect/batch', transport='loopback HTTP multipart',
                 server='本进程随机端口，独立结果目录，既有V4权重硬链接',
                 singles=32, batches16=6, mixed_order='冻结16图循环', state='每次实际save_state，未清空台账',
                 includes='HTTP/表单/PIL解码/BGR转换/预处理/GPU/NMS/画图/编码/写盘/台账JSON/响应',
                 excludes='浏览器/外网/USB盘/队列竞争/路径任务原始文件备份', startup='另计，不混入稳态'),
        decision_rules=dict(quality='逐类TP/FP/FN与成对框差异均报告；任何损伤TP下降或FP增加标为退化，不自动发布',
            speed='报告p50/p95/样本数及原始值；只描述本机上传契约，不外推产线路径吞吐',
            robust='逐变换报告基线与扰动TP/FP/FN；单变换不混合，不训练或调阈值',
            generalization='同集数值回归；V4已见历史数据，不能作为独立质量验收'),
        closure='算力；发布/真值/物理容差为人力或判断关闭')
    dump_new(path, plan)
    print(json.dumps(dict(plan_sha256=file_sha256(path), n=len(rows), selected=plan['selected']), ensure_ascii=False))


def ground_truth(row):
    if not row['label_json']:
        return []
    data = json.loads((ROOT / row['label_json']).read_text(encoding='utf8'))
    return [dict(cls=int(s['label']), xyxy=[min(p[0] for p in s['points']),
                 min(p[1] for p in s['points']), max(p[0] for p in s['points']),
                 max(p[1] for p in s['points'])]) for s in data['shapes']]


def prediction(result):
    data = result.boxes.data.cpu().tolist() if result.boxes is not None else []
    return [dict(xyxy=b[:4], conf=b[4], cls=int(b[5])) for b in data]


def stats(rows, predictions):
    if len(rows) != len(predictions):
        raise ValueError('评测结果数量与清单不符')
    per_image, groups = [], {}
    counts = [[0, 0, 0] for _ in range(3)]  # TP/FP/FN
    for row, pred in zip(rows, predictions):
        gt = ground_truth(row)
        up, ug = match_predictions(pred, gt, .5)
        c = [[0, 0, 0] for _ in range(3)]
        for p in up:
            c[pred[p]['cls']][0] += 1
        for p in set(range(len(pred))) - up:
            c[pred[p]['cls']][1] += 1
        for g in set(range(len(gt))) - ug:
            c[gt[g]['cls']][2] += 1
        key = row['cell_id'] or row['group']
        group = groups.setdefault(key, dict(n_gt=0, fn=0))
        group['n_gt'] += len(gt)
        group['fn'] += len(gt) - len(ug)
        for i in range(3):
            counts[i] = [a+b for a, b in zip(counts[i], c[i])]
        per_image.append(dict(name=row['name'], group=key, counts=c, n_gt=len(gt)))
    metrics = {}
    for label, (tp, fp, fn) in zip(('damage', 'dark', 'light'), counts):
        metrics[label] = dict(tp=tp, fp=fp, fn=fn, recall=tp/(tp+fn) if tp+fn else None)
    return dict(per_class=metrics, fp_per_1000=sum(c[1] for c in counts)*1000/len(rows),
                positive_groups=sum(g['n_gt'] > 0 for g in groups.values()),
                groups_any_gt_missed=sum(g['fn'] > 0 for g in groups.values()),
                n=len(rows), per_image=per_image)


def percentile(values):
    import numpy as np
    return dict(n=len(values), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), values=values)


def load_web(half, sandbox):
    models = sandbox / 'models'
    models.mkdir()
    os.link(WEIGHTS, models / 'best.pt')  # 既有权重硬链接；不复制/变更启用库
    os.environ.update(CT_MODEL_DIR=str(models), CT_RESULTS_DIR=str(sandbox / '结果'),
                      CT_FP16=str(half), CT_NO_BROWSER='1')
    spec = importlib.util.spec_from_file_location('ct_gpu_audit_web', ROOT / 'web/app.py')
    web = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = web
    spec.loader.exec_module(web)
    return web


def http_e2e(web, selected, sandbox):
    """使用真实GPU、真实Web路由和存储函数；不注入假模型或替换写盘。"""
    from werkzeug.serving import make_server, WSGIRequestHandler
    class QuietHandler(WSGIRequestHandler):
        def log(self, *args, **kwargs):
            pass
    server = make_server('127.0.0.1', 0, web.app, request_handler=QuietHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    data = [(r, (ROOT / r['src_path']).read_bytes()) for r in selected]
    times, failures, statuses, outputs = {'single': [], 'batch16': []}, [], [], []
    try:
        for route, repeat in [('single', 32), ('batch16', 6)]:
            for i in range(repeat):
                files = [data[i % len(data)]] if route == 'single' else data
                boundary = 'ct-audit-fixed-boundary-20261023'
                body = bytearray()
                fields = dict(conf='.30', batch_id='e2e-'+route+'-'+str(i),
                              file_paths=json.dumps([r['name']+'.jpg' for r, _ in files]))
                for key, value in fields.items():
                    body.extend(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+key+
                                 '"\r\n\r\n'+value+'\r\n').encode())
                for j, (row, payload) in enumerate(files):
                    key = 'file' if route == 'single' else 'files'
                    body.extend(('--'+boundary+'\r\nContent-Disposition: form-data; name="'+key+
                        '"; filename="sample'+str(j)+'.jpg"\r\nContent-Type: image/jpeg\r\n\r\n').encode())
                    body.extend(payload)
                    body.extend(b'\r\n')
                body.extend(('--'+boundary+'--\r\n').encode())
                url = f'http://127.0.0.1:{port}/api/detect/'+('image' if route == 'single' else 'batch')
                request = urllib.request.Request(url, bytes(body),
                    {'Content-Type': 'multipart/form-data; boundary='+boundary})
                start = time.perf_counter()
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                times[route].append((time.perf_counter()-start)*1000)
                items = [result] if route == 'single' else result['results']
                statuses += [item['status'] for item in items]
                if not result['success'] or any(item['status'] == 'failed' for item in items):
                    failures.append(result)
                outputs.append(dict(route=route, iteration=i, defects=sum(item['target_nums'] for item in items)))
        stored = list(Path(web.RESULTS_FOLDER).iterdir())  # 只统计本脚本沙盘
        return dict(timings_ms={k: percentile(v) for k, v in times.items()}, failures=failures,
                    status_counts={s: statuses.count(s) for s in set(statuses)}, output_counts=outputs,
                    ledger_records=len(web.file_records), sandbox_bytes=sum(p.stat().st_size for p in stored),
                    sandbox_files=len(stored), port=port, server_stopped=True)
    finally:
        server.shutdown()
        thread.join(timeout=10)
        server.server_close()


def measure(plan_path, half, out):
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from image_io import read_bgr_disk, read_bgr_upload
    from training_contract import other_training_processes
    if not torch.cuda.is_available():
        raise ValueError('必须真实CUDA，不可用CPU替代GPU实测')
    plan = json.loads(Path(plan_path).read_text(encoding='utf8'))
    rows, contract, fp = rows_and_contract()
    if (fp != plan['input_fingerprint'] or file_sha256(MANIFEST) != plan['manifest_sha256'] or
        file_sha256(WEIGHTS) != plan['weight_sha256'] or asdict(load_model_spec(WEIGHTS)) != plan['spec'] or
        ultralytics.__version__ != plan['ultralytics'] or
        any(file_sha256(ROOT / p) != v for p, v in plan['sources'].items())):
        raise ValueError('冻结计划的源文件、数据或规格已变')
    active = other_training_processes()
    if active:
        raise RuntimeError('已有训练，拒绝争抢GPU: '+repr(active))
    chosen = [next(r for r in rows if r['name'] == name) for name in plan['selected']]
    result = dict(schema='ct-stage5-measure/1', plan_sha256=file_sha256(plan_path), half=bool(half),
                  weight_sha256=plan['weight_sha256'], input_fingerprint=fp,
                  versions=dict(torch=torch.__version__, ultralytics=ultralytics.__version__, cv2=cv2.__version__),
                  device=torch.cuda.get_device_name(0), start=time.strftime('%Y-%m-%d %H:%M:%S%z'))
    with tempfile.TemporaryDirectory(prefix='ct_gpu_stage5_', dir=ROOT / '.cache') as tmp:
        sandbox = Path(tmp).resolve()
        if not sandbox.is_relative_to((ROOT / '.cache').resolve()):
            raise ValueError('沙盘必须在.cache内')
        start = time.perf_counter()
        web = load_web(half, sandbox)
        torch.cuda.synchronize()
        result['web_load_warmup_ms'] = (time.perf_counter()-start)*1000
        bundle = web.snapshot_model()
        if bundle.model_sha256 != plan['weight_sha256'] or asdict(bundle.spec) != plan['spec']:
            raise ValueError('Web快照不符合冻结规格')
        dtype = str(next(bundle.model.predictor.model.model.parameters()).dtype)
        expected = 'torch.float16' if half else 'torch.float32'
        if dtype != expected:
            raise ValueError('实际dtype错误: '+dtype)
        result['actual_dtype'] = dtype
        # 8.4.146将half别名转换为quantize，不能假设args仍保留half字段。
        result['actual_predict_args'] = {k: getattr(bundle.model.predictor.args, k, None) for k in
                                       ('imgsz', 'iou', 'max_det', 'rect', 'quantize', 'device')}
        result['actual_backend_fp16'] = bool(bundle.model.predictor.model.fp16)
        kw = plan['predict']
        if any(result['actual_predict_args'][k] != kw[k] for k in ('iou', 'max_det', 'rect')):
            raise ValueError('生产预测默认参数已漂移')
        torch.cuda.reset_peak_memory_stats()
        quality = {}
        color = dict(non_gray=0, pixels=0, unequal_pixels=0, max_channel_delta=0)
        for batch_size in plan['full_quality']['batch_sizes']:
            predictions, wall = [], []
            tensor_shapes = set()
            def record_shape(module, inputs):
                tensor_shapes.add(tuple(inputs[0].shape))
            hook = bundle.model.predictor.model.model.register_forward_pre_hook(record_shape)
            for offset in range(0, len(rows), batch_size):
                images = [read_bgr_disk(ROOT / r['src_path']) for r in rows[offset:offset+batch_size]]
                for r, im in zip(rows[offset:offset+batch_size], images):
                    w, h = map(int, r['size'].split('x'))
                    if im.shape != (h, w, 3):
                        raise ValueError('图片尺寸变化: '+r['name'])
                    if batch_size == 1:
                        diff = np.ptp(im.astype(np.int16), axis=2)
                        color['non_gray'] += int(bool(np.any(diff)))
                        color['pixels'] += diff.size
                        color['unequal_pixels'] += int(np.count_nonzero(diff))
                        color['max_channel_delta'] = max(color['max_channel_delta'], int(diff.max()))
                torch.cuda.synchronize()
                start = time.perf_counter()
                res = web._predict_inputs(bundle, images, .30)
                predictions.extend(prediction(r) for r in res)
                torch.cuda.synchronize()
                wall.append((time.perf_counter()-start)*1000)
                if offset % (batch_size*50) == 0:
                    print(f'FP{16 if half else 32} batch{batch_size}: {offset}/{len(rows)}', flush=True)
            quality[str(batch_size)] = dict(predictions=predictions, metrics=stats(rows, predictions),
                                            call_wall_ms=percentile(wall),
                                            actual_network_shapes=sorted(tensor_shapes))
            hook.remove()
        result['color'] = color
        result['quality'] = quality
        # 单因素强度扰动，固定像素位置，GT不变；不对生产母本写盘。
        perturb = {}
        codec = []
        for name in ('baseline', 'gamma_0.9', 'noise_sigma2', 'jpeg_q85'):
            predictions = []
            for index, row in enumerate(chosen):
                im = read_bgr_disk(ROOT / row['src_path'])
                if name == 'baseline':
                    upload = read_bgr_upload(io.BytesIO((ROOT / row['src_path']).read_bytes()))
                    codec.append(dict(name=row['name'], max_abs_delta=int(np.abs(
                        im.astype(np.int16)-upload.astype(np.int16)).max())))
                elif name == 'gamma_0.9':
                    lut = np.rint(255*(np.arange(256)/255.)**.9).astype(np.uint8)
                    im = cv2.LUT(im, lut)
                elif name == 'noise_sigma2':
                    noise = np.random.default_rng(20261023+index).normal(0, 2., im.shape)
                    im = np.clip(np.rint(im.astype(float)+noise), 0, 255).astype(np.uint8)
                else:
                    ok, encoded = cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise ValueError('扰动JPEG编码失败')
                    im = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                predictions.append(prediction(web._predict_inputs(bundle, [im], .30)[0]))
            perturb[name] = dict(predictions=predictions, metrics=stats(chosen, predictions))
        result['perturbations'] = perturb
        result['bounded_decoder_comparison'] = codec
        result['http_e2e'] = http_e2e(web, chosen, sandbox)
        torch.cuda.synchronize()
        result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['end'] = time.strftime('%Y-%m-%d %H:%M:%S%z')
        # 再核对标签和图像身份，避免测量期间漂移；不递归扫描生产目录。
        if rows_and_contract()[2] != fp or file_sha256(WEIGHTS) != plan['weight_sha256']:
            raise ValueError('测量期间输入漂移')
        dump_new(out, result)
        print(json.dumps(dict(out=out, dtype=dtype, n=len(rows)), ensure_ascii=False), flush=True)


def paired_predictions(a, b):
    changed, missing, overlaps, confidence = [], 0, [], []
    for i, (pa, pb) in enumerate(zip(a, b)):
        used_p, used_g = match_predictions(pb, pa, .5)
        missing += len(pa)+len(pb)-len(used_p)*2
        if len(pa) != len(pb) or len(used_p) != len(pa):
            changed.append(i)
        available = set(range(len(pa)))
        for pred in sorted(pb, key=lambda p: -p['conf']):
            candidates = [(iou_xyxy(pred['xyxy'], pa[j]['xyxy']), j) for j in available
                          if pa[j]['cls'] == pred['cls']]
            if not candidates:
                continue
            overlap, j = max(candidates)
            if overlap < .5:
                continue
            available.remove(j)
            overlaps.append(overlap)
            confidence.append(abs(pred['conf']-pa[j]['conf']))
    return dict(changed_count_or_unpaired_images=changed, n_unpaired_boxes=missing,
                paired_iou_min=min(overlaps) if overlaps else None,
                paired_conf_abs_delta_max=max(confidence) if confidence else None)


def bootstrap(a, b):
    """按当前cell_id-or-group成对重采样；不把这些组宣称为完整电芯UID。"""
    import numpy as np
    groups = {}
    for x, y in zip(a['per_image'], b['per_image']):
        groups.setdefault(x['group'], []).append((x, y))
    aggregate = []
    for images in groups.values():
        aggregate.append([[sum(p[j]['counts'][c][k] for p in images) for c in range(3) for k in range(3)]
                          + [len(images)] for j in (0, 1)])
    array = np.array(aggregate, dtype=float)
    rng = np.random.default_rng(20261023)
    values = []
    for _ in range(2000):
        s = array[rng.integers(0, len(array), len(array))].sum(axis=0)
        recall = s[:, 0]/np.maximum(1., s[:, 0]+s[:, 2])
        fp = s[:, [1, 4, 7]].sum(axis=1)*1000/s[:, 9]
        values.append([recall[1]-recall[0], fp[1]-fp[0]])
    return dict(group_count=len(groups), replicates=2000,
                delta_damage_recall_ci95=np.percentile(values, [2.5, 97.5], axis=0)[:, 0].tolist(),
                delta_fp_per_1000_ci95=np.percentile(values, [2.5, 97.5], axis=0)[:, 1].tolist(),
                limitation='条件于当前标签与分组的数值不确定性，不能校正泄漏、误标或外部泛化')


def compare(a_path, b_path, out):
    a, b = [json.loads(Path(p).read_text(encoding='utf8')) for p in (a_path, b_path)]
    if a['half'] or not b['half'] or a['plan_sha256'] != b['plan_sha256']:
        raise ValueError('需要同一冻结计划的FP32与FP16独立进程产物')
    result = dict(schema='ct-stage5-compare/1', plan_sha256=a['plan_sha256'],
                  source_sha256={str(a_path): file_sha256(a_path), str(b_path): file_sha256(b_path)}, quality={})
    for batch in ('1', '16'):
        x, y = a['quality'][batch], b['quality'][batch]
        result['quality'][batch] = dict(fp32=x['metrics'], fp16=y['metrics'],
            pairing=paired_predictions(x['predictions'], y['predictions']),
            paired_group_bootstrap=bootstrap(x['metrics'], y['metrics']))
    result['batch16_vs_batch1'] = {label: paired_predictions(v['quality']['1']['predictions'],
        v['quality']['16']['predictions']) for label, v in [('fp32', a), ('fp16', b)]}
    result['e2e_ms'] = {label: {route: {k: p[k] for k in ('n', 'p50', 'p95')}
                               for route, p in v['http_e2e']['timings_ms'].items()}
                       for label, v in [('fp32', a), ('fp16', b)]}
    result['perturbation_metrics'] = {label: {k: value['metrics'] for k, value in v['perturbations'].items()}
                                    for label, v in [('fp32', a), ('fp16', b)]}
    result['limitations'] = ['同集回归，无独立发布效力', '有界扰动不等同真实扫描器/工艺域移',
        '时延为本机无竞争上传HTTP，n32/n6不足以估计产线长尾，不含外盘/浏览器/路径备份']
    dump_new(out, result)
    print(json.dumps(result['e2e_ms'], ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode', choices=('freeze', 'measure', 'compare'), required=True)
    ap.add_argument('--plan')
    ap.add_argument('--half', type=int, choices=(0, 1))
    ap.add_argument('--out')
    ap.add_argument('--fp32')
    ap.add_argument('--fp16')
    args = ap.parse_args()
    if args.mode == 'freeze':
        freeze(args.plan)
    elif args.mode == 'measure':
        if args.half is None or not args.out:
            ap.error('measure需要--half和--out')
        measure(args.plan, args.half, args.out)
    else:
        compare(args.fp32, args.fp16, args.out)


if __name__ == '__main__':
    main()
