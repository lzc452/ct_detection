# -*- coding: utf-8 -*-
"""统一、可审计的训练入口；默认1920，不自动发布模型。
先以--dry-run --baseline-args <旧args.yaml> --name <新run>输出计划指纹；
用kb.py add-finding登记待验判据、关联run与plan_sha256，再传--finding F-XXX开训。
开训自动执行严格preflight、单变量对照、进程互斥；异常也保留meta并关闭日志。
用法：runtime/python.exe scripts/train_run.py --help
"""
import os
import sys
import json
import time
import datetime
import argparse
import hashlib
import logging
import contextlib
import math
import subprocess
import importlib.metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'core'))
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
from dataset_contract import inspect_dataset
from eval_cache import file_sha256
from path_safety import run_output, within
from training_contract import compare_config, plan_sha256, check_preregistration, training_slot


class _Tee:
    """把 stdout/stderr 同时写进日志文件。

    注意：光把 sys.stdout/sys.stderr 换掉**捕获不到** ultralytics 的日志行——
    它的 `LOGGER` 在 `from ultralytics import YOLO` 那一刻就把 StreamHandler 绑到了
    原始的 sys.stderr 上，之后换 sys.stderr 对它无效（2026-10-22 V51_ROI_1536 实测：
    train.log 只有 8 行表头）。因此还必须额外挂一个 logging.FileHandler（见 attach_logger）。
    这里同时补齐 isatty/fileno/encoding 等方法，避免 tqdm、rich 拿到裸对象后行为异常。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    # ---- 让 tqdm / rich 认为这是个正常的流 ----
    def isatty(self):
        return False

    def writable(self):
        return True

    def readable(self):
        return False

    def fileno(self):
        for s in self.streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError("no underlying fileno")

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.streams[0], "errors", "replace")

    def __getattr__(self, item):
        # 其余属性一律委托给第一条真实流
        return getattr(self.streams[0], item)


def attach_logger(log_path):
    """把 ultralytics 的 LOGGER（以及 root）额外写一份到 log_path。

    这是捕获每轮指标行的**唯一可靠手段**：LOGGER 的 StreamHandler 在 import 时就绑定了
    原始 stderr，事后替换 sys.stderr 拿不到它的输出。
    """
    import logging
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    try:
        import ultralytics
        ultralytics.LOGGER.addHandler(fh)
    except Exception:
        pass
    logging.getLogger().addHandler(fh)
    return fh


@contextlib.contextmanager
def training_log(path):
    stdout, stderr = sys.stdout, sys.stderr
    handler = None
    with open(path, 'x', encoding='utf-8', buffering=1) as log:
        try:
            handler = attach_logger(path)
            sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            if handler is not None:
                try:
                    import ultralytics
                    ultralytics.LOGGER.removeHandler(handler)
                except (ImportError, AttributeError):
                    pass
                logging.getLogger().removeHandler(handler)
                handler.close()


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=os.path.join("models", "pretrained", "yolov8n.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=1920, help="32对齐的正整数画布，默认1920")
    ap.add_argument("--batch", default="0.75", help="数字或 0~1 的显存占比")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fine-tune-from", "--resume-from", dest="resume_from", default="",
                    help="微调来源权重（上一轮 run 的 weights/best.pt）；替换 --model，不复用优化器状态")
    # ---- 优化器 ----
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    # ---- 增强（V5.1 全部显式化，不依赖 ultralytics 默认值）----
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--degrees", type=float, default=5.0)
    ap.add_argument("--translate", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--shear", type=float, default=0.0)
    ap.add_argument("--perspective", type=float, default=0.0)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--hsv-h", type=float, default=0.0)
    ap.add_argument("--hsv-s", type=float, default=0.0)
    ap.add_argument("--hsv-v", type=float, default=0.1)
    ap.add_argument("--erasing", type=float, default=0.4,
                    help="仅分类任务生效；当前检测增强不使用，保留用于历史配置对照")
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="注意：检测任务(无 segments)下 ultralytics 的 CopyPaste 是空操作，设了也无效")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    # ---- 损失 ----
    ap.add_argument("--box", type=float, default=7.5)
    ap.add_argument("--cls", type=float, default=0.5)
    ap.add_argument("--dfl", type=float, default=1.5)
    ap.add_argument("--cls-pw", type=float, default=0.0,
                    help="类别权重幂次 0~1：0 关闭；1=完全逆频率。不预设最佳值，变更须单因素对照")
    ap.add_argument('--roi', action='store_true', help='历史兼容参数；当前D-010禁止新ROI训练')
    ap.add_argument('--roi-k', type=float, default=None, help='不得与数据契约roi_k冲突，不再按目录名猜测')
    ap.add_argument("--no-archive", action="store_true",
                    help="训练结束不自动归档到 experiments/")
    ap.add_argument('--baseline-args', required=True, help='比较用的旧run args.yaml')
    ap.add_argument('--finding', default='', help='待验且绑定本run与计划指纹的预注册条目')
    ap.add_argument('--dry-run', action='store_true', help='只读校验并打印计划，不建run、不初始化模型/GPU')
    ap.add_argument('--cache', choices=['disk', 'ram', '0'], default='disk')
    return ap


def build_plan(args):
    if not args.name:
        raise ValueError('必须显式--name，以便计划与预注册稳定绑定')
    out = run_output(args.name)
    if out.exists():
        raise FileExistsError('run已存在，绝不覆盖或猜测后缀，请换新name: ' + str(out))
    inspection = inspect_dataset(args.data, args.imgsz)
    contract = inspection['contract']
    if contract['roi'] or args.roi:
        raise ValueError('D-010已停止当前ROI训练路线')
    if args.roi_k is not None and args.roi_k != contract['roi_k']:
        raise ValueError('roi_k与数据契约不一致，拒绝按目录名推断')
    model = Path(args.resume_from or args.model).resolve()
    if not model.is_file() or model.suffix.lower() != '.pt':
        raise ValueError('须指定已存在的本地.pt权重，禁止隐式下载')
    batch = float(args.batch)
    if not math.isfinite(batch) or batch <= 0 or (batch >= 1 and not batch.is_integer()):
        raise ValueError('batch须为正整数或0到1的显存比例')
    if args.epochs <= 0 or args.patience < 0 or args.workers < 0:
        raise ValueError('epochs/patience/workers非法')
    fields = ['optimizer', 'lr0', 'lrf', 'weight_decay', 'close_mosaic', 'seed',
              'box', 'cls', 'dfl', 'cls_pw', 'degrees', 'translate', 'scale', 'shear',
              'perspective', 'flipud', 'fliplr', 'hsv_h', 'hsv_s', 'hsv_v',
              'mosaic', 'mixup', 'copy_paste', 'cutmix', 'erasing']
    kw = {key: getattr(args, key) for key in fields}
    if any(isinstance(v, float) and not math.isfinite(v) for v in kw.values()):
        raise ValueError('超参不得为NaN/Inf')
    for key in ['mosaic', 'mixup', 'copy_paste', 'cutmix', 'erasing', 'flipud', 'fliplr',
                'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale', 'perspective', 'cls_pw']:
        if not 0 <= kw[key] <= 1:
            raise ValueError('概率/比例超参越界: ' + key)
    if min(args.lr0, args.lrf, args.box, args.cls, args.dfl) <= 0 or args.weight_decay < 0:
        raise ValueError('学习率/损失/衰减非法')
    kw.update(data=str(Path(args.data).resolve()), epochs=args.epochs, batch=int(batch) if batch >= 1 else batch,
              imgsz=args.imgsz, device=args.device, patience=args.patience,
              project=str(out.parent), name=out.name, exist_ok=True,
              momentum=.937, cos_lr=True, warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=.1,
              cache=False if args.cache == '0' else args.cache, workers=args.workers, val=True, plots=True)
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=dict(model=str(model), **kw))  # 只解析配置，不构造模型或CUDA上下文。
    differences = compare_config(dict(model=str(model), **kw), args.baseline_args)
    if set(differences) & {'erasing', 'copy_paste'}:
        raise ValueError('本数据是框检测，erasing/copy_paste不是有效干预，拒绝为此开训')
    payload = dict(schema='ct-training-plan/1', model=str(model), model_sha256=file_sha256(model),
                   data_contract_sha256=inspection['contract_sha256'],
                   data_yaml_sha256=inspection['data_yaml_sha256'], train_kwargs=kw,
                   baseline_sha256=file_sha256(args.baseline_args),
                   baseline_path=str(Path(args.baseline_args).resolve()),
                   entry_sha256=file_sha256(__file__), ultralytics=importlib.metadata.version('ultralytics'))
    payload['source_sha256'] = {name: file_sha256(PROJECT_ROOT / 'core' / name) for name in
                                ['training_contract.py', 'dataset_contract.py', 'path_safety.py', 'materialize.py']}
    digest = plan_sha256(payload)
    registration, error = None, None
    try:
        registration = check_preregistration(args.finding, args.name, digest)
    except ValueError as exc:
        error = str(exc)
    return dict(payload=payload, plan_sha256=digest, differences=differences, registration=registration,
                ready=registration is not None, preregistration_error=error, run_dir=str(out),
                dataset=inspection['stats'], roi=False, roi_k=0.0, no_archive=args.no_archive,
                fine_tune_from=args.resume_from or None)


def write_meta(path, meta):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(meta, stream, ensure_ascii=False, indent=2, default=str)
    os.replace(temp, path)


def verify_plan_inputs(plan):
    payload = plan['payload']
    if plan_sha256(payload) != plan['plan_sha256']:
        raise ValueError('计划内容已变，须重新dry-run并预注册')
    if payload['ultralytics'] != importlib.metadata.version('ultralytics'):
        raise ValueError('Ultralytics版本已变')
    for name, digest in payload['source_sha256'].items():
        if file_sha256(PROJECT_ROOT / 'core' / name) != digest:
            raise ValueError('训练契约源码已变: ' + name)
    for path, expected in [(payload['model'], payload['model_sha256']),
                           (payload['baseline_path'], payload['baseline_sha256']),
                           (payload['train_kwargs']['data'], payload['data_yaml_sha256']),
                           (Path(payload['train_kwargs']['data']).with_name('data_contract.json'),
                            payload['data_contract_sha256']), (__file__, payload['entry_sha256'])]:
        if file_sha256(path) != expected:
            raise ValueError('计划输入内容已变: ' + str(path))
    registration = check_preregistration(plan['registration']['finding'],
                                        payload['train_kwargs']['name'], plan['plan_sha256'])
    if registration != plan['registration']:
        raise ValueError('预注册条目已变')


def execute_plan(plan, model_factory=None):
    if not plan['ready']:
        raise ValueError(plan['preregistration_error'])
    kw = plan['payload']['train_kwargs']
    if Path(plan['run_dir']).resolve() != run_output(kw['name']) or Path(kw['project']).resolve() != run_output(kw['name']).parent:
        raise ValueError('计划输出路径非法')
    verify_plan_inputs(plan)
    # 正式开训前真的执行仓库规定的CLI，而不是只在文案中提醒。
    subprocess.run([sys.executable, '-B', '-X', 'utf8', str(PROJECT_ROOT / 'scripts/kb.py'), 'preflight',
                    '--data', kw['data'], '--imgsz', str(kw['imgsz'])], check=True)
    with training_slot(Path(plan['run_dir']).parent):
        verify_plan_inputs(plan)
        out = Path(plan['run_dir'])
        out.mkdir(exist_ok=False)
        meta_path = out / '_train_meta.json'
        meta = dict(schema='ct-train-meta/2', state='starting', created_at=time.strftime('%Y-%m-%d %H:%M:%S'),
                    data_yaml=kw['data'], base_model=plan['payload']['model'],
                    base_model_sha256=plan['payload']['model_sha256'], dataset=Path(kw['data']).parent.name,
                    roi=False, roi_k=0.0, roi_source='data-contract', roi_k_source='data-contract',
                    classes={0: 'damage', 1: 'dark', 2: 'light'}, hyperparams=kw,
                    plan=plan, n_train=plan['dataset']['train']['images'], n_val=plan['dataset']['val']['images'],
                    fine_tune_from=plan['fine_tune_from'], resume_optimizer=False)
        write_meta(meta_path, meta)
        model = None
        start = time.monotonic()
        def capture(trainer):
            actual = Path(trainer.save_dir).resolve()
            if actual != out.resolve():
                raise RuntimeError('实际save_dir偏离独占预留目录，拒绝继续: ' + str(actual))
            meta.update(save_dir=str(actual), hyperparams_effective=vars(trainer.args).copy(),
                        actual_batch=getattr(trainer, 'batch_size', None),
                        accumulate=getattr(trainer, 'accumulate', None), state='running')
            write_meta(meta_path, meta)
        try:
            with training_log(out / 'train.log'):
                if model_factory is None:
                    from ultralytics import YOLO
                    model_factory = YOLO
                model = model_factory(plan['payload']['model'])
                model.add_callback('on_pretrain_routine_start', capture)
                model.add_callback('on_train_start', capture)
                model.train(**kw)
                capture(model.trainer)
                best = Path(model.trainer.best).resolve()
                if not within(best, out) or not best.is_file():
                    raise RuntimeError('训练返回但实际best.pt缺失或越界')
                meta.update(state='completed', best_pt=str(best), best_pt_md5=hashlib.md5(best.read_bytes()).hexdigest(),
                            best_pt_sha256=file_sha256(best), completed_epochs=getattr(model.trainer, 'epoch', -1)+1)
        except BaseException as exc:
            meta.update(state='failed', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            meta['elapsed_s'] = round(time.monotonic() - start, 3)
            write_meta(meta_path, meta)
        if not plan['no_archive']:
            try:
                subprocess.run([sys.executable, '-B', '-X', 'utf8', str(PROJECT_ROOT / 'scripts/archive_experiment.py'),
                                '--run', out.name], check=True)
                meta['archive_status'] = 'completed'
            except (OSError, subprocess.CalledProcessError) as exc:
                meta.update(archive_status='failed', archive_error=str(exc))
                raise
            finally:
                write_meta(meta_path, meta)
        return meta


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    meta = execute_plan(plan)
    print('训练完成，未自动发布。实际best.pt: ' + meta['best_pt'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
