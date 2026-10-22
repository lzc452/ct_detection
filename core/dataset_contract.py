# -*- coding: utf-8 -*-
"""派生数据集完成契约与严格只读preflight；按清单精确访问，不扫描生产母本。"""
import json
import math
from pathlib import Path

import yaml
from eval_cache import file_sha256
from path_safety import ROOT, dataset_output

SCHEMA = 'ct-dataset/1'
NAMES = {0: 'damage', 1: 'dark', 2: 'light'}


def check_yolo(text, name=''):
    """空文件是显式负样本；缺文件不能用空字符串冒充。"""
    boxes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] not in {'0', '1', '2'}:
            raise ValueError('YOLO类别/列数错误: ' + name)
        xywh = [float(v) for v in parts[1:]]
        if not all(math.isfinite(v) for v in xywh):
            raise ValueError('YOLO含非有限坐标: ' + name)
        x, y, w, h = xywh
        if (w <= 0 or h <= 0 or min(x-w/2, y-h/2) < -1e-6
                or max(x+w/2, y+h/2) > 1+1e-6):
            raise ValueError('YOLO坐标越界/退化: ' + name)
        boxes.append((int(parts[0]), *xywh))
    if len(boxes) != len(set(boxes)):
        raise ValueError('重复YOLO标注: ' + name)
    return boxes


def inspect_dataset(data_yaml, imgsz=None, require_training=True):
    """只接受物化器完整产物；旧目录不猜规格，须按清单构建独立版本。"""
    path = Path(data_yaml).resolve()
    out = path.parent
    # 路径必须是允许的派生目录；不去遍历raw datasets/或生产母本。
    dataset_output(out.parent, out.name)
    if imgsz is not None and (type(imgsz) is not int or imgsz <= 0 or imgsz % 32):
        raise ValueError('imgsz必须为32对齐正整数')
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(config, dict) or config.get('download'):
        raise ValueError('无效data.yaml或存在自动下载指令')
    if config.get('nc') != 3 or config.get('names') != NAMES:
        raise ValueError('类别必须严格为0=damage,1=dark,2=light')
    if Path(config.get('path', '')).resolve() != out:
        raise ValueError('data.yaml path必须指向自身目录，不允许回退或指向生产母本')
    contract_path = out / 'data_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    if contract.get('schema') != SCHEMA or contract.get('state') != 'complete':
        raise ValueError('数据集未完成严格物化')
    if type(contract.get('roi')) is not bool:
        raise ValueError('数据契约缺少明确roi位')
    k = contract.get('roi_k')
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 0:
        raise ValueError('数据契约缺少合法roi_k')
    if not contract['roi'] and k != 0:
        raise ValueError('全图数据roi_k必须为0')
    outputs = contract.get('outputs')
    if not isinstance(outputs, dict) or not outputs:
        raise ValueError('数据契约缺少产物清单')
    actual, stats = set(), {}
    splits = contract.get('splits', [])
    if require_training and not {'train', 'val'} <= set(splits):
        raise ValueError('训练必须同时具有train和val')
    if not splits or len(splits) != len(set(splits)) or not set(splits) <= {'train', 'val', 'test'}:
        raise ValueError('数据split无效')
    for split in splits:
        if config.get(split) != f'{split}/images':
            raise ValueError('data.yaml split路径与契约不一致')
        images, labels = out / split / 'images', out / split / 'labels'
        if any(p.resolve() != p.absolute() for p in (images, labels)):
            raise ValueError('split目录不可为junction/symlink重定向')
        image_paths = [p for p in images.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        label_paths = list(labels.glob('*.txt'))
        if not image_paths or {p.stem for p in image_paths} != {p.stem for p in label_paths}:
            raise ValueError('空split或图片标签不一一对应: ' + split)
        if len({p.stem for p in image_paths}) != len(image_paths):
            raise ValueError('同名不同扩展图片: ' + split)
        n_box = n_neg = 0
        for p in image_paths + label_paths:
            relative = p.relative_to(out).as_posix()
            actual.add(relative)
            # 拒绝指向外部的软链接；图片硬链接合法，训练代码不得回写JPG。
            if p.is_symlink() or not p.is_file():
                raise ValueError('派生产物不能是软链接或目录: ' + relative)
            if outputs.get(relative) != file_sha256(p):
                raise ValueError('派生图片/标签内容已偏离完成契约: ' + relative)
            if p.suffix == '.txt':
                boxes = check_yolo(p.read_text(encoding='utf-8'), relative)
                n_box += len(boxes)
                n_neg += not boxes
        stats[split] = dict(images=len(image_paths), boxes=n_box, negative=n_neg)
    if actual != set(outputs):
        raise ValueError('物化产物清单漂移')
    return dict(contract=contract, contract_sha256=file_sha256(contract_path),
                data_yaml_sha256=file_sha256(path), stats=stats, root=str(out))
