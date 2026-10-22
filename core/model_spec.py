# -*- coding: utf-8 -*-
"""推理规格登记：新权重必须有哈希绑定meta；旧1920兼容仅限已登记的精确权重。"""
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from eval_cache import file_sha256

# 2026-10-23从现有人工启用目录精确读取；仅登记既有Web兼容口径，不是新发布/质量认可。
LEGACY_FULL_MODELS = {
    '030fedde2456077c5ed6c0687eb7a81f7270010df162fb03e5589c14e506acdb': '20261014_v2mild',
    'fac289704787d21394eadfe1e96c40cd1f18d069531734e84cc0d6ecc0bb6756': '20261016_V3',
    'cc54bf9a5d5db9f7ff88fd241d52645c6718b968060de5704b67a892c28d480c': '20261019_V4',
}


@dataclass(frozen=True)
class InferenceSpec:
    imgsz: int = 1920
    roi: bool = False
    roi_k: float = 0.0
    source: str = 'registered-full-1920'


def load_model_spec(pt_path):
    path = Path(pt_path).resolve()
    candidates = [path.parent / '_train_meta.json', path.parent.parent / '_train_meta.json']
    meta_path = next((p for p in candidates if p.exists()), None)
    if meta_path is None:
        legacy = LEGACY_FULL_MODELS.get(file_sha256(path))
        if legacy is None:
            raise ValueError('未知权重缺少_train_meta.json，不能猜测全图1920规格')
        return InferenceSpec(source='legacy-compat-1920:' + legacy)
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    if not isinstance(meta, dict) or not isinstance(meta.get('hyperparams'), dict):
        raise ValueError('meta缺少hyperparams对象')
    size, roi = meta['hyperparams'].get('imgsz'), meta.get('roi')
    k = meta.get('roi_k', 0.0 if roi is False else None)
    if type(size) is not int or size <= 0 or size % 32 or type(roi) is not bool:
        raise ValueError('meta imgsz必须32对齐、roi必须为布尔值')
    if isinstance(k, bool) or not isinstance(k, (float, int)) or not math.isfinite(k) or k < 0:
        raise ValueError('meta必须显式给出非负有限roi_k')
    if not roi and k != 0:
        raise ValueError('全图meta的roi_k必须为0')
    if meta.get('state') not in {None, 'completed'}:
        raise ValueError('训练未完成或失败，不接受为正式Web权重')
    if 'classes' in meta:
        if not isinstance(meta['classes'], dict) or {str(k): v for k, v in meta['classes'].items()} != {
                '0': 'damage', '1': 'dark', '2': 'light'}:
            raise ValueError('meta类别映射与本任务不符')
    if meta.get('best_pt_sha256'):
        matches = meta['best_pt_sha256'] == file_sha256(path)
    elif meta.get('best_pt_md5'):
        matches = meta['best_pt_md5'] == hashlib.md5(path.read_bytes()).hexdigest()
    else:
        matches = False
    if not matches:
        raise ValueError('权重与meta哈希未绑定或不匹配')
    return InferenceSpec(size, roi, float(k), str(meta_path))
