# -*- coding: utf-8 -*-
"""Windows输出路径约束：只创建派生产物，永不覆盖/递归删除用户目录。"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESERVED = {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)),
            *(f'LPT{i}' for i in range(1, 10))}


def safe_name(name):
    if (not isinstance(name, str) or not name or name in {'.', '..'}
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name[-1] in '. ' or name.split('.')[0].upper() in RESERVED):
        raise ValueError('名称必须是单个安全Windows目录/文件名: ' + repr(name))
    return name


def within(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    return path != parent and parent in path.parents


def dataset_output(root, name):
    """仅允许工程datasets_v*或.cache子树；解析junction/symlink后再检查。"""
    safe_name(name)
    requested = Path(root)
    candidate = (ROOT / requested / name).resolve()
    if not within(candidate, ROOT):
        raise ValueError('派生数据输出必须位于当前工程内')
    first = candidate.relative_to(ROOT.resolve()).parts[0]
    if first != '.cache' and not re.fullmatch(r'datasets_v[0-9][A-Za-z0-9_]*', first):
        raise ValueError('派生数据仅可写datasets_v*或.cache，不可写原始数据/runtime/models')
    # .cache本身或datasets_v*本身不能被当作某次数据集。
    if len(candidate.relative_to(ROOT.resolve()).parts) < 2:
        raise ValueError('必须指定独立数据集子目录')
    return candidate


def run_output(name):
    safe_name(name)
    parent = ROOT / 'runs/detect/CT_pro'
    if parent.resolve() != ROOT.resolve() / 'runs/detect/CT_pro':
        raise ValueError('训练输出父目录不可为junction/symlink重定向')
    candidate = (parent / name).resolve()
    if not within(candidate, ROOT) or not within(candidate, parent):
        raise ValueError('训练输出越出当前工程的runs/detect/CT_pro')
    return candidate
