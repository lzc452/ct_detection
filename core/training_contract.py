# -*- coding: utf-8 -*-
"""开训门槛：有效配置对照、预注册计划指纹、同工程训练互斥。"""
import contextlib
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import yaml
from eval_cache import file_sha256
from path_safety import ROOT, safe_name

# 这些是产物/运行位置，非模型干预。其余请求键都参与逐项比较。
OPERATIONAL = {'project', 'name', 'exist_ok', 'verbose', 'plots'}


def canonical(value, key):
    if key == 'device':
        return str(value).lower()
    if key in {'data', 'model'} and isinstance(value, str):
        value = value.replace('\\', '/')
        return os.path.normcase(str((ROOT / value).resolve()))
    return value


def compare_config(requested, baseline_path):
    baseline = yaml.safe_load(Path(baseline_path).read_text(encoding='utf-8'))
    if not isinstance(baseline, dict):
        raise ValueError('基线args.yaml无效')
    differences = {}
    for key, value in requested.items():
        if key in OPERATIONAL:
            continue
        if key not in baseline or canonical(value, key) != canonical(baseline[key], key):
            differences[key] = dict(before=baseline.get(key), after=value)
    if len(differences) > 1:
        raise ValueError('配置变化超过1项，拒绝开训: ' + json.dumps(differences, ensure_ascii=False))
    return differences


def plan_sha256(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def check_preregistration(finding, run_name, plan_hash, findings_path=None):
    text = Path(findings_path or ROOT / 'experiments/FINDINGS.md').read_text(encoding='utf-8')
    if not re.fullmatch(r'F-\d{3,}', finding or ''):
        raise ValueError('实际开训必须指定待验--finding F-XXX')
    match = re.search(r'^### ' + re.escape(finding) + r'\s.*?(?=^### |^## |\Z)', text, re.M | re.S)
    if not match:
        raise ValueError('预注册结论不存在: ' + finding)
    entry = match.group()
    if ('**状态**：待验' not in entry or f'**关联 run**：`{run_name}`' not in entry
            or f'plan_sha256={plan_hash}' not in entry):
        raise ValueError('预注册必须待验、关联本run，并在证据中登记精确plan_sha256')
    return dict(finding=finding, entry_sha256=hashlib.sha256(entry.encode('utf-8')).hexdigest())


def other_training_processes():
    """只读当前进程命令，不杀进程；不把生产Web当作训练。"""
    if os.name != 'nt':
        raise RuntimeError('当前入口的训练进程核对仅针对目标Windows环境')
    command = ('[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; '
               'Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" } | '
               'Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress')
    powershell = Path(os.environ.get('WINDIR', r'C:\Windows')) / 'System32/WindowsPowerShell/v1.0/powershell.exe'
    result = subprocess.run([str(powershell), '-NoProfile', '-Command', command],
                            capture_output=True, text=True, encoding='utf-8', check=True, timeout=20)
    rows = json.loads(result.stdout.lstrip('\ufeff')) if result.stdout.strip() else []
    rows = rows if isinstance(rows, list) else [rows]
    return [r for r in rows if r['ProcessId'] != os.getpid()
            and re.search(r'(?:scripts[\\/]+)?train(?:_run)?\.py(?:\s|"|$)',
                          r.get('CommandLine') or '', re.I)]


@contextlib.contextmanager
def training_slot(out_root):
    """本工程开训互斥；崩溃遗留锁不自动删除，防止抢占仍在运行的训练。"""
    active = other_training_processes()
    if active:
        raise RuntimeError('已有训练进程，拒绝争抢GPU: ' + repr(active))
    folder = Path(out_root)
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / '.ct_training.lock'
    with lock.open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid()), stream)
    try:
        yield
    finally:
        lock.unlink()
