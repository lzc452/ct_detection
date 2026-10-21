# -*- coding: utf-8 -*-
"""
复核类脚本的安全执行底座（#4）——默认 dry-run、写前备份、操作编号幂等、可回滚。

为什么要有它：exec_v4_review / prelabel / merge_v4_staging 三个脚本动的是
"人工标注过的母本"（问题电芯图片汇总/），这是项目里最贵的资产，一次误跑不可恢复。
本模块把危险动作收敛成"先建清单 → 打印 → 确认后落盘"，落盘前逐个文件备份到
<root>/_backup/<批次号>/，并写 journal.json 支持整批回滚。

用法（供脚本调用，不要直接运行）：

    from _review_safe import Plan, base_parser
    args = base_parser('脚本说明', epilog='示例').parse_args()
    plan = Plan(args, label='exec_v4_review')
    plan.delete(MASTER / '2-亮斑/2_0244.jpg', 'A 错题删除')
    plan.write_json(jp, doc, 'B 修框')
    if not plan.emit():          # dry-run：打印清单 + 写 plan CSV，然后 return
        return
    plan.commit()                # 落盘（逐项备份 + journal.json）

命令行通用参数：
    --apply            真正落盘（不加=只出清单）
    --batch NAME       批次号，默认时间戳；回滚时用它定位备份
    --backup-dir DIR   备份根目录，默认 _backup
    --force            忽略"本脚本已执行过"的标记
    --rollback BATCH   按批次号回滚（需配合 --apply 才真正还原）
"""
import argparse
import csv
import datetime
import json
import shutil
import sys
from pathlib import Path

OP_DELETE = 'DELETE'
OP_WRITE = 'WRITE'
OP_MOVE = 'MOVE'
OP_COPY = 'COPY'
OP_APPEND = 'APPEND'

DEFAULT_BACKUP_DIR = '_backup'


# ==================== 命令行 ====================

def base_parser(desc, epilog='', root=None):
    ap = argparse.ArgumentParser(
        description=desc, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--apply', action='store_true',
                    help='真正落盘；不加此参数只打印操作清单（默认 dry-run）')
    ap.add_argument('--batch', default=None,
                    help='批次号（默认当前时间戳），回滚时用它定位 _backup/<批次号>/')
    ap.add_argument('--backup-dir', default=DEFAULT_BACKUP_DIR,
                    help='备份根目录（相对工程根），默认 %s' % DEFAULT_BACKUP_DIR)
    ap.add_argument('--force', action='store_true',
                    help='忽略"已执行过"标记，强制再跑一遍')
    ap.add_argument('--rollback', metavar='BATCH', default=None,
                    help='按批次号回滚；不加 --apply 只预览将要还原什么')
    ap.add_argument('--root', default=str(root) if root else None,
                    help='工程根目录，默认由脚本位置推导')
    return ap


# ==================== 计划 ====================

class Plan:
    """把"要动的文件"先记账，再统一执行。

    每个操作在执行前都会把目标文件（若存在）复制到
    <root>/<backup-dir>/<批次号>/<相对路径>，因此任何一步都能整批回滚。
    """

    def __init__(self, args, label, root=None):
        self.label = label
        self.apply = bool(getattr(args, 'apply', False))
        self.force = bool(getattr(args, 'force', False))
        raw_root = root or getattr(args, 'root', None) or Path(__file__).resolve().parents[1]
        self.root = Path(raw_root).resolve()
        self.batch = getattr(args, 'batch', None) or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.backup_dir_name = getattr(args, 'backup_dir', DEFAULT_BACKUP_DIR) or DEFAULT_BACKUP_DIR
        self.backup_root = self.root / self.backup_dir_name
        self.batch_dir = self.backup_root / self.batch
        self.ops = []
        self.notes = []
        self._seen = set()

    # ---------- 记账 ----------
    def note(self, text):
        """人类可读的补充说明（不进 CSV，只打印）"""
        self.notes.append(text)

    def _add(self, op, target, src=None, note='', data=None):
        target = Path(target)
        # 内容也进指纹：同一文件先写 A 再追加 B 是两次独立操作，不能被去重吃掉
        try:
            dkey = repr(data) if data is not None else ''
        except Exception:
            dkey = 'obj'
        key = (op, str(target), str(src or ''), dkey)
        if key in self._seen:      # 同一操作重复登记只算一次
            return
        self._seen.add(key)
        self.ops.append({'op': op, 'target': target,
                         'src': Path(src) if src else None,
                         'note': note, 'data': data})

    def delete(self, path, note=''):
        if Path(path).exists():
            self._add(OP_DELETE, path, note=note)
            return True
        return False

    def write_text(self, path, text, note=''):
        self._add(OP_WRITE, path, note=note, data=text)

    def write_bytes(self, path, raw, note=''):
        self._add(OP_WRITE, path, note=note, data=raw)

    def write_json(self, path, obj, note='', stamp_op=True):
        """写 labelme json；stamp_op=True 时在文档里留 ct_meta（执行后补 op_id）"""
        if stamp_op and isinstance(obj, dict):
            meta = obj.setdefault('ct_meta', {})
            meta.setdefault('script', self.label)
            meta.setdefault('batch', self.batch)
            meta['op_id'] = None       # commit 时填
        self._add(OP_WRITE, path, note=note, data=obj)

    def move(self, src, dst, note=''):
        self._add(OP_MOVE, dst, src=src, note=note)

    def copy(self, src, dst, note=''):
        self._add(OP_COPY, dst, src=src, note=note)

    def append_text(self, path, text, note=''):
        self._add(OP_APPEND, path, note=note, data=text)

    # ---------- 清单 ----------
    def op_id(self, i):
        return '%s-%03d' % (self.batch, i)

    def _rows(self):
        rows = []
        for i, op in enumerate(self.ops, 1):
            rows.append((self.op_id(i), op['op'], str(op['target']),
                         str(op['src'] or ''), op['note']))
        return rows

    def summary(self):
        c = {}
        for op in self.ops:
            c[op['op']] = c.get(op['op'], 0) + 1
        return c

    def emit(self, limit=200):
        """打印清单 + 写 plan CSV。dry-run 返回 False（调用方据此提前 return）。"""
        rows = self._rows()
        for r in rows[:limit]:
            print('%-6s %s  %s%s' % (r[1], r[0], r[2], ('  ← ' + r[3]) if r[3] else ''))
        if len(rows) > limit:
            print('... 其余 %d 项见 plan CSV' % (len(rows) - limit))
        for n in self.notes:
            print('  ' + n)

        plan_dir = self.backup_root / 'plans'
        plan_dir.mkdir(parents=True, exist_ok=True)
        plan_csv = plan_dir / ('plan_%s_%s.csv' % (self.label, self.batch))
        with open(plan_csv, 'w', encoding='utf-8-sig', newline='') as f:
            w = csv.writer(f)
            w.writerow(['操作编号', '操作', '目标', '来源', '说明'])
            w.writerows(rows)

        c = self.summary()
        stat = ' / '.join('%s %d' % (k, v) for k, v in sorted(c.items())) or '无操作'
        print('\n[%s] 共 %d 项（%s）' % (self.label, len(rows), stat))
        print('[清单] %s' % plan_csv)
        if not self.apply:
            print('[dry-run] 未改动任何文件。确认无误后加 --apply 执行；'
                  '\n          执行后可用 --rollback %s --apply 回滚。' % self.batch)
            return False
        return True

    # ---------- 执行 ----------
    def _rel(self, p):
        """备份用的相对路径：不在工程内时退回文件名（加 external 前缀）"""
        p = Path(p).resolve()
        try:
            return p.relative_to(self.root)
        except ValueError:
            return Path('external') / p.name

    def _backup(self, p):
        """备份单个文件，返回备份路径（目标不存在/是目录时返回 None）

        同一文件在同一批次被多次触及（先写后追加）时，备份名递增，
        绝不覆盖前一份备份——否则回滚拿到的是中间态而不是原始内容。
        """
        p = Path(p)
        if not p.is_file():
            return None
        bak = self.batch_dir / self._rel(p)
        bak.parent.mkdir(parents=True, exist_ok=True)
        if bak.exists():
            i = 2
            while True:
                cand = bak.with_name(bak.name + '.__%d' % i)
                if not cand.exists():
                    bak = cand
                    break
                i += 1
        shutil.copy2(str(p), str(bak))
        return bak

    def applied_marker(self, extra=None):
        return self.batch_dir / ('applied_%s.json' % self.label)

    def already_applied(self):
        """上一次同脚本的 APPLY 是否已完成（幂等闸门）"""
        if not self.batch_dir.exists():
            return None
        for m in self.batch_dir.glob('applied_*.json'):
            try:
                info = json.loads(m.read_text(encoding='utf-8'))
            except Exception:
                continue
            if info.get('label') == self.label:
                return info
        return None

    def commit(self):
        """依次执行；每执行完一项就刷新 journal，部分失败也能回滚"""
        self.batch_dir.mkdir(parents=True, exist_ok=True)
        prev = self.already_applied()
        if prev and not self.force:
            print('[跳过] 本批次 %s 已由 %s 执行过（%s）。重跑加 --force，或先 --rollback。'
                  % (self.batch, self.label, prev.get('time', '')))
            return 0

        journal = []
        ok = 0
        for i, op in enumerate(self.ops, 1):
            rec = {'op_id': self.op_id(i), 'op': op['op'],
                   'target': str(op['target']), 'src': str(op['src'] or ''),
                   'note': op['note'], 'backup': None, 'existed': False,
                   'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            try:
                self._exec(op, rec)
                rec['ok'] = True
                ok += 1
            except Exception as e:
                rec['ok'] = False
                rec['error'] = '%s: %s' % (type(e).__name__, e)
                journal.append(rec)
                self._write_journal(journal)
                print('[FAIL] %s %s -> %s' % (rec['op'], rec['target'], rec['error']))
                print('[提示] 已完成 %d 项，可用 --rollback %s --apply 还原。' % (ok, self.batch))
                raise
            journal.append(rec)
            self._write_journal(journal)
            print('[%s] %s %s' % (rec['op'], rec['op_id'], rec['target']))

        info = {'label': self.label, 'batch': self.batch, 'ops': len(self.ops),
                'ok': ok, 'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'root': str(self.root)}
        self.applied_marker().write_text(json.dumps(info, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
        print('\n[完成] %d 项已落盘；备份 %s' % (ok, self.batch_dir))
        print('[回滚] python scripts/review_rollback.py --batch %s --apply' % self.batch)
        return ok

    def _exec(self, op, rec):
        t = Path(op['target'])
        # 只备份"将被覆盖的目标"：MOVE 的反向动作是移回去，COPY 的反向动作是删掉副本，
        # 都不需要源的备份（备份源会让回滚把副本"还原"出来，等于没回滚）
        if t.exists():
            rec['existed'] = True
            b = self._backup(t)
            if b:
                rec['backup'] = str(b)

        kind = op['op']
        if kind == OP_DELETE:
            t.unlink()
        elif kind == OP_WRITE:
            t.parent.mkdir(parents=True, exist_ok=True)
            data = op['data']
            if isinstance(data, dict):
                meta = data.get('ct_meta')
                if isinstance(meta, dict) and meta.get('op_id') is None:
                    meta['op_id'] = rec['op_id']
                t.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
            elif isinstance(data, bytes):
                t.write_bytes(data)
            else:
                t.write_text(data, encoding='utf-8')
        elif kind == OP_MOVE:
            if t.exists():
                # 目标已存在绝不能覆盖：那可能是另一份人工标注
                raise FileExistsError('目标已存在，拒绝覆盖：%s' % t)
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(op['src']), str(t))
        elif kind == OP_COPY:
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(op['src']), str(t))
        elif kind == OP_APPEND:
            t.parent.mkdir(parents=True, exist_ok=True)
            with open(t, 'a', encoding='utf-8') as f:
                f.write(op['data'])
        else:
            raise ValueError('未知操作 %r' % kind)

    def _write_journal(self, journal):
        (self.batch_dir / 'journal.json').write_text(
            json.dumps(journal, ensure_ascii=False, indent=1), encoding='utf-8')


# ==================== 回滚 ====================

def rollback(root, batch, backup_dir=DEFAULT_BACKUP_DIR, apply=False):
    """按批次号逆序还原。返回 (还原项数, 失败列表)"""
    root = Path(root).resolve()
    batch_dir = root / backup_dir / batch
    jp = batch_dir / 'journal.json'
    if not jp.exists():
        raise SystemExit('找不到 journal：%s' % jp)
    journal = json.loads(jp.read_text(encoding='utf-8'))
    done = [r for r in journal if r.get('ok')]
    fails = []
    n = 0
    for rec in reversed(done):
        t = Path(rec['target'])
        bak = Path(rec['backup']) if rec.get('backup') else None
        act = None
        if rec['op'] == OP_MOVE:
            act = ('把 %s 移回 %s' % (t, rec['src']))
            if apply:
                Path(rec['src']).parent.mkdir(parents=True, exist_ok=True)
                if t.exists():
                    shutil.move(str(t), str(rec['src']))
                n += 1
        elif rec['op'] in (OP_DELETE, OP_WRITE, OP_APPEND):
            if bak and bak.exists():
                act = '还原 %s ← 备份' % t
                if apply:
                    t.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(bak), str(t))
                    n += 1
            else:
                act = '删除 %s（原本不存在）' % t
                if apply:
                    t.unlink(missing_ok=True)
                    n += 1
        elif rec['op'] == OP_COPY:
            if bak and bak.exists():
                act = '还原 %s ← 备份' % t
                if apply:
                    shutil.copy2(str(bak), str(t))
                    n += 1
            else:
                act = '删除 %s（本次新建）' % t
                if apply:
                    t.unlink(missing_ok=True)
                    n += 1
        if act:
            print('  %-8s %s' % (rec['op'], act))
    print('\n%s %d 项' % ('[已回滚]' if apply else '[dry-run 将回滚]', len(done) if not apply else n))
    if not apply:
        print('确认无误后加 --apply')
    return n, fails
