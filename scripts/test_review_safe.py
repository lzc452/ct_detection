# -*- coding: utf-8 -*-
"""
_review_safe 安全网自测（#4）：在临时沙盘里验证"计划 → 落盘 → 备份 → 回滚"。

覆盖点：
1. dry-run（不 commit）绝不改文件；
2. commit 后目标删除/改写/移动/复制都按计划生效；
3. 每个被触及的文件在 _backup/<批次>/ 下有原始副本；
4. rollback 后沙盘恢复成执行前逐字节一致（md5 全等）；
5. 同一文件先写后追加，两份备份都保留，回滚能回到最初内容。

用法：runtime\\python.exe scripts/test_review_safe.py
"""
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_safe import Plan, rollback   # noqa: E402


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)).replace('\\', '/'): md5(p)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


class Args:
    def __init__(self, **kw):
        self.apply = kw.get('apply', True)
        self.batch = kw.get('batch', 'TESTBATCH')
        self.backup_dir = kw.get('backup_dir', '_backup')
        self.force = kw.get('force', False)
        self.root = kw.get('root', None)
        self.rollback = None


def build_sandbox(tmp):
    img = tmp / '问题电芯图片汇总' / '1-暗斑' / '1_0001.jpg'
    img.parent.mkdir(parents=True)
    img.write_bytes(b'JPEGDATA-ORIGINAL')
    js = img.with_suffix('.json')
    js.write_text(json.dumps({'shapes': [{'label': '1', 'points': [[1, 1], [9, 9]]}]}),
                  encoding='utf-8')
    neg = tmp / '问题电芯图片汇总' / '负样本' / '9_0007.jpg'
    neg.parent.mkdir(parents=True)
    neg.write_bytes(b'NEGIMG')
    stage = tmp / '待微调_V4捡漏' / '暗斑_补标'
    stage.mkdir(parents=True)
    (stage / 'newimg.jpg').write_bytes(b'NEWIMG')
    (stage / 'newimg.json').write_text(json.dumps(
        {'shapes': [{'label': '1', 'points': [[2, 2], [8, 8]]}]}), encoding='utf-8')
    log = tmp / '问题电芯图片汇总' / '删除记录.txt'
    log.write_text('# 旧记录\n', encoding='utf-8')
    return img, js, neg, stage, log


def case_dryrun_untouched(tmp):
    before = snapshot(tmp)
    args = Args(apply=False)
    plan = Plan(args, label='t_dry', root=tmp)
    plan.delete(tmp / '问题电芯图片汇总' / '1-暗斑' / '1_0001.jpg')
    plan.write_text(tmp / '问题电芯图片汇总' / '删除记录.txt', '重写')
    assert plan.emit() is False, 'dry-run 的 emit() 必须返回 False'
    after = snapshot(tmp)
    after.pop('_backup/plans/plan_t_dry_TESTBATCH.csv', None)   # 清单 CSV 允许写
    assert before == after, 'dry-run 改动了文件：%s' % (
        set(before.items()) ^ set(after.items()))
    return 'dry-run 零改动'


def case_apply_and_rollback(tmp):
    img, js, neg, stage, log = (tmp / '问题电芯图片汇总' / '1-暗斑' / '1_0001.jpg',
                                tmp / '问题电芯图片汇总' / '1-暗斑' / '1_0001.json',
                                tmp / '问题电芯图片汇总' / '负样本' / '9_0007.jpg',
                                tmp / '待微调_V4捡漏' / '暗斑_补标',
                                tmp / '问题电芯图片汇总' / '删除记录.txt')
    before = snapshot(tmp)
    before.pop('_backup/plans/plan_t_dry_TESTBATCH.csv', None)

    args = Args(apply=True, batch='T2')
    plan = Plan(args, label='t_apply', root=tmp)
    plan.delete(img, '删除测试')
    plan.write_json(js, {'shapes': [{'label': '1', 'points': [[0, 0], [5, 5]]}]}, '改标注')
    plan.write_text(log, '# 新内容\n', '重写日志')
    plan.append_text(log, '追加一行\n', '追加日志')
    plan.move(neg, tmp / '问题电芯图片汇总' / '1-暗斑' / '9_0007.jpg', '转正')
    plan.copy(stage / 'newimg.jpg', tmp / '问题电芯图片汇总' / '1-暗斑' / 'newimg.jpg', '拷贝')
    assert plan.emit() is True
    plan.commit()

    assert not img.exists(), '删除没生效'
    doc = json.loads(js.read_text(encoding='utf-8'))
    assert len(doc['shapes']) == 1 and doc['shapes'][0]['points'][0] == [0, 0], '改写没生效'
    assert doc.get('ct_meta', {}).get('op_id'), 'op_id 没写进 json'
    assert '追加一行' in log.read_text(encoding='utf-8'), '追加没生效'
    assert (tmp / '问题电芯图片汇总' / '1-暗斑' / '9_0007.jpg').exists(), '移动没生效'
    assert not neg.exists(), '移动后源文件还在'

    bak = tmp / '_backup' / 'T2'
    assert (bak / '问题电芯图片汇总' / '1-暗斑' / '1_0001.jpg').exists(), '删除前没备份图片'
    assert (bak / '问题电芯图片汇总' / '1-暗斑' / '1_0001.json').exists(), '改写前没备份 json'
    assert (bak / '问题电芯图片汇总' / '删除记录.txt').exists(), '重写前没备份日志'

    # 同一文件被写两次（改写 + 追加）→ 应有两份备份，且第一份是最初内容
    baks = sorted((bak / '问题电芯图片汇总').glob('删除记录.txt*'))
    assert len(baks) == 2, '多次触及同一文件应保留 2 份备份，实际 %d' % len(baks)

    rollback(tmp, 'T2', apply=True)
    after = snapshot(tmp)
    after.pop('_backup/plans/plan_t_dry_TESTBATCH.csv', None)
    after = {k: v for k, v in after.items() if not k.startswith('_backup/')}
    diff = [k for k in set(before) | set(after) if before.get(k) != after.get(k)]
    assert not diff, '回滚后不一致：%s' % diff
    return '落盘 + 备份 + 回滚一致'


def case_idempotent(tmp):
    # 同批次重复 commit：标记存在，第二次应跳过
    args = Args(apply=True, batch='T2')
    plan = Plan(args, label='t_apply', root=tmp)
    plan.write_text(tmp / 'x.txt', 'should-not-write', '重复执行')
    plan.emit()
    n = plan.commit()
    assert n == 0 and not (tmp / 'x.txt').exists(), '同批次重复执行没有跳过'
    plan2 = Plan(Args(apply=True, batch='T2', force=True), label='t_apply', root=tmp)
    plan2.write_text(tmp / 'x.txt', 'forced', '强制执行')
    plan2.emit()
    assert plan2.commit() == 1 and (tmp / 'x.txt').read_text(encoding='utf-8') == 'forced'
    (tmp / 'x.txt').unlink()
    return '幂等闸门生效'


def main():
    tmp = Path(tempfile.mkdtemp(prefix='review_safe_'))
    fails = []
    try:
        build_sandbox(tmp)
        for name, fn in (('dry_run', case_dryrun_untouched),
                         ('apply_rollback', case_apply_and_rollback),
                         ('idempotent', case_idempotent)):
            try:
                msg = fn(tmp)
                print('  [PASS] %-14s %s' % (name, msg))
            except AssertionError as e:
                fails.append('%s: %s' % (name, e))
                print('  [FAIL] %-14s %s' % (name, e))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if fails:
        print('\n[FAIL] %d 项' % len(fails))
        return 1
    print('\n[PASS] _review_safe 备份/回滚/幂等全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
