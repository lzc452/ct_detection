# -*- coding: utf-8 -*-
"""
merge_v4_staging 行为测试（#4）：编号不撞车 / 幂等标记 / json 合并 / 可回滚。

在临时沙盘里跑真脚本（subprocess --root <tmp> --apply），不动仓库里的任何数据。
覆盖：
1. 一批里两张负样本转正 → 编号必须连续且不同（旧版 max+1 会撞成同一号并互相覆盖图片）；
2. 同名 json 合并 → 按坐标去重，只加新框；
3. 空标注 json 跳过；
4. 重跑 → 全部被 .merged 标记跳过（幂等）；
5. --rollback → 沙盘恢复到执行前逐字节一致。

用法：runtime\\python.exe scripts/test_merge_staging.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / 'merge_v4_staging.py'


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def snapshot(root):
    return {str(p.relative_to(root)).replace('\\', '/'): md5(p)
            for p in sorted(Path(root).rglob('*')) if p.is_file()}


def shape(label, x1, y1, x2, y2):
    return {"label": label, "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "shape_type": "rectangle", "flags": {}, "attributes": {}}


def build(root):
    m = root / '问题电芯图片汇总'
    (m / '1-暗斑').mkdir(parents=True)
    (m / '负样本').mkdir(parents=True)
    (m / '1-暗斑' / '1_0007.jpg').write_bytes(b'MASTER-0007')
    (m / '1-暗斑' / '1_0007.json').write_text(json.dumps(
        {"imagePath": "1_0007.jpg", "shapes": [shape("1", 10, 10, 50, 50),
                                               shape("1", 100, 100, 140, 140)]},
        ensure_ascii=False), encoding='utf-8')
    (m / '负样本' / '9_0001.jpg').write_bytes(b'NEG-0001')
    (m / '负样本' / '9_0002.jpg').write_bytes(b'NEG-0002')
    (m / '重命名对照表.csv').write_text('目录,旧文件名,新文件名', encoding='utf-8')

    st = root / '待微调_V4捡漏' / '暗斑_补标'
    st.mkdir(parents=True)
    # 两张负样本转正（编号必须不同）
    for src in ('9_0001', '9_0002'):
        (st / ('neg_%s.jpg' % src)).write_bytes(('IMG-' + src).encode())
        (st / ('neg_%s.json' % src)).write_text(json.dumps(
            {"imagePath": "neg_%s.jpg" % src, "shapes": [shape("1", 5, 5, 25, 25)]},
            ensure_ascii=False), encoding='utf-8')
    # 同名合并：1 个重复框 + 1 个新框
    (st / '1_0007.jpg').write_bytes(b'MASTER-0007')
    (st / '1_0007.json').write_text(json.dumps(
        {"imagePath": "1_0007.jpg", "shapes": [shape("1", 10, 10, 50, 50),
                                               shape("1", 200, 200, 240, 240)]},
        ensure_ascii=False), encoding='utf-8')
    # 空标注 → 跳过
    (st / '1_9999.jpg').write_bytes(b'EMPTY')
    (st / '1_9999.json').write_text(json.dumps({"imagePath": "1_9999.jpg", "shapes": []}),
                                    encoding='utf-8')


def run(root, *extra):
    cmd = [sys.executable, str(SCRIPT), '--root', str(root), '--batch', 'MTEST'] + list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return r


def main():
    tmp = Path(tempfile.mkdtemp(prefix='merge_stage_'))
    fails = []

    def check(cond, msg):
        if not cond:
            fails.append(msg)
            print('  [FAIL] ' + msg)

    try:
        build(tmp)
        before = snapshot(tmp)
        m = tmp / '问题电芯图片汇总'

        r = run(tmp, '--apply')
        print(r.stdout[-1200:] if r.stdout else '')
        check(r.returncode == 0, 'apply 退出码 %s\n%s' % (r.returncode, r.stderr[-800:]))

        # 1) 两张负样本转正，编号不同
        a, b = m / '1-暗斑' / '1_0008.jpg', m / '1-暗斑' / '1_0009.jpg'
        check(a.exists() and b.exists(),
              '转正编号不是 1_0008/1_0009：%s' % sorted(p.name for p in (m / '1-暗斑').glob('1_00*.jpg')))
        check(a.read_bytes() != b.read_bytes(), '两张转正图内容相同（撞号覆盖）')
        check(not (m / '负样本' / '9_0001.jpg').exists(), '负样本原图没删')
        check((m / '1-暗斑' / '1_0008.json').exists() and (m / '1-暗斑' / '1_0009.json').exists(),
              '转正 json 缺失')

        # 2) 同名合并：2 旧框 + 1 新框
        doc = json.loads((m / '1-暗斑' / '1_0007.json').read_text(encoding='utf-8'))
        check(len(doc['shapes']) == 3, '合并后框数应为 3，实际 %d' % len(doc['shapes']))

        # 3) 空标注跳过
        check(not (m / '1-暗斑' / '1_9999.json').exists(), '空标注被误合并')

        # 4) 对照表追加 2 行
        lines = [l for l in (m / '重命名对照表.csv').read_text(encoding='utf-8').splitlines() if l.strip()]
        check(len(lines) == 3, '重命名对照表应有 3 行（表头+2），实际 %d' % len(lines))

        # 5) 重跑幂等
        r2 = run(tmp, '--apply')
        check('已处理过跳过 3' in r2.stdout, '重跑没有跳过已处理项：%s' % r2.stdout[-300:])

        # 6) 回滚
        r3 = run(tmp, '--apply', '--rollback', 'MTEST')
        check(r3.returncode == 0, 'rollback 退出码 %s\n%s' % (r3.returncode, r3.stderr[-800:]))
        after = snapshot(tmp)
        after = {k: v for k, v in after.items() if not k.startswith('_backup/')}
        diff = [k for k in set(before) | set(after) if before.get(k) != after.get(k)
                and not k.startswith('待微调_V4捡漏/.merged/')]
        check(not diff, '回滚后与执行前不一致：%s' % diff[:8])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        print('\n[FAIL] %d 项' % len(fails))
        for f in fails:
            print('  -', f)
        return 1
    print('\n[PASS] merge_v4_staging 编号/合并/幂等/回滚 全部通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())
