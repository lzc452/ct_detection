# -*- coding: utf-8 -*-
"""
按批次号回滚复核脚本（#4）落过的盘。

exec_v4_review / merge_v4_staging / prelabel 加 --apply 执行时，会把每一个被触及的
文件在 _backup/<批次号>/ 下留一份拷贝，并写 journal.json。本脚本按 journal 逆序还原。

用法：
  # 先看会还原什么（默认 dry-run）
  runtime\\python.exe scripts\\review_rollback.py --batch 20261021_213000
  # 确认后真正回滚
  runtime\\python.exe scripts\\review_rollback.py --batch 20261021_213000 --apply
  # 查看都有哪些批次
  ls _backup
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_safe import DEFAULT_BACKUP_DIR, rollback   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description='按批次号回滚复核脚本的落盘操作')
    ap.add_argument('--batch', required=True, help='要回滚的批次号（_backup/<批次号>/）')
    ap.add_argument('--backup-dir', default=DEFAULT_BACKUP_DIR, help='备份根目录，默认 _backup')
    ap.add_argument('--apply', action='store_true', help='真正还原；不加只预览')
    args = ap.parse_args()
    rollback(ROOT, args.batch, backup_dir=args.backup_dir, apply=args.apply)


if __name__ == '__main__':
    main()
