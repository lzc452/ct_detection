# -*- coding: utf-8 -*-
"""按冻结清单创建独立派生数据版本（硬链接母本，不复制、不覆盖、不递归删除）。
用法：runtime/python.exe scripts/materialize_dataset.py --name CT_DATASET_NEW
先加--verify-only只读校验；输出限工程datasets_v*或.cache。已有目录请换版本名。
"""
import os
import sys
import argparse
import json
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, 'core'))
from materialize import build_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='reports/splits/v5_split_manifest.csv')
    parser.add_argument('--name', default='CT_DATASET_V5_GROUPED')
    parser.add_argument('--root', default='datasets_v5')
    parser.add_argument('--splits', default='train,val')
    parser.add_argument('--force', action='store_true', help='兼容旧参数；已有目录仍拒绝，绝不删除')
    parser.add_argument('--verify-only', '--verify', action='store_true', help='只读逐清单校验，不写产物')
    args = parser.parse_args(argv)
    result = build_dataset(args.manifest, args.root, args.name, args.splits,
                           verify_only=args.verify_only, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
