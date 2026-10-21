# -*- coding: utf-8 -*-
"""为收割结果生成 X-AnyLabeling 预标注 json（人工只需核对修正）。

【安全约定 —— 2026-10-21 改造】
1. 默认 dry-run：只统计"会新建 / 会跳过 / 会覆盖"多少张，写清单 CSV，不动任何文件。
2. 默认**不覆盖**已存在的同名 json（收割目录里可能已经有人工改过的标注）；
   确需覆盖时加 --overwrite，且覆盖前自动备份到 _backup/<批次>/。
3. ROOT 由脚本位置推导（旧版硬编码 D:\\CT Detection，会改到另一套工程树上）。
4. 中文路径统一 imdecode（OpenCV 原生 imread 对中文路径静默失败）。

用法：
  runtime\\python.exe scripts\\prelabel.py                 # 只统计（dry-run）
  runtime\\python.exe scripts\\prelabel.py --apply         # 真正生成（跳过已有 json）
  runtime\\python.exe scripts\\prelabel.py --apply --overwrite   # 覆盖已有 json（先备份）
  runtime\\python.exe scripts\\prelabel.py --src 收割结果_xxx --model models/启用/xxx/best.pt
"""
import csv
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _review_safe import Plan, base_parser, rollback   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NAMES = {0: 'damage', 1: 'dark', 2: 'light'}
DEFAULT_SRC = '收割结果_积攒_V4'
DEFAULT_MODEL = 'models/启用/20261019_V4/best.pt'
CONF = 0.40
BATCH = 16


def imread_cn(p):
    import cv2
    import numpy as np
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


def make_json(name, W, H, dets):
    shapes = []
    for c, cf, (x1, y1, x2, y2) in dets:
        shapes.append({
            "label": str(c), "score": None,
            "points": [[float(x1), float(y1)], [float(x2), float(y1)],
                       [float(x2), float(y2)], [float(x1), float(y2)]],
            "group_id": None, "description": NAMES[c], "difficult": False,
            "shape_type": "rectangle", "flags": {}, "attributes": {}, "kie_linking": []
        })
    return {"version": "4.0.6", "flags": {}, "checked": False, "shapes": shapes,
            "imagePath": name, "imageData": None, "imageHeight": H, "imageWidth": W}


def main():
    ap = base_parser("收割结果 → X-AnyLabeling 预标注 json（默认 dry-run）",
                     epilog="示例：\n"
                            "  runtime\\python.exe scripts\\prelabel.py            # 只统计\n"
                            "  runtime\\python.exe scripts\\prelabel.py --apply    # 生成")
    ap.add_argument('--src', default=DEFAULT_SRC, help='收割目录（相对工程根或绝对路径）')
    ap.add_argument('--model', default=DEFAULT_MODEL, help='预标注用模型')
    ap.add_argument('--overwrite', action='store_true', help='覆盖已存在的 json（先备份）')
    ap.add_argument('--conf', type=float, default=CONF, help='置信度阈值，默认 0.40')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 张（调试用）')
    args = ap.parse_args()

    if args.rollback:
        rollback(ROOT, args.rollback, backup_dir=args.backup_dir, apply=args.apply)
        return

    src = Path(args.src) if os.path.isabs(args.src) else ROOT / args.src
    if not src.is_dir():
        raise SystemExit('[中止] 收割目录不存在：%s' % src)
    imgs = sorted(glob.glob(os.path.join(str(src), '*', '*.jpg')))
    if not imgs:
        raise SystemExit('[中止] 目录里没有 *.jpg：%s' % src)

    plan = Plan(args, label='prelabel', root=ROOT)
    new, skip, overwrite = [], [], []
    for p in imgs:
        jp = os.path.splitext(p)[0] + '.json'
        if not os.path.exists(jp):
            new.append(p)
        elif args.overwrite:
            overwrite.append(p)
        else:
            skip.append(p)
    if args.limit:
        new = new[:args.limit]
        overwrite = overwrite[:args.limit]

    print('[根目录] %s' % ROOT)
    print('[来源] %s' % src)
    print('[模型] %s' % (Path(args.model) if os.path.isabs(args.model) else ROOT / args.model))
    print('[批次] %s   共 %d 张：新建 %d / 跳过(已有json) %d / 覆盖 %d'
          % (plan.batch, len(imgs), len(new), len(skip), len(overwrite)))
    for p in (new + overwrite)[:20]:
        print('  + %s' % p)
    if len(new) + len(overwrite) > 20:
        print('  ... 其余 %d 张见清单 CSV' % (len(new) + len(overwrite) - 20))

    plan_dir = plan.backup_root / 'plans'
    plan_dir.mkdir(parents=True, exist_ok=True)
    plan_csv = plan_dir / ('plan_prelabel_%s.csv' % plan.batch)
    with open(plan_csv, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['图片', '动作', '说明'])
        for p in new:
            w.writerow([p, '新建', '生成预标注 json'])
        for p in overwrite:
            w.writerow([p, '覆盖', '已存在 json，先备份再覆盖'])
        for p in skip:
            w.writerow([p, '跳过', '已存在 json（加 --overwrite 才覆盖）'])
    print('[清单] %s' % plan_csv)

    if not args.apply:
        print('[dry-run] 未生成任何 json。确认无误后加 --apply；覆盖已有标注还需 --overwrite。')
        return
    if skip:
        print('[跳过] %d 张已有 json，保持原样（不覆盖人工标注）' % len(skip))
    todo = new + overwrite
    if not todo:
        print('[完成] 没有需要生成的 json。')
        return

    model_path = Path(args.model) if os.path.isabs(args.model) else ROOT / args.model
    if not model_path.is_file():
        raise SystemExit('[中止] 模型不存在：%s（模型不入库，请先人工确认后放到 models/启用/）'
                         % model_path)
    from ultralytics import YOLO
    m = YOLO(str(model_path))
    n = 0
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        arrs = [imread_cn(p) for p in chunk]
        import numpy as np
        rs = m.predict([a if a is not None else np.zeros((16, 16, 3), np.uint8) for a in arrs],
                       imgsz=1920, conf=args.conf, verbose=False)
        for p, im, r in zip(chunk, arrs, rs):
            if im is None:
                print('[跳过] 读图失败：%s' % p)
                continue
            H, W = im.shape[:2]
            dets = ([(int(b.cls), float(b.conf), b.xyxy[0].tolist()) for b in r.boxes]
                    if r.boxes is not None else [])
            jp = Path(os.path.splitext(p)[0] + '.json')
            plan.write_json(jp, make_json(os.path.basename(p), W, H, dets),
                            '预标注 %s（%d 框）' % (jp.name, len(dets)))
            n += 1
        print('[进度] %d/%d' % (min(i + BATCH, len(todo)), len(todo)))

    plan.commit()
    print('[完成] %d 个 json 已生成（与图片同目录同名）' % n)


if __name__ == '__main__':
    main()
