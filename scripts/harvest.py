# -*- coding: utf-8 -*-
"""产线收割机：高阈值跑海量产线图，导出高置信损伤原图（供人工核对+X-AnyLabeling标注）。
用法: python scripts/harvest.py <源文件夹> [导出阈值=0.7]
特性: 断点续跑 / 只跑正面文件夹 / CSV记录全部>=0.30检测 / 默认只导出损伤类(EXPORT_CLS可改)
"""
import os, sys, glob, csv, time
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = r'D:\CT Detection'
MODEL = os.path.join(ROOT, 'models', '启用', '20261019_V4', 'best.pt')
MODEL_TAG = 'V4'
ZH = {0: '极片损伤', 1: '暗斑', 2: '亮斑'}
CONF_RECORD = 0.30   # 记录进CSV的最低阈值（方便以后放宽不重跑）
EXPORT_CLS = {0}     # 只导出这些类别的高置信图（0=损伤）；想全类导出改成 {0,1,2}
TIME_BUDGET = float(os.environ.get('HARVEST_BUDGET', '0'))  # 秒；0=不限时（独立窗口跑），Bash短跑时由调用方设定
SKIP_DIRS = [        # 这些文件夹里的图已人工核过，本次不再导出（避免重复劳动）
    os.path.join(ROOT, '收割结果_积攒', '极片损伤'),
    os.path.join(ROOT, '收割结果_积攒', '极片损伤_低置信备查'),
]

def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite_cn(p, img):
    cv2.imencode('.jpg', img)[1].tofile(p)

def main():
    src = sys.argv[1]
    conf_export = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
    tag = os.path.basename(src.rstrip('\\/')) or 'harvest'
    out_csv = os.path.join(ROOT, 'harvest_%s_%s.csv' % (tag, MODEL_TAG))
    out_dir = os.path.join(ROOT, '收割结果_%s_%s' % (tag, MODEL_TAG))

    # 智能检测结构：大文件夹/组(A、B…)/正面/图；只跑"正面"文件夹
    imgs = sorted(glob.glob(os.path.join(src, '*', '正面', '*.jpg')))
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(src, '**', '正面', '*.jpg'), recursive=True))
    if not imgs:
        imgs = sorted(glob.glob(os.path.join(src, '**', '*.jpg'), recursive=True))
    done = set()
    if os.path.exists(out_csv):
        for row in csv.reader(open(out_csv, encoding='utf-8-sig')):
            if row and row[0] != 'path':
                done.add(row[0])
    todo = [p for p in imgs if p not in done]

    # 老图跳过名单：以往收割已人工核过的导出文件名（组_图名.jpg）
    skip = set()
    for sd in SKIP_DIRS:
        if os.path.isdir(sd):
            skip.update(os.listdir(sd))
    print('源: %s\n总图 %d 张, 已完成 %d, 本次待跑 %d (导出阈值 %.2f, 已核老图跳过 %d 张)' % (src, len(imgs), len(done), len(todo), conf_export, len(skip)))
    if not todo:
        print('全部完成！'); return

    m = YOLO(MODEL)
    f = open(out_csv, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    if not done:
        w.writerow(['path', 'name', 'n_export', 'dets(类:置信度@>=0.30)'])
    t0 = time.time(); n_done = 0; n_hit = 0; n_skip = 0
    B = 16
    for i in range(0, len(todo), B):
        chunk = todo[i:i+B]
        arrs = [imread_cn(p) for p in chunk]
        arrs = [a if a is not None else np.zeros((16, 16, 3), np.uint8) for a in arrs]
        rs = m.predict(arrs, imgsz=1920, conf=CONF_RECORD, verbose=False)
        for p, im, r in zip(chunk, arrs, rs):
            name = os.path.basename(p)
            dets = [(int(b.cls), float(b.conf), b.xyxy[0].tolist()) for b in r.boxes] if r.boxes is not None else []
            exps = [d for d in dets if d[1] >= conf_export and d[0] in EXPORT_CLS]
            desc = '; '.join('%s:%.2f' % (ZH[c], cf) for c, cf, _ in dets)
            w.writerow([p, name, len(exps), desc])
            if exps:
                group = os.path.basename(os.path.dirname(os.path.dirname(p)))
                out_name = '%s_%s' % (group, name)
                if out_name in skip:
                    n_skip += 1
                    continue
                # 主类 = 最高置信度；只导原图（人工标注用），文件名带组前缀
                primary = ZH[max(exps, key=lambda t: t[1])[0]]
                gd = os.path.join(out_dir, primary)
                os.makedirs(gd, exist_ok=True)
                imwrite_cn(os.path.join(gd, out_name), im)
                n_hit += 1
        n_done += len(chunk)
        f.flush()
        if n_done % 320 < B:
            print('进度 %d/%d | 已收割 %d 张 | 跳过已核 %d 张 | 用时 %.0fs' % (n_done, len(todo), n_hit, n_skip, time.time()-t0))
        if TIME_BUDGET > 0 and time.time() - t0 > TIME_BUDGET:
            print('到点收工（已保存），再双击续跑。本次 %d 张，收割 %d 张' % (n_done, n_hit)); break
    f.close()

if __name__ == '__main__':
    main()
