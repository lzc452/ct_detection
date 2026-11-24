# -*- coding: utf-8 -*-
"""荣（新型号）全量收割：V4 跑全部正面图，CSV 记录全部检测；
跑完后按"每组高置信 Top-K"精选导出（图+标注框），最大限度压复核量。
用法: python scripts/harvest_rong.py            # 第一阶段：全量推理（断点续跑）
      python scripts/harvest_rong.py --export   # 第二阶段：按CSV精选导出（不重跑推理）
"""
import os, sys, glob, csv, time, json, collections, heapq
import numpy as np
import cv2

ROOT = r'D:\CT Detection'
SRC = r'E:\荣'
MODEL = os.path.join(ROOT, 'models', '启用', '20261019_V4', 'best.pt')
ZH = {0: '极片损伤', 1: '暗斑', 2: '亮斑'}
EN = {0: 'damage', 1: 'dark', 2: 'light'}
CONF_RECORD = 0.30      # 进CSV的最低置信度
OUT_CSV = os.path.join(ROOT, 'harvest_荣_V4.csv')
OUT_DIR = os.path.join(ROOT, '收割结果_荣1123_V4')
EXPORT_MIN_CONF = 0.50  # 入选导出的最低置信度
TOP_K_PER_GROUP = 12    # 每组最多导出张数（按最高置信度排序）

def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def natural_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def group_of(p):
    # E:\荣\1-10\正面\xxx.jpg → 1-10
    return os.path.basename(os.path.dirname(os.path.dirname(p)))

# ─────────── 第一阶段：全量推理 ───────────
def infer():
    imgs = sorted(glob.glob(os.path.join(SRC, '*', '正面', '*.jpg')), key=natural_key)
    print('源: %s | 正面图共 %d 张' % (SRC, len(imgs)))
    done = set()
    if os.path.exists(OUT_CSV):
        for row in csv.reader(open(OUT_CSV, encoding='utf-8-sig')):
            if row and row[0] != 'path':
                done.add(row[0])
    todo = [p for p in imgs if p not in done]
    print('已完成 %d，本次待跑 %d（断点续跑，中断后双击同一bat即可接着跑）' % (len(done), len(todo)))
    if not todo:
        print('全部跑完！再运行 export 阶段即可'); return
    from ultralytics import YOLO
    m = YOLO(MODEL)
    f = open(OUT_CSV, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    if not done:
        w.writerow(['path', 'name', 'group', 'dets(类|置信度|x1|y1|x2|y2 ; ...)'])
    t0 = time.time(); n_done = 0; n_det = 0
    B = 32
    for i in range(0, len(todo), B):
        chunk = todo[i:i+B]
        arrs = [imread_cn(p) for p in chunk]
        arrs = [a if a is not None else np.zeros((16, 16, 3), np.uint8) for a in arrs]
        rs = m.predict(arrs, imgsz=1920, batch=32, conf=CONF_RECORD, device=0, half=True, verbose=False)
        for p, r in zip(chunk, rs):
            dets = []
            if r.boxes is not None:
                for b in r.boxes:
                    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                    dets.append('%d|%.3f|%d|%d|%d|%d' % (int(b.cls[0]), float(b.conf[0]), x1, y1, x2, y2))
            if dets:
                n_det += 1
            w.writerow([p, os.path.basename(p), group_of(p), '; '.join(dets)])
        n_done += len(chunk)
        f.flush()
        if n_done % 320 < B:
            spd = n_done / max(1, time.time() - t0)
            eta = (len(todo) - n_done) / max(0.01, spd) / 60
            print('进度 %d/%d | 有检出 %d 张 | %.1f 张/秒 | 预计剩余 %.0f 分钟' % (n_done, len(todo), n_det, spd, eta), flush=True)
    f.close()
    print('第一阶段完成！总跑 %d 张。双击同一bat或加 --export 进入精选导出。' % n_done)

# ─────────── 第二阶段：每组 Top-K 精选导出 ───────────
def export():
    if not os.path.exists(OUT_CSV):
        print('找不到 %s，先跑第一阶段' % OUT_CSV); return
    # 每组保留置信度最高的 K 张图
    heaps = collections.defaultdict(list)   # group -> heap of (maxconf, idx, path, dets_str)
    n_rows = 0
    for row in csv.reader(open(OUT_CSV, encoding='utf-8-sig')):
        if not row or row[0] == 'path' or len(row) < 4 or not row[3]:
            continue
        n_rows += 1
        best = 0.0
        for tok in row[3].split('; '):
            try:
                best = max(best, float(tok.split('|')[1]))
            except (IndexError, ValueError):
                pass
        if best < EXPORT_MIN_CONF:
            continue
        h = heaps[row[2]]
        item = (best, n_rows, row[0], row[3])
        if len(h) < TOP_K_PER_GROUP:
            heapq.heappush(h, item)
        elif best > h[0][0]:
            heapq.heapreplace(h, item)
    picked = []
    for g, h in heaps.items():
        for conf, _, p, dets in sorted(h, key=lambda t: -t[0]):
            picked.append((g, conf, p, dets))
    print('CSV 检出 %d 行，精选导出 %d 张（%d 个组，每组≤%d，置信度≥%.2f）' % (n_rows, len(picked), len(heaps), TOP_K_PER_GROUP, EXPORT_MIN_CONF))

    n_ok = 0
    stats = collections.Counter()
    w = csv.writer(open(os.path.join(ROOT, '收割清单_荣.csv'), 'w', newline='', encoding='utf-8-sig'))
    w.writerow(['导出文件名', '组别', '主类别', '最高置信度', '原始路径'])
    for g, conf, p, dets_str in picked:
        if not os.path.exists(p):
            continue
        shapes = []
        for tok in dets_str.split('; '):
            parts = tok.split('|')
            if len(parts) != 6:
                continue
            cid = int(parts[0]); cf = float(parts[1])
            x1, y1, x2, y2 = [float(v) for v in parts[2:]]
            shapes.append({"label": str(cid), "score": round(cf, 4),
                           "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           "group_id": None, "description": EN.get(cid, ''),
                           "difficult": False, "shape_type": "rectangle",
                           "flags": {}, "attributes": {}, "kie_linking": []})
            stats[ZH.get(cid, str(cid))] += 1
        if not shapes:
            continue
        primary = ZH.get(int(max(shapes, key=lambda s: s['score'])['label']), 'x')
        name = '%s_%s' % (g, os.path.basename(p))
        gd = os.path.join(OUT_DIR, primary)
        os.makedirs(gd, exist_ok=True)
        img = imread_cn(p)
        if img is None:
            continue
        hgt, wid = img.shape[:2]
        cv2.imencode('.jpg', img)[1].tofile(os.path.join(gd, name))
        doc = {"version": "4.0.6", "flags": {}, "checked": False, "shapes": shapes,
               "imagePath": name, "imageData": None, "imageHeight": hgt, "imageWidth": wid}
        with open(os.path.join(gd, os.path.splitext(name)[0] + '.json'), 'w', encoding='utf-8') as fp:
            json.dump(doc, fp, ensure_ascii=False)
        w.writerow([name, g, primary, '%.3f' % conf, p])
        n_ok += 1
        if n_ok % 200 == 0:
            print('  导出 %d/%d' % (n_ok, len(picked)), flush=True)
    print('导出完成 %d 张 → %s' % (n_ok, OUT_DIR))
    for k, v in stats.most_common():
        print('  %s: %d 个框' % (k, v))

if __name__ == '__main__':
    if '--export' in sys.argv:
        export()
    else:
        infer()
