# -*- coding: utf-8 -*-
"""产线实测练兵战场：批量跑模型，统计检出+导出标注图。支持断点续跑。
用法: python scripts/battlefield.py <model_path> <tag>
"""
import os, sys, glob, csv, time
import numpy as np
import cv2
from ultralytics import YOLO

ROOT = r'D:\CT Detection'
SRC = os.path.join(ROOT, '产线实测练兵战场')
CONF_SAVE = 0.15      # 网页默认阈值：达到才导出标注图
CONF_REC = 0.05       # 记录此阈值以上的检测明细
TIME_BUDGET = 255     # 秒，到点自动退出，重跑续上

def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite_cn(p, img):
    cv2.imencode('.jpg', img)[1].tofile(p)

def main():
    model_path, tag = sys.argv[1], sys.argv[2]
    out_csv = os.path.join(ROOT, 'battlefield_%s.csv' % tag)
    out_dir = os.path.join(ROOT, 'battlefield_标注_%s' % tag)
    os.makedirs(out_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(SRC, '*', '正面', '*.jpg')))
    done = set()
    if os.path.exists(out_csv):
        for row in csv.reader(open(out_csv, encoding='utf-8-sig')):
            if row and row[0] != 'path':
                done.add(row[0])
    todo = [p for p in imgs if p not in done]
    print('总数%d 已完成%d 本次待跑%d' % (len(imgs), len(done), len(todo)))
    if not todo:
        return

    m = YOLO(model_path)
    f = open(out_csv, 'a', newline='', encoding='utf-8-sig')
    w = csv.writer(f)
    if not done:
        w.writerow(['path', 'group', 'name', 'n_det_015', 'dets(类:置信度@>=0.05)'])
    t0 = time.time(); n_done = 0
    B = 16
    for i in range(0, len(todo), B):
        chunk = todo[i:i+B]
        arrs = []
        for p in chunk:
            im = imread_cn(p)
            arrs.append(im if im is not None else np.zeros((16, 16, 3), np.uint8))
        rs = m.predict(arrs, imgsz=1920, conf=CONF_REC, verbose=False)
        for p, im, r in zip(chunk, arrs, rs):
            group = os.path.basename(os.path.dirname(os.path.dirname(p)))
            name = os.path.basename(p)
            dets = [(int(b.cls), float(b.conf), b.xyxy[0].tolist()) for b in r.boxes] if r.boxes is not None else []
            n15 = sum(1 for _, c, _ in dets if c >= CONF_SAVE)
            desc = '; '.join('%s:%.2f' % (m.names[c], cf) for c, cf, _ in dets)
            w.writerow([p, group, name, n15, desc])
            if n15:
                vis = r.plot(img=im, conf=False, line_width=2)
                # 只画 >=0.15 的框：重画
                vis = im.copy()
                for c, cf, xy in dets:
                    if cf < CONF_SAVE: continue
                    x1, y1, x2, y2 = map(int, xy)
                    color = (0, 0, 255) if c == 0 else ((255, 128, 0) if c == 1 else (0, 200, 255))
                    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(vis, '%s %.2f' % (m.names[c], cf), (x1, max(20, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                gd = os.path.join(out_dir, group)
                os.makedirs(gd, exist_ok=True)
                imwrite_cn(os.path.join(gd, name), vis)
        n_done += len(chunk)
        f.flush()
        if n_done % 160 < B:
            print('进度 %d/%d 用时%.0fs' % (n_done, len(todo), time.time() - t0))
        if time.time() - t0 > TIME_BUDGET:
            print('时间到，已续跑保存，本次完成%d' % n_done)
            break
    f.close()
    print('本次完成%d，输出: %s / %s' % (n_done, out_csv, out_dir))

if __name__ == '__main__':
    main()
