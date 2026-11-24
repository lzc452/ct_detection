# -*- coding: utf-8 -*-
"""新型号"荣"抽样试跑：V4 模型跑前 2 组正面图，检出图+标注框输出到复核文件夹
用法: python sample_rong.py [组数] [置信度]
"""
import os, sys, json, csv, collections
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

ROOT = r'E:\荣'
OUT = r'D:\CT Detection\复核_荣1123'
MODEL = r'D:\CT Detection\models\启用\20261019_V4\best.pt'
CLS_ID = {'极片损伤': '0', '暗斑': '1', '亮斑': '2'}
CLS_EN = {0: 'damage', 1: 'dark', 2: 'light'}

n_groups = int(sys.argv[1]) if len(sys.argv) > 1 else 2
CONF = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4

def natural_key(s):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

groups = sorted([g for g in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, g))], key=natural_key)
groups = groups[:n_groups]
EXT = ('.jpg', '.jpeg', '.png', '.bmp')

imgs = []
for g in groups:
    fp = os.path.join(ROOT, g, '正面')
    if not os.path.isdir(fp):
        continue
    for f in sorted(os.listdir(fp), key=natural_key):
        if f.lower().endswith(EXT):
            imgs.append((g, os.path.join(fp, f)))
print(f'抽样 {len(groups)} 组: {groups}', flush=True)
print(f'共 {len(imgs)} 张, 置信度阈值 {CONF}, 模型 {os.path.basename(os.path.dirname(MODEL))}', flush=True)

from ultralytics import YOLO
m = YOLO(MODEL)

made = 0
rows = []
stats = collections.Counter()
batch, batch_meta = [], []

def flush():
    global made
    if not batch:
        return
    res = m.predict(batch, imgsz=1920, batch=16, conf=CONF, device=0, half=True, verbose=False)
    for (g, p), r in zip(batch_meta, res):
        if r.boxes is None or len(r.boxes) == 0:
            continue
        f = os.path.basename(p)
        with Image.open(p) as im:
            w, h = im.size
        shapes = []
        for b in r.boxes:
            cid = int(b.cls[0])
            cname = r.names[cid]
            conf = float(b.conf[0])
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            shapes.append({"label": CLS_ID.get(cname, str(cid)), "score": round(conf, 4),
                           "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                           "group_id": None, "description": CLS_EN.get(cid, ''),
                           "difficult": False, "shape_type": "rectangle",
                           "flags": {}, "attributes": {}, "kie_linking": []})
            stats[cname] += 1
            rows.append([f'{g}_{f}', g, cname, f'{conf:.3f}', int(x1), int(y1), int(x2), int(y2)])
        cd = os.path.join(OUT, f"{CLS_ID.get(r.names[int(r.boxes[0].cls[0])], 'x')}-{r.names[int(r.boxes[0].cls[0])]}")
        # 多类别图：放进"多缺陷"
        clsset = {s['description'] for s in shapes}
        if len(clsset) > 1:
            cd = os.path.join(OUT, '多缺陷')
        os.makedirs(cd, exist_ok=True)
        import shutil
        dst = os.path.join(cd, f'{g}_{f}')
        if not os.path.exists(dst):
            shutil.copyfile(p, dst)
        doc = {"version": "4.0.6", "flags": {}, "checked": False, "shapes": shapes,
               "imagePath": f'{g}_{f}', "imageData": None, "imageHeight": h, "imageWidth": w}
        with open(os.path.join(cd, f'{g}_{os.path.splitext(f)[0]}.json'), 'w', encoding='utf-8') as fp:
            json.dump(doc, fp, ensure_ascii=False)
        made += 1
    batch.clear(); batch_meta.clear()

for i, (g, p) in enumerate(imgs):
    batch.append(p); batch_meta.append((g, p))
    if len(batch) >= 32:
        flush()
    if (i + 1) % 320 == 0:
        print(f'  进度 {i+1}/{len(imgs)} 检出图 {made}', flush=True)
flush()

with open(os.path.join(OUT, '抽样清单.csv'), 'w', newline='', encoding='utf-8-sig') as fp:
    w = csv.writer(fp)
    w.writerow(['文件名', '组别', '类别', '置信度', 'Xmin', 'Ymin', 'Xmax', 'Ymax'])
    w.writerows(rows)

print(f'\n完成：检 {len(imgs)} 张 → 问题图 {made} 张')
for k, v in stats.most_common():
    print(f'  {k}: {v} 个')
print('输出:', OUT)
