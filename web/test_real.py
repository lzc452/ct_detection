# -*- coding: utf-8 -*-
"""用真有缺陷的图复现导出报错"""
import os, sys, time, glob

ROOT = r'D:\CT Detection'
os.chdir(os.path.join(ROOT, 'web'))
sys.path.insert(0, os.path.join(ROOT, 'web'))
import app as webapp

# 找3张有标注的考卷图
val_img = os.path.join(ROOT, 'datasets', 'CT_DATASET_FINAL_V4_AUG', 'val', 'images')
val_lbl = os.path.join(ROOT, 'datasets', 'CT_DATASET_FINAL_V4_AUG', 'val', 'labels')
picks = []
for p in sorted(glob.glob(val_img + '/*.jpg')):
    lb = os.path.join(val_lbl, os.path.splitext(os.path.basename(p))[0] + '.txt')
    if os.path.exists(lb) and os.path.getsize(lb) > 0:
        picks.append(p)
    if len(picks) >= 3:
        break
print('测试图:', [os.path.basename(p) for p in picks])

c = webapp.app.test_client()
files = [(open(p, 'rb'), os.path.basename(p)) for p in picks]
paths = ['考卷/正面/%s' % os.path.basename(p) for p in picks]
import json
r = c.post('/api/detect/batch', data={
    'files': files, 'conf': '0.30', 'file_paths': json.dumps(paths),
}, content_type='multipart/form-data').get_json()
for fp, _ in files:
    fp.close()
print('检测:', r.get('success'))
print('缺陷明细:', [(x['filename'], x['target_nums'], x.get('classes')) for x in r.get('results', [])])
print('问题图登记数:', len(webapp.problem_images))
for it in webapp.problem_images:
    print('  ', it['相对目录'], '|', it['文件名'], '|', it['缺陷'])

r3 = c.post('/api/export').get_json()
print('导出启动:', r3)
if r3.get('success'):
    tid = r3['task_id']
    st = {}
    for _ in range(120):
        st = c.get('/api/export_status/%s' % tid).get_json()
        if st['status'] in ('done', 'error'):
            break
        time.sleep(0.5)
    print('导出结果:', st)
