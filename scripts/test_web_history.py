# -*- coding: utf-8 -*-
"""历史结果/批次/逐条删除 — 自测（结束后恢复现场）"""
import os, sys, json, shutil, subprocess, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'web', 'results')
STATE = os.path.join(RESULTS, '_state.json')
TASKF = os.path.join(RESULTS, '_detect_task.json')
BASE = 'http://127.0.0.1:5001'
TESTDIR = os.path.join(ROOT, '测试样本_自测2')

passed, failed = [], []
def check(name, cond, extra=''):
    (passed if cond else failed).append(name)
    print(('  [PASS] ' if cond else '  [FAIL] ') + name + (f' | {extra}' if extra else ''))

def api(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, method=method,
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

state_bak = STATE + '.testbak2'
if os.path.exists(STATE): shutil.copyfile(STATE, state_bak)
task_bak = TASKF + '.testbak2'
had_task = os.path.exists(TASKF)
if had_task: shutil.copyfile(TASKF, task_bak)
before_files = set(os.listdir(RESULTS))
before_rows = 0
try:
    before_rows = len(json.load(open(STATE, encoding='utf-8')).get('detection_results', []))
except Exception: pass

# 注入合成的"旧格式"数据（无批次ID、两个相隔30分钟的时间簇）验证迁移
legacy_state = {
    'detection_results': [
        {"序号": 1, "组别": "T1", "文件路径": "实战A/正面/a.jpg", "文件全路径": "", "类别ID": 1,
         "类别名称": "暗斑", "置信度(数值)": 0.9, "置信度(百分比)": "90.00%",
         "X最小值": 1, "Y最小值": 2, "X最大值": 3, "Y最大值": 4, "检测时间": "2026-10-20 14:00:00"},
        {"序号": 2, "组别": "T1", "文件路径": "实战A/正面/b.jpg", "文件全路径": "", "类别ID": 2,
         "类别名称": "亮斑", "置信度(数值)": 0.8, "置信度(百分比)": "80.00%",
         "X最小值": 5, "Y最小值": 6, "X最大值": 7, "Y最大值": 8, "检测时间": "2026-10-20 14:05:00"},
        {"序号": 3, "组别": "T2", "文件路径": "实战B/正面/c.jpg", "文件全路径": "", "类别ID": 0,
         "类别名称": "极片损伤", "置信度(数值)": 0.7, "置信度(百分比)": "70.00%",
         "X最小值": 9, "Y最小值": 10, "X最大值": 11, "Y最大值": 12, "检测时间": "2026-10-20 14:40:00"},
    ],
    'problem_images': [
        {"文件名": "a.jpg", "相对目录": "实战A/正面", "组别": "T1", "缺陷": "暗斑", "原图": None, "标注图": None},
        {"文件名": "c.jpg", "相对目录": "实战B/正面", "组别": "T2", "缺陷": "极片损伤", "原图": None, "标注图": None},
    ],
}
with open(STATE, 'w', encoding='utf-8') as f:
    json.dump(legacy_state, f, ensure_ascii=False)
before_rows = 3

srv = None
try:
    EXAM = os.path.join(ROOT, 'datasets', '考卷_清洗版', 'images')
    exam_imgs = sorted(f for f in os.listdir(EXAM) if f.lower().endswith('.jpg'))
    t1 = os.path.join(TESTDIR, 'T1', '正面')
    os.makedirs(t1, exist_ok=True)
    for i, n in enumerate(exam_imgs[0:3]):
        shutil.copyfile(os.path.join(EXAM, n), os.path.join(t1, f'{i:02d}_{n}'))

    env = dict(os.environ, PROCESSOR_ARCHITECTURE='AMD64', CT_PORT='5001', CT_NO_BROWSER='1')
    srv = subprocess.Popen([os.path.join(ROOT, 'runtime', 'python.exe'),
                            os.path.join(ROOT, 'web', 'app.py')],
                           cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + '/api/models', timeout=1); break
        except Exception:
            if srv.poll() is not None: raise SystemExit('server died')
            time.sleep(1)

    # 1. 旧数据迁移：合成旧数据应按10分钟间隔分成 L01/L02 两批
    code, st = api('GET', '/api/state')
    bids = [r.get('批次ID') for r in st['detection_results']]
    check('旧数据批次迁移', bids == ['L01', 'L01', 'L02'], f'实际 {bids}')
    pim_bids = {p['文件名']: p.get('批次ID') for p in st['problem_images']}
    check('旧问题图批次匹配', pim_bids == {'a.jpg': 'L01', 'c.jpg': 'L02'}, str(pim_bids))

    # 2. 跑一次检测 → 新批次
    code, r = api('POST', '/api/detect/start', {'path': TESTDIR, 'conf': 0.30})
    tid = r.get('task_id')
    final = None
    for _ in range(120):
        code, s = api('GET', f'/api/detect_status/{tid}?cursor=0')
        if s.get('status') in ('done', 'error'): final = s; break
        time.sleep(0.5)
    check('检测完成', final and final.get('status') == 'done')

    code, st2 = api('GET', '/api/state')
    new_rows = [r for r in st2['detection_results'] if not r.get('批次ID', '').startswith('L')]
    new_bids = set(r['批次ID'] for r in new_rows)
    check('新检测行带批次ID', len(new_rows) > 0 and len(new_bids) == 1, f"{len(new_rows)}行 bid={new_bids}")
    new_pims = [p for p in st2['problem_images'] if p.get('批次ID') in new_bids]
    check('新问题图带同批次ID', len(new_pims) >= 1, f'{len(new_pims)}张')
    bid = list(new_bids)[0]

    # 2.1 批次元信息：路径/分组/模式 + orig_url
    bi = (st2.get('batch_info') or {}).get(bid)
    check('批次元信息记录', bool(bi) and bi.get('root', '').endswith('测试样本_自测2') and bi.get('groups') == ['T1'],
          str(bi))
    check('问题图有原图备份字段', all(p.get('原图') for p in new_pims))
    img_rel = new_pims[0]['相对目录'] + '/' + new_pims[0]['文件名'] if new_pims[0]['相对目录'] else new_pims[0]['文件名']
    ann = new_pims[0]['标注图']

    # 3. 批次范围导出/下载
    code, e1 = api('POST', '/api/export', {'batch_id': bid})
    check('按批次导出报表', code == 200 and e1.get('success') and bid and '检测报表_' in e1.get('download_url', ''),
          e1.get('download_url', ''))
    code, e2 = api('POST', '/api/download-all', {'batch_id': bid})
    check('按批次打包启动', e2.get('success'))
    et = e2.get('task_id')
    efin = None
    for _ in range(120):
        code, es = api('GET', f'/api/export_status/{et}')
        if es.get('status') in ('done', 'error'): efin = es; break
        time.sleep(0.3)
    check('按批次打包完成且数量正确', efin and efin.get('status') == 'done' and efin.get('total') == len(new_pims),
          f"total={efin and efin.get('total')} vs {len(new_pims)}")
    code, e3 = api('POST', '/api/download-all', {'batch_id': '不存在的批次'})
    check('空批次打包报错', not e3.get('success'))

    # 4. 逐条删除
    code, d = api('POST', '/api/state/delete_image', {'batch_id': bid, 'filename': img_rel})
    check('逐条删除接口', code == 200 and d.get('success') and d.get('removed_rows') >= 1,
          f"rows={d.get('removed_rows')} files={d.get('removed_files')}")
    check('备份文件已删', not os.path.exists(os.path.join(RESULTS, ann)))
    code, st3 = api('GET', '/api/state')
    check('删除后行数减少', len(st3['detection_results']) == before_rows + len(new_rows) - d['removed_rows'])

    # 5. 整批删除
    code, d2 = api('POST', '/api/state/delete_batch', {'batch_id': bid})
    check('整批删除接口', code == 200 and d2.get('success'))
    code, st4 = api('GET', '/api/state')
    remain = [r for r in st4['detection_results'] if r.get('批次ID') == bid]
    remain_pim = [p for p in st4['problem_images'] if p.get('批次ID') == bid]
    check('该批已清空', len(remain) == 0 and len(remain_pim) == 0)
    check('批次元信息同步删除', bid not in (st4.get('batch_info') or {}))
    check('旧数据未受影响', len(st4['detection_results']) == before_rows, f"{len(st4['detection_results'])} vs {before_rows}")
    code, d3 = api('POST', '/api/state/delete_batch', {})
    check('缺参数报错', code == 400)

finally:
    if srv:
        try: srv.terminate(); srv.wait(timeout=10)
        except Exception: srv.kill()
    time.sleep(1)
    shutil.rmtree(TESTDIR, ignore_errors=True)
    for f in set(os.listdir(RESULTS)) - before_files:
        p = os.path.join(RESULTS, f)
        try: os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)
        except Exception: pass
    if os.path.exists(state_bak):
        shutil.copyfile(state_bak, STATE); os.remove(state_bak)
    if had_task: shutil.copyfile(task_bak, TASKF); os.remove(task_bak)
    elif os.path.exists(TASKF): os.remove(TASKF)

print(f'\n===== 自测: {len(passed)} 通过 / {len(failed)} 失败 =====')
if failed:
    print('失败项:', failed); sys.exit(1)
