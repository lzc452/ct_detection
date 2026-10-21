# -*- coding: utf-8 -*-
"""路径直读后台检测 — 全链路自测（结束后恢复现场，不污染真实结果）"""
import os, sys, json, shutil, subprocess, time, urllib.request, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\CT Detection
RESULTS = os.path.join(ROOT, 'web', 'results')
STATE = os.path.join(RESULTS, '_state.json')
TASKF = os.path.join(RESULTS, '_detect_task.json')
BASE = 'http://127.0.0.1:5001'
TESTDIR = os.path.join(ROOT, '测试样本_自测')

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

# ── 0. 现场保护 ──
state_bak = STATE + '.testbak'
shutil.copyfile(STATE, state_bak) if os.path.exists(STATE) else None
task_bak = TASKF + '.testbak'
had_task = os.path.exists(TASKF)
if had_task: shutil.copyfile(TASKF, task_bak)
before_files = set(os.listdir(RESULTS))

# ── 1. 搭测试目录 ──
EXAM = os.path.join(ROOT, 'datasets', '考卷_清洗版', 'images')
exam_imgs = sorted(f for f in os.listdir(EXAM) if f.lower().endswith('.jpg'))
def cp(dst_dir, names):
    os.makedirs(dst_dir, exist_ok=True)
    for i, n in enumerate(names):
        shutil.copyfile(os.path.join(EXAM, n), os.path.join(dst_dir, f'{i:03d}_{n}'))
try:
    cp(os.path.join(TESTDIR, 'T1', '正面'), exam_imgs[0:3])
    cp(os.path.join(TESTDIR, 'T1', '侧面'), exam_imgs[3:4])   # 应被忽略
    cp(os.path.join(TESTDIR, 'T2', '正面'), exam_imgs[4:6])
    cp(os.path.join(TESTDIR, 'flat'), [])                    # flat 模式单独建
    os.makedirs(os.path.join(TESTDIR, 'flat'), exist_ok=True)
    for i, n in enumerate(exam_imgs[6:8]):
        shutil.copyfile(os.path.join(EXAM, n), os.path.join(TESTDIR, 'flat', f'f{i}_{n}'))
    big = os.path.join(TESTDIR, 'big', '正面')
    os.makedirs(big, exist_ok=True)
    for i in range(40):
        shutil.copyfile(os.path.join(EXAM, exam_imgs[i % 20]), os.path.join(big, f'b{i:03d}.jpg'))

    # ── 2. 起服务 ──
    env = dict(os.environ, PROCESSOR_ARCHITECTURE='AMD64', CT_PORT='5001', CT_NO_BROWSER='1')
    srv = subprocess.Popen([os.path.join(ROOT, 'runtime', 'python.exe'),
                            os.path.join(ROOT, 'web', 'app.py')],
                           cwd=ROOT, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    up = False
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + '/api/models', timeout=1)
            up = True; break
        except Exception:
            if srv.poll() is not None: break
            time.sleep(1)
    check('服务启动', up)
    if not up:
        raise SystemExit('server failed to start')

    # ── 3. 状态接口 ──
    code, st = api('GET', '/api/state')
    check('GET /api/state', code == 200 and st.get('success') and 'detection_results' in st)

    # ── 4. 结构扫描 ──
    code, r = api('POST', '/api/scan_path', {'path': TESTDIR})
    groups = {g['name']: g['count'] for g in r.get('groups', [])}
    check('扫描多组模式', code == 200 and r.get('mode') == 'groups', f"mode={r.get('mode')}")
    check('多组识别T1/T2/big', groups.get('T1') == 3 and groups.get('T2') == 2 and groups.get('big') == 40, str(groups))
    check('侧面被忽略', r.get('total') == 47, f"total={r.get('total')}（45张正面+flat组2张，侧面1张未计入）")

    code, r = api('POST', '/api/scan_path', {'path': os.path.join(TESTDIR, 'flat')})
    check('flat模式', r.get('mode') == 'flat' and r.get('total') == 2, f"{r.get('mode')}/{r.get('total')}")
    code, r = api('POST', '/api/scan_path', {'path': os.path.join(TESTDIR, 'T1')})
    check('view模式(选到T1)', r.get('mode') == 'view' and r.get('total') == 3, f"{r.get('mode')}/{r.get('total')}")
    code, r = api('POST', '/api/scan_path', {'path': os.path.join(TESTDIR, 'T1', '正面')})
    check('flat模式(选到正面)', r.get('mode') == 'flat' and r.get('total') == 3)
    code, r = api('POST', '/api/scan_path', {'path': r'D:\不存在的路径xyz'})
    check('坏路径报错', not r.get('success'))

    # ── 5. 打勾部分组检测（只选 T2，2张）──
    code, r = api('POST', '/api/detect/start', {'path': TESTDIR, 'groups': ['T2'], 'conf': 0.30})
    check('启动部分组检测', code == 200 and r.get('success') and r.get('total') == 2, f"total={r.get('total')}")
    tid = r.get('task_id')
    final = None
    for _ in range(120):
        code, s = api('GET', f'/api/detect_status/{tid}?cursor=0')
        if s.get('status') in ('done', 'error'): final = s; break
        time.sleep(0.5)
    check('T2检测完成', final and final.get('status') == 'done', f"status={final and final.get('status')} msg={final and final.get('message')}")
    check('T2检出缺陷图(考卷图应有缺陷)', final and final.get('defect_imgs') >= 1, f"defect_imgs={final and final.get('defect_imgs')}")
    check('进度字段齐全', final and all(k in final for k in ('speed', 'eta_s', 'items', 'cursor')))
    new_files = set(os.listdir(RESULTS)) - before_files
    check('问题图已备份(原图+标注图)', any(f.startswith('orig_') for f in new_files) and any(f.startswith('result_') for f in new_files))

    # ── 6. 中断 + 续检（40张跑到一半停止）──
    code, r = api('POST', '/api/detect/start', {'path': os.path.join(TESTDIR, 'big'), 'conf': 0.30})
    tid2 = r.get('task_id')
    check('启动big检测', r.get('success') and r.get('total') == 40)
    stopped = None
    for _ in range(200):
        code, s = api('GET', f'/api/detect_status/{tid2}?cursor=0')
        if s.get('done', 0) >= 8:
            api('POST', '/api/detect/stop', {'task_id': tid2})
        if s.get('status') in ('stopped', 'done', 'error'):
            stopped = s; break
        time.sleep(0.3)
    check('中途停止生效', stopped and stopped.get('status') == 'stopped', f"status={stopped and stopped.get('status')}")
    done_at_stop = stopped.get('done', 0) if stopped else 0
    check('停止时部分完成', 0 < done_at_stop < 40, f'done={done_at_stop}')
    code, st = api('GET', '/api/state')
    check('中断任务可见', bool(st.get('interrupted_task')), str(st.get('interrupted_task', {}).get('done')))
    code, r = api('POST', '/api/detect/start', {'path': os.path.join(TESTDIR, 'big'), 'conf': 0.30, 'resume': True})
    check('续检启动并跳过已完成', r.get('success') and r.get('skipped') == done_at_stop, f"skipped={r.get('skipped')} vs {done_at_stop}")
    tid3 = r.get('task_id')
    final3 = None
    for _ in range(200):
        code, s = api('GET', f'/api/detect_status/{tid3}?cursor=0')
        if s.get('status') in ('done', 'error'): final3 = s; break
        time.sleep(0.5)
    check('续检跑完全部40张', final3 and final3.get('status') == 'done' and final3.get('done') == 40, f"done={final3 and final3.get('done')}")

    code, s = api('GET', '/api/detect_status/nonexistent')
    check('未知任务404', code == 404)

finally:
    # ── 7. 恢复现场 ──
    try: srv.terminate(); srv.wait(timeout=10)
    except Exception: srv.kill()
    time.sleep(1)
    shutil.rmtree(TESTDIR, ignore_errors=True)
    for f in set(os.listdir(RESULTS)) - before_files:
        p = os.path.join(RESULTS, f)
        try:
            os.remove(p) if os.path.isfile(p) else shutil.rmtree(p)
        except Exception: pass
    if os.path.exists(state_bak):
        shutil.copyfile(state_bak, STATE); os.remove(state_bak)
    if had_task: shutil.copyfile(task_bak, TASKF); os.remove(task_bak)
    elif os.path.exists(TASKF): os.remove(TASKF)

print(f'\n===== 自测结果: {len(passed)} 通过 / {len(failed)} 失败 =====')
if failed:
    print('失败项:', failed); sys.exit(1)
