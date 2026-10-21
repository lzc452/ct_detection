# -*- coding: utf-8 -*-
"""
CT 缺陷检测 — Web 后端（文件上传 + 后端 batch 优化版）
改动要点：
1. 单张/批量/视频均走文件上传（FormData），前端体验不变。
2. 单张：内存中直接 PIL Image → 推理，不保存原始文件。
3. 批量：前端选文件夹一次性/分 chunk 发送 files，后端内存中 batch 推理。
4. 视频：保存到 results/ 临时文件（OpenCV 需要路径），处理后可删除。
5. 推理参数统一优化：imgsz=640, half=False(FP32), max_det=300, verbose=False。
6. 移除 uploads 目录，所有文件只保存在 results/。
"""
import os
# 修复：部分 Windows 环境缺少 PROCESSOR_ARCHITECTURE 变量，
# 会导致 ultralytics 依赖的 py-cpuinfo 直接崩溃（误认为非 x86 CPU）
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
import sys
import time
import uuid
import threading
import datetime
import cv2
import numpy as np
import shutil
import glob
import zipfile
from io import BytesIO
import json
import traceback

from flask import Flask, render_template, request, jsonify, send_file
import torch

# 将项目根目录和 core 目录加入路径，以便复用 core/Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(BASE_DIR, '..')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core'))

import Config
from device_utils import pick_device
from ultralytics import YOLO
from PIL import Image, ImageFont
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

# 切换到项目根目录，确保 Config 中的相对路径和字体路径正确
os.chdir(PROJECT_ROOT)

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024  # 10GB，支持大批量上传

RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ==================== 设备与推理参数 ====================
# 自动适配：有可用显卡用 GPU，无显卡/显卡太旧自动退回 CPU
DEVICE = pick_device(verbose=True)
# 2026-10-21 实测：FP16 与 FP32 检出框零偏差（考卷对比），推理 41→20ms/张；无显卡的电脑自动退回 FP32
USE_FP16 = bool(torch.cuda.is_available()) and str(DEVICE).startswith('cuda')
INFER_SIZE = 1920   # ★必须与训练分辨率一致：1920训练的模型在640下几乎检不出缺陷（2026-10-14 实测）
BATCH_SIZE = 16     # 2026-10-21 实测：5060 FP16 batch16 峰值显存仅 1.66G，20ms/张
MAX_DET = 300
SMART_VIEW = '正面'  # 智能检测只测这个视角
CONF_DEFAULT = 0.30   # 产线实测后上调（2026-10-15）：v2mild上岗，用户要求更严格工作点压低误报
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

print(f"[Device] {DEVICE} | FP16: {USE_FP16} | imgsz: {INFER_SIZE} | batch: {BATCH_SIZE}")

# ==================== 模型扫描与注册 ====================
# 网页可选模型只来自 models/启用/ —— 训练出的模型不会自动出现在这里，
# 必须由人工确认效果后，手动把 .pt 文件复制进来才会出现在网页模型列表。
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models', '启用')
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_REGISTRY = {}

def scan_models():
    """扫描 models/启用/ 目录下所有 .pt 文件"""
    global MODEL_REGISTRY
    MODEL_REGISTRY = {}
    pt_files = []
    for root, dirs, files in os.walk(MODEL_DIR):
        for f in files:
            if f.endswith('.pt'):
                pt_files.append(os.path.join(root, f))
    for pt_path in pt_files:
        rel_path = os.path.relpath(pt_path, PROJECT_ROOT)
        model_id = rel_path.replace(os.sep, '/').replace('.pt', '')
        basename = os.path.basename(pt_path).replace('.pt', '')
        # 显示名：相对于 models/启用/ 的路径（如 "旧版/best"），子文件夹用于区分同名模型
        rel_to_dir = os.path.relpath(pt_path, MODEL_DIR).replace(os.sep, '/').replace('.pt', '')
        MODEL_REGISTRY[model_id] = {
            'id': model_id,
            'path': pt_path,
            'rel_path': rel_path.replace(os.sep, '/'),
            'name': basename,
            'label': rel_to_dir,
            'names': Config.names,
            'CH_names': Config.CH_names,
        }
    print(f"[Model Scan] Found {len(MODEL_REGISTRY)} models:")
    for mid, info in MODEL_REGISTRY.items():
        print(f"  - {mid}: {info['name']}")

scan_models()

# ==================== 模型加载管理 ====================
current_model = None
current_model_id = None

def load_model(model_id, warmup=True):
    """
    加载指定模型，自动卸载旧模型释放显存
    """
    global current_model, current_model_id
    if model_id not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_id}' not found in registry")
    if current_model_id == model_id and current_model is not None:
        print(f"[Model] Already loaded: {model_id}")
        return current_model
    # 卸载旧模型
    if current_model is not None:
        print(f"[Model] Unloading old model: {current_model_id}")
        del current_model
        current_model = None
        current_model_id = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # 加载新模型
    model_info = MODEL_REGISTRY[model_id]
    model_path = model_info['path']
    print(f"[Model] Loading: {model_id} ({model_path})")
    new_model = YOLO(model_path, task='detect')
    current_model = new_model
    current_model_id = model_id
    # 预热
    if warmup:
        _warmup_imgs = [np.zeros((INFER_SIZE, INFER_SIZE, 3), dtype=np.uint8)] * BATCH_SIZE
        new_model(_warmup_imgs, imgsz=INFER_SIZE, half=USE_FP16, device=DEVICE, verbose=False)
        print(f"[Model] Warmup done for {model_id}")
    return new_model

# 加载默认模型：优先 V4（2026-10-19 清洗考卷 mAP50 93.3%，生产口径三类召回 95.8/96.6/90.3，全面优于 V3）
if MODEL_REGISTRY:
    _preferred = 'models/启用/20261019_V4/best'
    _first_id = _preferred if _preferred in MODEL_REGISTRY else list(MODEL_REGISTRY.keys())[0]
    load_model(_first_id)
else:
    raise FileNotFoundError(
        f"models/启用/ 里没有 .pt 模型文件。请把训练好并确认可用的模型手动复制到该文件夹后再启动。")

# ==================== 全局状态 ====================
detection_results = []
video_tasks = {}
problem_images = []   # 有检出的图：{文件名, 组别, 原图文件, 标注图文件}，供导出打包
export_tasks = {}     # 导出任务进度：{task_id: {status, done, total, url, message}}
batch_info = {}       # 批次元信息：{批次ID: {root, groups, mode, time}}，历史结果展示用

STATE_FILE = os.path.join(RESULTS_FOLDER, '_state.json')


def new_batch_id():
    """一次检测 = 一个批次（时间戳+短随机，按检测时间组织历史结果）"""
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:4]


def save_state():
    """检测结果落盘：刷新页面/重启服务后可恢复（致命数据不再丢）"""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({'detection_results': detection_results,
                       'problem_images': problem_images,
                       'batch_info': batch_info}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] save_state failed: {e}")


def _migrate_legacy_batches():
    """旧数据没有批次ID：按检测时间间隔>10分钟归为一批；问题图按文件路径匹配批次"""
    if not detection_results or any('批次ID' in r for r in detection_results):
        return
    def _t(r):
        try:
            return datetime.datetime.strptime(r.get('检测时间', ''), "%Y-%m-%d %H:%M:%S")
        except Exception:
            return datetime.datetime.min
    batch_no = 0
    last_t = None
    rel2batch = {}
    for r in detection_results:   # 行本身按时间顺序追加，顺序扫描分批
        t = _t(r)
        if last_t is None or abs((t - last_t).total_seconds()) > 600:
            batch_no += 1
        r['批次ID'] = f'L{batch_no:02d}'
        last_t = t
        rel = (r.get('文件路径') or '').replace('\\', '/')
        rel2batch.setdefault(rel, r['批次ID'])
    for p in problem_images:
        rel = ((p.get('相对目录') + '/') if p.get('相对目录') else '') + p.get('文件名', '')
        p['批次ID'] = rel2batch.get(rel, 'L01')
    print(f"[State] 旧数据批次迁移完成：{batch_no} 批")


def load_state():
    """启动时恢复上次检测结果"""
    global detection_results
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, encoding='utf-8') as f:
                st = json.load(f)
            detection_results = st.get('detection_results', [])
            problem_images.extend(st.get('problem_images', []))
            batch_info.update(st.get('batch_info', {}))
            _migrate_legacy_batches()
            if detection_results:
                print(f"[State] 恢复上次检测结果 {len(detection_results)} 条 / 问题图 {len(problem_images)} 张")
    except Exception as e:
        print(f"[WARN] load_state failed: {e}")


load_state()


def system_version():
    """系统版本号 = 启用库里最新模型的版本：最高 Vn → V1.n.0（与当前选中哪个模型无关）"""
    import re
    vs = [int(m.group(1)) for k in MODEL_REGISTRY for m in [re.search(r'[Vv](\d+)', k)] if m]
    return 'V1.%d.0' % max(vs) if vs else 'V1.0.0'


def save_annotated_image(img, filename):
    """保存结果图到 results/，确保文件名安全（提取纯文件名，不含路径）"""
    ext = os.path.splitext(os.path.basename(filename))[1].lower() or '.jpg'
    result_filename = f"result_{uuid.uuid4().hex}{ext}"
    result_path = os.path.join(RESULTS_FOLDER, result_filename)
    try:
        cv2.imwrite(result_path, img)
        return result_filename
    except Exception as e:
        print(f"[Save Error] Failed to save {result_path}: {e}")
        traceback.print_exc()
        return None


def save_original_image(rgb_img):
    """有检出时保存原图（RGB numpy，来自上传流），供导出打包用"""
    orig_filename = f"orig_{uuid.uuid4().hex}.jpg"
    orig_path = os.path.join(RESULTS_FOLDER, orig_filename)
    try:
        Image.fromarray(rgb_img).save(orig_path, 'JPEG', quality=95)
        return orig_filename
    except Exception as e:
        print(f"[Save Error] Failed to save original {orig_path}: {e}")
        return None


def register_problem_image(filename, orig_file, annotated_file, cls_ids, batch_id=''):
    """记录一张问题图（原图+标注图+缺陷类别+相对路径+批次），供导出打包"""
    names = []
    for c in sorted(set(cls_ids)):
        n = Config.CH_names[c] if c < len(Config.CH_names) else '未知'
        if n not in names:
            names.append(n)
    rel = filename.replace('\\', '/').strip('/')
    problem_images.append({
        "文件名": os.path.basename(rel),
        "相对目录": os.path.dirname(rel),
        "组别": extract_group(filename),
        "缺陷": '&'.join(names),
        "原图": orig_file,
        "标注图": annotated_file,
        "批次ID": batch_id,
    })


def result_guolv_yolov8(result, conf_threshold):
    """
    YOLOv8 结果过滤 — 与 MainProgram.py 的 result_guolv 完全一致
    直接操作 result.boxes.conf 张量，而非 YOLOv5 的 DataFrame 索引
    """
    try:
        if result is None or result.boxes is None:
            return result
        boxes = result.boxes
        conf_mask = boxes.conf >= conf_threshold
        filtered_boxes = boxes[conf_mask]
        result.boxes = filtered_boxes
        return result
    except Exception as e:
        print(f"[ERROR] result_guolv_yolov8 failed: {e}")
        traceback.print_exc()
        return result  # 出错时返回原始结果，不崩溃


def extract_group(filepath):
    """从相对路径提取分组（电芯）名：
    '图片合集/A/正面/xxx.jpg' -> 'A'；'图片合集/A/侧面/x.jpg' -> 'A'；
    '某文件夹/xxx.jpg' -> '某文件夹'；'xxx.jpg' -> ''
    """
    try:
        parts = [p for p in filepath.replace('\\', '/').split('/') if p]
        for view in ('正面', '侧面', '上面'):
            if view in parts:
                i = parts.index(view)
                if i > 0:
                    return parts[i - 1]
        if len(parts) > 1:
            return parts[-2]
    except Exception:
        pass
    return ''


def parse_results_to_global(results, original_filename, filepath='', batch_id=''):
    """将结果解析并追加到全局 detection_results（用于后续报表导出）"""
    global detection_results
    try:
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return 0

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        group = extract_group(original_filename)

        count = 0
        for idx in range(len(cls_ids)):
            cls_id = int(cls_ids[idx])
            conf_val = float(confs[idx])
            detection_results.append({
                "序号": len(detection_results) + 1,
                "批次ID": batch_id,
                "组别": group,
                "文件路径": original_filename,
                "文件全路径": filepath,
                "类别ID": cls_id,
                "类别名称": Config.CH_names[cls_id] if cls_id < len(Config.CH_names) else "未知",
                "置信度(数值)": conf_val,
                "置信度(百分比)": f"{conf_val * 100:.2f}%",
                "X最小值": int(xyxy[idx][0]),
                "Y最小值": int(xyxy[idx][1]),
                "X最大值": int(xyxy[idx][2]),
                "Y最大值": int(xyxy[idx][3]),
                "检测时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            count += 1
        return count
    except Exception as e:
        print(f"[ERROR] parse_results_to_global failed: {e}")
        traceback.print_exc()
        return 0


# ==================== 全局错误处理 ====================

@app.errorhandler(Exception)
def handle_exception(e):
    """全局异常捕获：任何未处理的异常都返回 JSON 错误信息"""
    print(f"[GLOBAL ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
    return jsonify({
        "success": False,
        "message": f"服务器内部错误: {type(e).__name__}: {str(e)}"
    }), 500


# ==================== 路由 ====================

@app.route('/')
def login():
    return render_template('login.html', sys_ver=system_version())


@app.route('/main')
def index():
    return render_template('index.html', sys_ver=system_version())


@app.route('/api/models')
def list_models():
    """返回可用模型列表"""
    return jsonify({
        "success": True,
        "current": current_model_id,
        "models": [
            {
                "id": mid,
                "name": info["name"],
                "label": info["label"],
                "path": info["rel_path"],
                "active": mid == current_model_id
            }
            for mid, info in MODEL_REGISTRY.items()
        ]
    })


@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    """切换当前模型"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        model_id = data.get('model_id')
        if not model_id:
            model_id = request.form.get('model_id')
        if not model_id or model_id not in MODEL_REGISTRY:
            return jsonify({"success": False, "message": "模型不存在或ID无效"}), 400
        try:
            load_model(model_id)
            return jsonify({
                "success": True,
                "model_id": model_id,
                "model_name": MODEL_REGISTRY[model_id]["name"]
            })
        except Exception as e:
            print(f"[ERROR] switch_model failed: {e}")
            traceback.print_exc()
            return jsonify({"success": False, "message": f"模型加载失败: {str(e)}"}), 500
    except Exception as e:
        print(f"[ERROR] switch_model error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)}), 500


@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """单图检测（文件上传方式）"""
    try:
        global detection_results
        file = request.files.get('file')
        if not file:
            return jsonify({"success": False, "message": "未收到文件"})

        conf_threshold = float(request.form.get('conf', CONF_DEFAULT))
        batch_id = request.form.get('batch_id') or new_batch_id()
        if batch_id not in batch_info:
            batch_info[batch_id] = {'root': '', 'groups': [], 'mode': 'single',
                                    'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        file_paths = json.loads(request.form.get('file_paths', '[]'))
        filename = file_paths[0] if file_paths else file.filename

        # --- 计时：读取 ---
        t0 = time.time()
        try:
            # PIL 对象直接喂模型有兼容问题（可能静默漏检），统一转 numpy RGB
            img = np.array(Image.open(file.stream).convert('RGB'))
        except Exception as e:
            return jsonify({"success": False, "message": f"无法读取图片: {str(e)}"})
        t_read = time.time() - t0

        # --- 计时：模型推理 ---
        t1 = time.time()
        results = current_model(
            img,
            imgsz=INFER_SIZE,
            half=USE_FP16,
            device=DEVICE,
            max_det=MAX_DET,
            verbose=False,
            conf=0.25  # 低阈值进入NMS，保留所有候选框
        )[0]
        t2 = time.time()
        t_infer = t2 - t1

        # --- 计时：置信度过滤 ---
        t3 = time.time()
        results = result_guolv_yolov8(results, conf_threshold)
        t_filter = time.time() - t3

        has_det = results.boxes is not None and len(results.boxes) > 0

        # 安全调用 parse_results_to_global（避免某些加载场景下 NameError）
        if 'parse_results_to_global' in globals():
            parse_results_to_global(results, filename, '', batch_id)
        else:
            print("[WARNING] parse_results_to_global not found in globals, skipping result storage")

        # --- 计时：画图 ---
        t4 = time.time()
        draw_img = results.plot()
        t_plot = time.time() - t4

        # --- 计时：保存 ---
        t5 = time.time()
        result_filename = save_annotated_image(draw_img, filename)
        t_save = time.time() - t5
        if not result_filename:
            return jsonify({"success": False, "message": "结果图片保存失败，请检查 results 目录权限"})

        # --- 汇总 ---
        boxes = results.boxes
        locations = [list(map(int, e)) for e in boxes.xyxy.tolist()] if has_det else []
        classes = [int(i) for i in boxes.cls.tolist()] if has_det else []
        confs = ['%.2f %%' % (each * 100) for each in boxes.conf.tolist()] if has_det else []

        # 有检出：留存原图并登记类别，供导出打包
        orig_file = None
        if has_det:
            orig_file = save_original_image(img)
            register_problem_image(filename, orig_file, result_filename, classes, batch_id)
            save_state()

        # 统计各类别数量
        cls_count = {}
        for c in classes:
            name = Config.CH_names[c] if c < len(Config.CH_names) else '未知'
            cls_count[name] = cls_count.get(name, 0) + 1
        cls_summary = ','.join(f'{k}x{v}' for k, v in cls_count.items()) if cls_count else '无'

        t_total = time.time() - t0
        print(f"[Image] {filename} | 读取:{t_read:.3f}s | 推理:{t_infer:.3f}s | 过滤:{t_filter:.3f}s | 画图:{t_plot:.3f}s | 保存:{t_save:.3f}s | 总计:{t_total:.3f}s | 缺陷:{len(classes)}个({cls_summary})")

        return jsonify({
            "success": True,
            "time": f"{t_total:.3f}",
            "target_nums": len(classes),
            "locations": locations,
            "classes": classes,
            "confs": confs,
            "image_url": f"/api/results/{result_filename}",
            "orig_url": f"/api/results/{orig_file}" if orig_file else None,
            "batch_id": batch_id,
            "filename": filename
        })
    except Exception as e:
        print(f"[ERROR] detect_image failed: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"检测失败: {type(e).__name__}: {str(e)}"}), 500


@app.route('/api/detect/batch', methods=['POST'])
def detect_batch():
    """批量检测（文件上传方式）"""
    try:
        global detection_results
        files = request.files.getlist('files')
        if not files:
            return jsonify({"success": False, "message": "未收到文件"})

        conf_threshold = float(request.form.get('conf', CONF_DEFAULT))
        batch_id = request.form.get('batch_id') or new_batch_id()
        file_paths = json.loads(request.form.get('file_paths', '[]'))
        if batch_id not in batch_info:
            root0 = ''
            if file_paths:
                parts0 = [p for p in file_paths[0].replace('\\', '/').split('/') if p]
                root0 = parts0[0] if len(parts0) > 1 else ''
            batch_info[batch_id] = {'root': root0, 'groups': [], 'mode': 'upload',
                                    'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        response = []
        t_start = time.time()

        total_files = len(files)
        total_defect_imgs = 0
        total_no_defect_imgs = 0
        total_defects = 0

        for i in range(0, total_files, BATCH_SIZE):
            chunk_files = files[i:i + BATCH_SIZE]
            chunk_imgs = []
            chunk_names = []

            # 读取计时
            t_read0 = time.time()
            for j, f in enumerate(chunk_files):
                try:
                    # PIL 对象直接喂模型有兼容问题（可能静默漏检），统一转 numpy RGB
                    img = np.array(Image.open(f.stream).convert('RGB'))
                    chunk_imgs.append(img)
                    chunk_names.append(file_paths[i + j] if (i + j) < len(file_paths) else f.filename)
                except Exception as e:
                    print(f"[Skip] Cannot read {f.filename}: {e}")
                    continue
            t_read = time.time() - t_read0

            if not chunk_imgs:
                continue

            # 推理计时
            t_infer0 = time.time()
            try:
                results = current_model(
                    chunk_imgs,
                    imgsz=INFER_SIZE,
                    half=USE_FP16,
                    device=DEVICE,
                    max_det=MAX_DET,
                    verbose=False,
                    conf=0.25
                )
            except Exception as e:
                print(f"[ERROR] model() batch inference failed: {e}")
                traceback.print_exc()
                return jsonify({"success": False, "message": f"模型推理失败: {type(e).__name__}: {str(e)}"}), 500
            t_infer = time.time() - t_infer0

            t_filter = 0
            t_plot = 0
            t_save = 0
            chunk_defects = 0

            for img, filename, result in zip(chunk_imgs, chunk_names, results):
                try:
                    t_f0 = time.time()
                    result = result_guolv_yolov8(result, conf_threshold)
                    t_filter += (time.time() - t_f0)

                    boxes = result.boxes
                    has_det = boxes is not None and len(boxes) > 0
                    orig_file = None

                    _g = extract_group(filename)
                    if _g and _g not in batch_info[batch_id]['groups']:
                        batch_info[batch_id]['groups'].append(_g)

                    if has_det:
                        total_defect_imgs += 1
                        chunk_defects += len(boxes)
                    else:
                        total_no_defect_imgs += 1

                    # 安全调用 parse_results_to_global
                    if 'parse_results_to_global' in globals():
                        parse_results_to_global(result, filename, '', batch_id)
                    else:
                        print("[WARNING] parse_results_to_global not found in globals, skipping result storage")

                    # 提速（2026-10-21）：无检出的干净图不画图不存盘（产线大多数是干净图，省100~300ms/张）
                    if has_det:
                        t_p0 = time.time()
                        draw_img = result.plot()
                        t_plot += (time.time() - t_p0)

                        t_s0 = time.time()
                        result_filename = save_annotated_image(draw_img, filename)
                        t_save += (time.time() - t_s0)
                    else:
                        result_filename = None

                    locations = [list(map(int, e)) for e in boxes.xyxy.tolist()] if has_det else []
                    classes = [int(i) for i in boxes.cls.tolist()] if has_det else []
                    confs = ['%.2f %%' % (each * 100) for each in boxes.conf.tolist()] if has_det else []

                    # 有检出：留存原图并登记类别，供导出打包
                    if has_det:
                        orig_file = save_original_image(img)
                        register_problem_image(filename, orig_file, result_filename, classes, batch_id)

                    response.append({
                        "filename": filename,
                        "target_nums": len(classes),
                        "locations": locations,
                        "classes": classes,
                        "confs": confs,
                        "image_url": f"/api/results/{result_filename}" if result_filename else None,
                        "orig_url": f"/api/results/{orig_file}" if orig_file else None,
                        "batch_id": batch_id
                    })
                except Exception as e:
                    print(f"[ERROR] Processing result for {filename}: {e}")
                    traceback.print_exc()
                    response.append({
                        "filename": filename,
                        "target_nums": 0,
                        "locations": [],
                        "classes": [],
                        "confs": [],
                        "image_url": None,
                        "error": str(e)
                    })

            total_defects += chunk_defects
            chunk_done = min(i + BATCH_SIZE, total_files)
            chunk_time = t_read + t_infer + t_filter + t_plot + t_save
            avg_per_img = chunk_time / len(chunk_imgs) if chunk_imgs else 0
            print(f"[Batch Chunk {chunk_done}/{total_files}] 读取:{t_read:.3f}s | 推理:{t_infer:.3f}s | 过滤:{t_filter:.3f}s | 画图:{t_plot:.3f}s | 保存:{t_save:.3f}s | {len(chunk_imgs)}张 | 平均:{avg_per_img*1000:.1f}ms/张 | 缺陷:{chunk_defects}个")

        elapsed = time.time() - t_start
        avg_total = elapsed / total_files * 1000 if total_files > 0 else 0
        print(f"[Batch Done] 共{total_files}张 | 总耗时:{elapsed:.2f}s | 平均:{avg_total:.1f}ms/张 | 有缺陷:{total_defect_imgs}张 | 无缺陷:{total_no_defect_imgs}张 | 缺陷总数:{total_defects}个")
        save_state()

        return jsonify({
            "success": True,
            "total": total_files,
            "time": f"{elapsed:.3f}",
            "results": response
        })
    except Exception as e:
        print(f"[ERROR] detect_batch failed: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"批量检测失败: {type(e).__name__}: {str(e)}"}), 500


@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    """视频检测（文件上传方式）"""
    try:
        file = request.files.get('file')
        if not file:
            return jsonify({"success": False, "message": "未收到文件"})

        ext = os.path.splitext(file.filename)[1].lower()
        temp_filename = f"temp_{uuid.uuid4().hex}{ext}"
        temp_path = os.path.join(RESULTS_FOLDER, temp_filename)
        file.save(temp_path)

        conf_threshold = float(request.form.get('conf', CONF_DEFAULT))

        task_id = uuid.uuid4().hex
        video_tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "filename": file.filename,
            "result_url": None,
            "message": ""
        }

        thread = threading.Thread(
            target=process_video_task,
            args=(task_id, temp_path, conf_threshold, file.filename)
        )
        thread.start()

        return jsonify({"success": True, "task_id": task_id})
    except Exception as e:
        print(f"[ERROR] detect_video failed: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"视频提交失败: {type(e).__name__}: {str(e)}"}), 500


def process_video_task(task_id, temp_path, conf_threshold, original_filename):
    """后台线程处理视频"""
    try:
        cap = cv2.VideoCapture(temp_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps > 120:
            fps = 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = 1000

        result_filename = f"video_result_{task_id}.mp4"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))

        cur = 0
        t_start = time.time()
        t_total_infer = 0
        t_total_filter = 0
        t_total_plot = 0
        total_frames_with_defects = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                t_i0 = time.time()
                results = current_model(
                    frame,
                    imgsz=INFER_SIZE,
                    half=USE_FP16,
                    device=DEVICE,
                    max_det=MAX_DET,
                    verbose=False,
                    conf=0.25
                )[0]
                t_total_infer += (time.time() - t_i0)

                t_f0 = time.time()
                results = result_guolv_yolov8(results, conf_threshold)
                t_total_filter += (time.time() - t_f0)

                has_det = results.boxes is not None and len(results.boxes) > 0
                if has_det:
                    total_frames_with_defects += 1

                t_p0 = time.time()
                annotated = results.plot() if has_det else frame
                t_total_plot += (time.time() - t_p0)
            except Exception as e:
                print(f"[Video Error] frame {cur}: {e}")
                traceback.print_exc()
                annotated = frame
            out.write(annotated)
            cur += 1
            progress = int(cur / total * 100) if total > 0 else 0
            video_tasks[task_id]['progress'] = min(progress, 99)

            # 每100帧打印一次中间统计
            if cur % 100 == 0:
                t_elapsed = time.time() - t_start
                fps_avg = cur / t_elapsed if t_elapsed > 0 else 0
                print(f"[Video Progress] {cur}/{total} frames | 已耗时:{t_elapsed:.1f}s | 平均FPS:{fps_avg:.1f} | 有缺陷帧:{total_frames_with_defects}")

        cap.release()
        out.release()

        try:
            os.remove(temp_path)
        except Exception:
            pass

        t_total = time.time() - t_start
        avg_fps = cur / t_total if t_total > 0 else 0
        print(f"[Video Done] {original_filename} | {cur}帧 | 总耗时:{t_total:.2f}s | 平均FPS:{avg_fps:.1f} | 有缺陷帧:{total_frames_with_defects}/{cur} | 推理累计:{t_total_infer:.2f}s | 过滤累计:{t_total_filter:.2f}s | 画图累计:{t_total_plot:.2f}s")

        video_tasks[task_id]['status'] = 'done'
        video_tasks[task_id]['result_url'] = f"/api/results/{result_filename}"
        video_tasks[task_id]['progress'] = 100
    except Exception as e:
        print(f"[ERROR] process_video_task failed: {e}")
        traceback.print_exc()
        video_tasks[task_id]['status'] = 'error'
        video_tasks[task_id]['message'] = str(e)


@app.route('/api/task/<task_id>')
def get_task_status(task_id):
    task = video_tasks.get(task_id, {})
    return jsonify(task)


@app.route('/api/results/<filename>')
def serve_result(filename):
    try:
        path = os.path.join(RESULTS_FOLDER, filename)
        if not os.path.exists(path):
            return jsonify({"success": False, "message": "文件不存在"}), 404

        ext = os.path.splitext(filename)[1].lower()
        if ext in ['.xlsx', '.mp4', '.avi', '.mov', '.zip']:
            return send_file(path, as_attachment=True)
        return send_file(path)
    except Exception as e:
        print(f"[ERROR] serve_result failed: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"文件读取失败: {str(e)}"}), 500


def _build_report_excel(filepath, rows):
    """生成 Excel 报表到指定路径（rows：要导出的检测行，可为某一批次）"""
    df = pd.DataFrame(rows)
    wb = Workbook()
    ws = wb.active
    ws.title = "检测结果明细"

    header_font = Font(name="微软雅黑", size=12, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )
    data_font = Font(name="微软雅黑", size=11)
    data_alignment = Alignment(horizontal="center", vertical="center")

    headers = ["序号", "组别", "文件路径", "类别名称", "置信度(百分比)", "X最小值", "Y最小值", "X最大值", "Y最大值", "检测时间"]
    for col, header in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = thin_border

    for row_idx, (_, row) in enumerate(df.iterrows(), 2):
        ws.cell(row=row_idx, column=1, value=row["序号"]).font = data_font
        ws.cell(row=row_idx, column=2, value=row.get("组别", "")).font = data_font
        ws.cell(row=row_idx, column=3, value=row["文件路径"]).font = data_font
        ws.cell(row=row_idx, column=4, value=row["类别名称"]).font = data_font
        ws.cell(row=row_idx, column=5, value=row["置信度(百分比)"]).font = data_font
        ws.cell(row=row_idx, column=6, value=row["X最小值"]).font = data_font
        ws.cell(row=row_idx, column=7, value=row["Y最小值"]).font = data_font
        ws.cell(row=row_idx, column=8, value=row["X最大值"]).font = data_font
        ws.cell(row=row_idx, column=9, value=row["Y最大值"]).font = data_font
        ws.cell(row=row_idx, column=10, value=row["检测时间"]).font = data_font
        for col in range(1, 11):
            cell = ws.cell(row=row_idx, column=col)
            cell.alignment = data_alignment
            cell.border = thin_border

    # 汇总表
    ws_summary = wb.create_sheet(title="统计汇总")
    summary_data = [
        ["统计项", "数值"],
        ["检测文件数", len(set(item["文件全路径"] for item in rows))],
        ["检测目标总数", len(rows)],
        ["平均置信度", f"{df['置信度(数值)'].mean() * 100:.2f}%"],
        ["最高置信度", f"{df['置信度(数值)'].max() * 100:.2f}%"],
        ["最低置信度", f"{df['置信度(数值)'].min() * 100:.2f}%"],
        ["报表生成时间", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
    ]
    for row_idx, (key, value) in enumerate(summary_data, 1):
        if row_idx == 1:
            for col, val in enumerate([key, value], 1):
                cell = ws_summary.cell(row=1, column=col, value=val)
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = header_alignment
                cell.border = thin_border
        else:
            ws_summary.cell(row=row_idx, column=1, value=key).font = data_font
            ws_summary.cell(row=row_idx, column=2, value=value).font = data_font
            for col in range(1, 3):
                cell = ws_summary.cell(row=row_idx, column=col)
                cell.alignment = data_alignment
                cell.border = thin_border

    ws_summary.column_dimensions['A'].width = 20
    ws_summary.column_dimensions['B'].width = 25
    for col, width in enumerate([8, 12, 35, 15, 15, 10, 10, 10, 10, 20], 1):
        ws.column_dimensions[chr(64 + col)].width = width

    # 分组汇总表（智能检测模式：按 A/B/C... 电芯文件夹统计）
    if "组别" in df.columns and df["组别"].astype(str).str.len().gt(0).any():
        ws_group = wb.create_sheet(title="分组汇总")
        g_headers = ["组别", "缺陷框总数", "极片损伤", "暗斑", "亮斑", "涉及图片数", "平均置信度"]
        for col, header in enumerate(g_headers, 1):
            cell = ws_group.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
            cell.border = thin_border
        grp = df.groupby(df["组别"].astype(str))
        for row_idx, (gname, gdf) in enumerate(sorted(grp), 2):
            vals = [
                gname if gname else "（未分组）",
                len(gdf),
                int((gdf["类别ID"] == 0).sum()),
                int((gdf["类别ID"] == 1).sum()),
                int((gdf["类别ID"] == 2).sum()),
                gdf["文件路径"].nunique(),
                f"{gdf['置信度(数值)'].mean() * 100:.2f}%",
            ]
            for col, val in enumerate(vals, 1):
                cell = ws_group.cell(row=row_idx, column=col, value=val)
                cell.font = data_font
                cell.alignment = data_alignment
                cell.border = thin_border
        for col, width in enumerate([16, 12, 10, 10, 10, 12, 12], 1):
            ws_group.column_dimensions[chr(64 + col)].width = width

    wb.save(filepath)


def _export_worker(task_id):
    """后台打包：问题图（原图+标注图，镜像目录结构）→ zip，带进度；可按批次过滤"""
    import zipfile
    task = export_tasks[task_id]
    try:
        bid = task.get('batch_id')
        stamp = datetime.datetime.now().strftime('%Y%m%d')
        zip_name = f"检测结果_{_export_root_name(batch_id=bid)}_{stamp}.zip"
        zip_path = os.path.join(RESULTS_FOLDER, zip_name)

        imgs = [p for p in problem_images if not bid or p.get('批次ID', '') == bid]
        task['total'] = len(imgs)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # 问题图（目录结构镜像检测源目录；原名_类别_原图.jpg / 原名_类别_标注.jpg）
            used = set()
            for it in imgs:
                base, ext = os.path.splitext(it['文件名'])
                ext = ext or '.jpg'
                suffix = ('_' + it['缺陷']) if it.get('缺陷') else ''
                rel_dir = it.get('相对目录', '')
                name = f"{base}{suffix}"
                dup = 1
                while f"{rel_dir}/{name}" in used:  # 同目录同名才需区分
                    dup += 1
                    name = f"{base}_{dup}{suffix}"
                used.add(f"{rel_dir}/{name}")
                zip_dir = f"问题图/{rel_dir}" if rel_dir else "问题图"
                if it['原图']:
                    p = os.path.join(RESULTS_FOLDER, it['原图'])
                    if os.path.exists(p):
                        zf.write(p, f"{zip_dir}/{name}_原图{ext}")
                if it['标注图']:
                    p = os.path.join(RESULTS_FOLDER, it['标注图'])
                    if os.path.exists(p):
                        zf.write(p, f"{zip_dir}/{name}_标注{ext}")
                task['done'] += 1
                task['message'] = f"正在打包问题图 {task['done']}/{len(imgs)}"

        task['status'] = 'done'
        task['url'] = f"/api/results/{zip_name}"
        task['message'] = f"完成：{len(imgs)} 张问题图"
    except Exception as e:
        print(f"[ERROR] export worker failed: {e}")
        traceback.print_exc()
        task['status'] = 'error'
        task['message'] = f"{type(e).__name__}: {e}"


def _start_pack_task(batch_id=None):
    """打包任务启动器（防抖：进行中直接拒绝）"""
    for t in export_tasks.values():
        if t.get('status') == 'running':
            return None, "已有打包任务进行中，请稍候"
    task_id = uuid.uuid4().hex[:12]
    export_tasks[task_id] = {'status': 'running', 'done': 0, 'total': 0, 'url': None,
                             'message': '准备中…', 'batch_id': batch_id}
    threading.Thread(target=_export_worker, args=(task_id,), daemon=True).start()
    return task_id, None


def _export_root_name(batch_id=None):
    """导出文件命名用的源目录名：优先批次元信息的检测路径，其次问题图相对路径第一层"""
    if batch_id and batch_id in batch_info:
        root = (batch_info[batch_id].get('root') or '').replace('\\', '/').strip('/')
        if root:
            return root.split('/')[-1]   # 路径末层文件夹名（如 E:\xx\图片合集 → 图片合集）
    for it in problem_images:
        if batch_id and it.get('批次ID', '') != batch_id:
            continue
        rel = (it.get('相对目录') or '').strip('/')
        if rel:
            return rel.split('/')[0]
    for r in detection_results:
        if batch_id and r.get('批次ID', '') != batch_id:
            continue
        p = (r.get('文件路径') or '').replace('\\', '/').strip('/')
        if '/' in p:
            return p.split('/')[0]
    return '未分组'


@app.route('/api/export', methods=['POST'])
def export_report():
    """导出检测报表（仅 Excel，同步生成直接下载）；带 batch_id 时只导出该批次"""
    try:
        data = request.get_json(force=True, silent=True) or {}
        bid = data.get('batch_id') or None
        rows = [r for r in detection_results if not bid or r.get('批次ID', '') == bid]
        if not rows:
            return jsonify({"success": False, "message": "暂无检测结果可导出"})
        filename = f"检测报表_{_export_root_name(batch_id=bid)}_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        _build_report_excel(filepath, rows)
        return jsonify({"success": True, "download_url": f"/api/results/{filename}"})
    except Exception as e:
        print(f"[ERROR] export_report failed: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"报表导出失败: {type(e).__name__}: {str(e)}"}), 500


@app.route('/api/export_status/<task_id>')
def export_status(task_id):
    task = export_tasks.get(task_id)
    if not task:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    return jsonify({"success": True, **task})


# ==================== 路径直读检测（后台任务，2026-10-21） ====================
# 产线主力模式：图片在本地/移动硬盘，后端直接读盘零上传；后台线程检测不阻塞页面；
# 每批落盘可断点续检；只有问题图存备份（原图+标注图），干净图只计数不落盘。
detect_tasks = {}
TASK_FILE = os.path.join(RESULTS_FOLDER, '_detect_task.json')


def read_rgb_disk(path):
    """中文路径安全读图 → RGB numpy（与上传模式 PIL 读出的 RGB 一致）"""
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError('无法解码图片')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _list_images(d):
    try:
        return sorted(f for f in os.listdir(d) if f.lower().endswith(IMG_EXTS))
    except OSError:
        return []


def scan_structure(root):
    """分析目录结构，返回 (mode, groups)；group = {name, dir, count}
    flat:   root 下直接是图        → 隐式单组
    view:   root 下有"正面"子目录  → 隐式单组（只测正面）
    groups: root 下多个子文件夹    → 每个子文件夹一组（优先取其"正面"，支持再嵌套一层）
    """
    direct = _list_images(root)
    if direct:
        return 'flat', [{'name': '', 'dir': root, 'count': len(direct)}]
    front = os.path.join(root, SMART_VIEW)
    if os.path.isdir(front) and _list_images(front):
        return 'view', [{'name': '', 'dir': front, 'count': len(_list_images(front))}]
    groups = []
    for sub in sorted(os.listdir(root)):
        sd = os.path.join(root, sub)
        if not os.path.isdir(sd):
            continue
        f = os.path.join(sd, SMART_VIEW)
        direct_sub = _list_images(sd)
        if os.path.isdir(f) and _list_images(f):
            groups.append({'name': sub, 'dir': f, 'count': len(_list_images(f))})
        elif direct_sub:
            groups.append({'name': sub, 'dir': sd, 'count': len(direct_sub)})
        else:  # 再深一层（如选了 图片合集：图片合集/A/1-10/正面）
            for sub2 in sorted(os.listdir(sd)):
                sd2 = os.path.join(sd, sub2)
                if not os.path.isdir(sd2):
                    continue
                f2 = os.path.join(sd2, SMART_VIEW)
                direct2 = _list_images(sd2)
                if os.path.isdir(f2) and _list_images(f2):
                    groups.append({'name': f'{sub}/{sub2}', 'dir': f2, 'count': len(_list_images(f2))})
                elif direct2:
                    groups.append({'name': f'{sub}/{sub2}', 'dir': sd2, 'count': len(direct2)})
    return ('groups', groups) if groups else ('empty', [])


def backup_original_file(abs_path):
    """问题图原图备份：原样拷贝进 results/（不转码，保留原字节）"""
    ext = os.path.splitext(abs_path)[1].lower() or '.jpg'
    name = f"orig_{uuid.uuid4().hex}{ext}"
    try:
        shutil.copyfile(abs_path, os.path.join(RESULTS_FOLDER, name))
        return name
    except Exception as e:
        print(f"[Save Error] backup original failed: {e}")
        return None


def _write_task_file(info):
    try:
        with open(TASK_FILE, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] write task file failed: {e}")


def _interrupted_task():
    """开页/启动时检查：上次是否有没跑完的任务（可续检）"""
    try:
        if not os.path.exists(TASK_FILE):
            return None
        with open(TASK_FILE, encoding='utf-8') as f:
            info = json.load(f)
        if info.get('status') in ('running', 'stopped'):
            return info
    except Exception as e:
        print(f"[WARN] read task file failed: {e}")
    return None


@app.route('/api/pick_folder', methods=['POST'])
def api_pick_folder():
    """弹出 Windows 原生文件夹选择窗口（浏览器拿不到真实路径，由本地后端代劳）"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)   # 窗口置顶，防止被浏览器挡住
        path = filedialog.askdirectory(title='选择图片文件夹')
        root.destroy()
        if not path:
            return jsonify({"success": False, "message": "已取消选择"})
        return jsonify({"success": True, "path": os.path.normpath(path)})
    except Exception as e:
        print(f"[ERROR] pick_folder failed: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"无法打开选择窗口: {e}"})


@app.route('/api/scan_path', methods=['POST'])
def api_scan_path():
    """扫描本地文件夹结构：返回模式与分组清单（打勾选择的依据）"""
    data = request.get_json(force=True, silent=True) or {}
    raw = (data.get('path') or '').strip().strip('"').strip("'")
    if not raw:
        return jsonify({"success": False, "message": "请输入文件夹路径"})
    root = os.path.normpath(raw)
    if not os.path.isdir(root):
        return jsonify({"success": False, "message": f"路径不存在或不是文件夹：{raw}"})
    mode, groups = scan_structure(root)
    if mode == 'empty':
        return jsonify({"success": False, "message": "该文件夹（含两层子文件夹）内没有找到图片"})
    root_name = os.path.basename(root.rstrip('\\/')) or root
    out = [{'name': g['name'] or root_name, 'count': g['count']} for g in groups]
    return jsonify({"success": True, "mode": mode, "root": root,
                    "groups": out, "total": sum(g['count'] for g in groups)})


@app.route('/api/detect/start', methods=['POST'])
def api_detect_start():
    """启动路径直读后台检测任务（groups 缺省=全部；resume=续检上次中断任务）"""
    data = request.get_json(force=True, silent=True) or {}
    raw = (data.get('path') or '').strip().strip('"').strip("'")
    conf = float(data.get('conf', CONF_DEFAULT))
    sel = data.get('groups')
    resume = bool(data.get('resume'))
    if not raw:
        return jsonify({"success": False, "message": "缺少文件夹路径"})
    root = os.path.normpath(raw)
    if not os.path.isdir(root):
        return jsonify({"success": False, "message": "路径不存在，移动硬盘是否已连接？"})
    for t in detect_tasks.values():
        if t.get('status') == 'running':
            return jsonify({"success": False, "message": "已有检测任务进行中，请先停止或等待完成"})

    mode, groups = scan_structure(root)
    if not groups:
        return jsonify({"success": False, "message": "没有找到可检测的图片"})
    if sel:
        root_name = os.path.basename(root.rstrip('\\/')) or root
        groups = [g for g in groups if (g['name'] or root_name) in sel]
        if not groups:
            return jsonify({"success": False, "message": "所选分组没有图片"})

    parent = os.path.dirname(root.rstrip('\\/')) or root
    jobs = []
    for g in groups:
        for fn in _list_images(g['dir']):
            ap = os.path.join(g['dir'], fn)
            jobs.append((ap, os.path.relpath(ap, parent).replace('\\', '/')))

    skip = 0
    batch_id = new_batch_id()
    if resume:
        prev = _interrupted_task()
        if prev and os.path.normpath(prev.get('root', '')) == root:
            skip = min(int(prev.get('done', 0)), len(jobs))  # 顺序由扫描排序保证一致
            batch_id = prev.get('batch_id') or batch_id      # 续检沿用原批次：一次任务仍是一批
    if skip:
        jobs = jobs[skip:]

    task_id = uuid.uuid4().hex[:12]
    detect_tasks[task_id] = {
        'status': 'running', 'done': skip, 'total': skip + len(jobs),
        'defect_imgs': 0, 'defects': 0, 'errors': 0,
        'speed': 0.0, 'eta_s': -1, 'current': '', 'message': '',
        'items': [], 'stop': False, 'batch_id': batch_id,
    }
    if batch_id not in batch_info:   # 续检沿用原批次信息
        batch_info[batch_id] = {
            'root': root, 'mode': mode,
            'groups': [g['name'] or (os.path.basename(root.rstrip('\\/')) or root) for g in groups],
            'time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
    threading.Thread(target=_detect_worker,
                     args=(task_id, root, jobs, conf, skip, batch_id),
                     daemon=True).start()
    return jsonify({"success": True, "task_id": task_id, "batch_id": batch_id,
                    "total": skip + len(jobs), "skipped": skip})


def _detect_worker(task_id, root, jobs, conf, skip, batch_id):
    """后台检测线程：batch 读盘推理，问题图即存备份，逐批落盘可续检"""
    task = detect_tasks[task_id]
    model = current_model          # 捕获引用：任务中途切换模型不影响本任务
    t0 = time.time()
    processed = 0
    try:
        for i in range(0, len(jobs), BATCH_SIZE):
            if task.get('stop'):
                task['status'] = 'stopped'
                break
            chunk = jobs[i:i + BATCH_SIZE]
            imgs, aps, rels = [], [], []
            for ap, rel in chunk:
                try:
                    imgs.append(read_rgb_disk(ap))
                    aps.append(ap)
                    rels.append(rel)
                except Exception as e:
                    print(f"[Skip] 读图失败 {ap}: {e}")
                    task['errors'] += 1
            if imgs:
                results = model(imgs, imgsz=INFER_SIZE, half=USE_FP16, device=DEVICE,
                                max_det=MAX_DET, verbose=False, conf=0.25)
                for img, ap, rel, res in zip(imgs, aps, rels, results):
                    res = result_guolv_yolov8(res, conf)
                    boxes = res.boxes
                    has = boxes is not None and len(boxes) > 0
                    if has:
                        parse_results_to_global(res, rel, ap, batch_id)
                        draw = res.plot()
                        ann = save_annotated_image(draw, rel)
                        orig = backup_original_file(ap)
                        classes = [int(c) for c in boxes.cls.tolist()]
                        register_problem_image(rel, orig, ann, classes, batch_id)
                        task['defect_imgs'] += 1
                        task['defects'] += len(boxes)
                        task['items'].append({
                            "filename": rel,
                            "locations": [list(map(int, e)) for e in boxes.xyxy.tolist()],
                            "classes": classes,
                            "confs": ['%.2f %%' % (c * 100) for c in boxes.conf.tolist()],
                            "image_url": f"/api/results/{ann}" if ann else None,
                            "orig_url": f"/api/results/{orig}" if orig else None,
                            "batch_id": batch_id,
                            "time": datetime.datetime.now().strftime("%H:%M:%S"),
                        })
            processed += len(chunk)
            task['done'] = skip + processed
            task['current'] = rels[-1] if rels else ''
            el = time.time() - t0
            task['speed'] = round(processed / el, 1) if el > 0 else 0
            remain = len(jobs) - processed
            task['eta_s'] = int(remain / task['speed']) if task['speed'] > 0 else -1
            # 逐批落盘：检测结果 + 任务进度（断点续检依据）
            save_state()
            _write_task_file({'root': root, 'conf': conf, 'total': task['total'],
                              'done': task['done'], 'status': task['status'],
                              'batch_id': batch_id,
                              'updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
            print(f"[DetectTask] {task['done']}/{task['total']} | 缺陷图 {task['defect_imgs']} | {task['speed']}张/s")
        else:
            task['status'] = 'done'
        save_state()
        _write_task_file({'root': root, 'conf': conf, 'total': task['total'],
                          'done': task['done'], 'status': task['status'],
                          'batch_id': batch_id,
                          'updated': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        print(f"[DetectTask Done] {task['done']}/{task['total']} | 缺陷图 {task['defect_imgs']} | 缺陷 {task['defects']} | 状态 {task['status']}")
    except Exception as e:
        print(f"[ERROR] detect worker failed: {e}")
        traceback.print_exc()
        task['status'] = 'error'
        task['message'] = f"{type(e).__name__}: {e}"
        save_state()


@app.route('/api/detect_status/<task_id>')
def api_detect_status(task_id):
    """检测任务进度（cursor 增量返回新检出的问题图；干净图只计数不刷屏）"""
    task = detect_tasks.get(task_id)
    if not task:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    cursor = int(request.args.get('cursor', 0))
    items = task['items']
    return jsonify({"success": True, "status": task['status'],
                    "done": task['done'], "total": task['total'],
                    "defect_imgs": task['defect_imgs'], "defects": task['defects'],
                    "errors": task['errors'], "speed": task['speed'], "eta_s": task['eta_s'],
                    "current": task['current'], "message": task['message'],
                    "items": items[cursor:], "cursor": len(items)})


@app.route('/api/detect/stop', methods=['POST'])
def api_detect_stop():
    data = request.get_json(force=True, silent=True) or {}
    task = detect_tasks.get(data.get('task_id', ''))
    if not task:
        return jsonify({"success": False, "message": "任务不存在"}), 404
    task['stop'] = True
    return jsonify({"success": True, "message": "正在停止…"})


@app.route('/api/state')
def api_state():
    """开页恢复：历史检测结果 + 问题图映射 + 进行中/中断的任务（如有）"""
    active = None
    for tid, t in detect_tasks.items():
        if t.get('status') == 'running':
            active = {'task_id': tid, 'done': t['done'], 'total': t['total'],
                      'items_count': len(t['items']), 'batch_id': t.get('batch_id', '')}
            break
    return jsonify({"success": True,
                    "detection_results": detection_results,
                    "problem_images": problem_images,
                    "batch_info": batch_info,
                    "active_task": active,
                    "interrupted_task": _interrupted_task()})


@app.route('/api/state/delete_image', methods=['POST'])
def api_state_delete_image():
    """逐条删除：某批次内一张问题图的检测行 + 备份文件一并删除"""
    global detection_results
    data = request.get_json(force=True, silent=True) or {}
    rel = (data.get('filename') or '').replace('\\', '/').strip('/')
    bid = data.get('batch_id') or ''
    if not rel:
        return jsonify({"success": False, "message": "缺少文件名"}), 400
    before = len(detection_results)
    detection_results = [r for r in detection_results
                         if not ((r.get('文件路径') or '').replace('\\', '/').strip('/') == rel
                                 and (not bid or r.get('批次ID', '') == bid))]
    removed_rows = before - len(detection_results)
    files = []
    keep = []
    for p in problem_images:
        prel = ((p.get('相对目录') + '/') if p.get('相对目录') else '') + p.get('文件名', '')
        if prel == rel and (not bid or p.get('批次ID', '') == bid):
            files += [p.get('原图'), p.get('标注图')]
        else:
            keep.append(p)
    problem_images[:] = keep
    removed_files = 0
    for fn in files:
        if fn:
            try:
                os.remove(os.path.join(RESULTS_FOLDER, fn))
                removed_files += 1
            except OSError:
                pass
    save_state()
    return jsonify({"success": True, "removed_rows": removed_rows, "removed_files": removed_files})


@app.route('/api/state/delete_batch', methods=['POST'])
def api_state_delete_batch():
    """整批删除：一批的所有检测行 + 问题图备份文件"""
    global detection_results
    data = request.get_json(force=True, silent=True) or {}
    bid = data.get('batch_id') or ''
    if not bid:
        return jsonify({"success": False, "message": "缺少批次ID"}), 400
    before = len(detection_results)
    detection_results = [r for r in detection_results if r.get('批次ID', '') != bid]
    removed_rows = before - len(detection_results)
    files = []
    keep = []
    for p in problem_images:
        if p.get('批次ID', '') == bid:
            files += [p.get('原图'), p.get('标注图')]
        else:
            keep.append(p)
    removed_imgs = len(problem_images) - len(keep)
    problem_images[:] = keep
    batch_info.pop(bid, None)
    for fn in files:
        if fn:
            try:
                os.remove(os.path.join(RESULTS_FOLDER, fn))
            except OSError:
                pass
    save_state()
    return jsonify({"success": True, "removed_rows": removed_rows, "removed_imgs": removed_imgs})


@app.route('/api/batch/<bid>')
def api_batch(bid):
    """单批次的检测结果与问题图（历史结果点进主界面查看用）"""
    rows = [r for r in detection_results if r.get('批次ID', '') == bid]
    pims = [p for p in problem_images if p.get('批次ID', '') == bid]
    return jsonify({"success": True, "detection_results": rows, "problem_images": pims,
                    "info": batch_info.get(bid, {})})


@app.route('/api/clear', methods=['POST'])
def clear_results():
    """清空检测结果（results 目录）"""
    try:
        global detection_results
        detection_results = []
        problem_images.clear()
        batch_info.clear()
        try:
            os.remove(STATE_FILE)
        except OSError:
            pass

        for f in os.listdir(RESULTS_FOLDER):
            fpath = os.path.join(RESULTS_FOLDER, f)
            try:
                if os.path.isfile(fpath):
                    os.remove(fpath)
                elif os.path.isdir(fpath):
                    shutil.rmtree(fpath)
            except Exception as e:
                print(f'[Clear Warning] results/{f}: {e}')
        return jsonify({"success": True, "message": "已清空所有结果"})
    except Exception as e:
        print(f"[ERROR] clear_results failed: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"清空失败: {str(e)}"}), 500


@app.route('/api/download-all', methods=['POST'])
def download_all():
    """打包问题图（原图+标注图，规范命名）为 zip——异步任务；带 batch_id 时只打包该批次"""
    data = request.get_json(force=True, silent=True) or {}
    bid = data.get('batch_id') or None
    imgs = [p for p in problem_images if not bid or p.get('批次ID', '') == bid]
    if not imgs:
        return jsonify({"success": False, "message": "暂无问题图可下载（没有检出的图不打包）"})
    task_id, err = _start_pack_task(batch_id=bid)
    if err:
        return jsonify({"success": False, "message": err})
    return jsonify({"success": True, "task_id": task_id})


if __name__ == '__main__':
    PORT = int(os.environ.get('CT_PORT', '5000'))

    def open_browser():
        """等服务器真正就绪再打开（模型预热期间不再白屏）；
        优先以应用模式打开固定尺寸窗口（第一眼排版最佳）；无 Chrome/Edge 则退回默认浏览器"""
        import subprocess
        import urllib.request
        url = f'http://127.0.0.1:{PORT}/'
        for _ in range(120):   # 最多等 2 分钟（模型预热 10-20 秒）
            try:
                urllib.request.urlopen(url, timeout=1)
                break
            except Exception:
                time.sleep(1)
        candidates = [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
            os.path.expandvars(r'%LocalAppData%\Google\Chrome\Application\chrome.exe'),
            r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
            r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
        ]
        for exe in candidates:
            if os.path.exists(exe):
                try:
                    subprocess.Popen([exe, f'--app={url}', '--window-size=1600,950'])
                    print(f"[Browser] 应用模式打开 1600x950 ({os.path.basename(exe)})")
                    return
                except Exception as e:
                    print(f"[Browser] {exe} 启动失败: {e}")
        webbrowser.open(url)
        print("[Browser] 默认浏览器打开（尺寸由浏览器决定）")

    _it = _interrupted_task()
    if _it:
        print(f"[State] 发现未完成检测任务：{_it.get('root')} {_it.get('done')}/{_it.get('total')}，开页后可续检")

    if not os.environ.get('CT_NO_BROWSER'):
        threading.Thread(target=open_browser, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
