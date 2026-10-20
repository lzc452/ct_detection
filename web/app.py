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
USE_FP16 = False
INFER_SIZE = 1920   # ★必须与训练分辨率一致：1920训练的模型在640下几乎检不出缺陷（2026-10-14 实测）
BATCH_SIZE = 4      # 1920 下单批4张，兼顾显存与批量速度
MAX_DET = 300
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

STATE_FILE = os.path.join(RESULTS_FOLDER, '_state.json')


def save_state():
    """检测结果落盘：刷新页面/重启服务后可恢复（致命数据不再丢）"""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump({'detection_results': detection_results,
                       'problem_images': problem_images}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[WARN] save_state failed: {e}")


def load_state():
    """启动时恢复上次检测结果"""
    global detection_results
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, encoding='utf-8') as f:
                st = json.load(f)
            detection_results = st.get('detection_results', [])
            problem_images.extend(st.get('problem_images', []))
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


def register_problem_image(filename, orig_file, annotated_file, cls_ids):
    """记录一张问题图（原图+标注图+缺陷类别+相对路径），供导出打包"""
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


def parse_results_to_global(results, original_filename, filepath=''):
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
            parse_results_to_global(results, filename, '')
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
        if has_det:
            orig_file = save_original_image(img)
            register_problem_image(filename, orig_file, result_filename, classes)
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
        file_paths = json.loads(request.form.get('file_paths', '[]'))

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

                    if has_det:
                        total_defect_imgs += 1
                        chunk_defects += len(boxes)
                    else:
                        total_no_defect_imgs += 1

                    # 安全调用 parse_results_to_global
                    if 'parse_results_to_global' in globals():
                        parse_results_to_global(result, filename, '')
                    else:
                        print("[WARNING] parse_results_to_global not found in globals, skipping result storage")

                    t_p0 = time.time()
                    draw_img = result.plot()
                    t_plot += (time.time() - t_p0)

                    t_s0 = time.time()
                    result_filename = save_annotated_image(draw_img, filename)
                    t_save += (time.time() - t_s0)

                    locations = [list(map(int, e)) for e in boxes.xyxy.tolist()] if has_det else []
                    classes = [int(i) for i in boxes.cls.tolist()] if has_det else []
                    confs = ['%.2f %%' % (each * 100) for each in boxes.conf.tolist()] if has_det else []

                    # 有检出：留存原图并登记类别，供导出打包
                    if has_det:
                        orig_file = save_original_image(img)
                        register_problem_image(filename, orig_file, result_filename, classes)

                    response.append({
                        "filename": filename,
                        "target_nums": len(classes),
                        "locations": locations,
                        "classes": classes,
                        "confs": confs,
                        "image_url": f"/api/results/{result_filename}" if result_filename else None
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


def _build_report_excel(filepath):
    """生成 Excel 报表到指定路径（导出任务的子步骤）"""
    df = pd.DataFrame(detection_results)
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
        ["检测文件数", len(set(item["文件全路径"] for item in detection_results))],
        ["检测目标总数", len(detection_results)],
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
    """后台打包：问题图（原图+标注图，镜像目录结构）→ zip，带进度"""
    import zipfile
    task = export_tasks[task_id]
    try:
        stamp = datetime.datetime.now().strftime('%Y%m%d')
        zip_name = f"检测结果_{_export_root_name()}_{stamp}.zip"
        zip_path = os.path.join(RESULTS_FOLDER, zip_name)

        imgs = list(problem_images)  # 快照，避免打包期间被新增
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


def _start_pack_task():
    """打包任务启动器（防抖：进行中直接拒绝）"""
    for t in export_tasks.values():
        if t.get('status') == 'running':
            return None, "已有打包任务进行中，请稍候"
    task_id = uuid.uuid4().hex[:12]
    export_tasks[task_id] = {'status': 'running', 'done': 0, 'total': 0, 'url': None, 'message': '准备中…'}
    threading.Thread(target=_export_worker, args=(task_id,), daemon=True).start()
    return task_id, None


def _export_root_name():
    """导出文件命名用的源目录名：取问题图相对路径的第一层（如 图片合集/A/正面/x.jpg → 图片合集）"""
    for it in problem_images:
        rel = (it.get('相对目录') or '').strip('/')
        if rel:
            return rel.split('/')[0]
    for r in detection_results:
        p = (r.get('文件路径') or '').replace('\\', '/').strip('/')
        if '/' in p:
            return p.split('/')[0]
    return '未分组'


@app.route('/api/export', methods=['POST'])
def export_report():
    """导出检测报表（仅 Excel，同步生成直接下载）"""
    try:
        if not detection_results:
            return jsonify({"success": False, "message": "暂无检测结果可导出"})
        filename = f"检测报表_{_export_root_name()}_{datetime.datetime.now().strftime('%Y%m%d')}.xlsx"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        _build_report_excel(filepath)
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


@app.route('/api/clear', methods=['POST'])
def clear_results():
    """清空检测结果（results 目录）"""
    try:
        global detection_results
        detection_results = []
        problem_images.clear()
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
    """打包问题图（原图+标注图，规范命名）为 zip——异步任务，共用进度接口"""
    if not problem_images:
        return jsonify({"success": False, "message": "暂无问题图可下载（没有检出的图不打包）"})
    task_id, err = _start_pack_task()
    if err:
        return jsonify({"success": False, "message": err})
    return jsonify({"success": True, "task_id": task_id})


if __name__ == '__main__':
    def open_browser():
        """优先以应用模式打开固定尺寸窗口（第一眼排版最佳）；无 Chrome/Edge 则退回默认浏览器"""
        import subprocess
        url = 'http://127.0.0.1:5000/'
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

    # 服务启动后自动打开浏览器（延迟2秒等服务器就绪）
    threading.Timer(2.0, open_browser).start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
