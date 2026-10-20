# -*- coding: utf-8 -*-
"""
CT缺陷检测系统 — 环境检查脚本
运行后会检查所有关键依赖和文件是否就绪
"""
import sys
import os

# 把 core 目录加入路径，以便导入 Config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'core'))

# 项目根目录（无论从哪里运行，检查都以根目录为基准）
PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
os.chdir(PROJECT_ROOT)

print("=" * 60)
print("  CT缺陷检测系统 — 环境检查")
print("=" * 60)
print()

# 检查1: Python版本
print("[检查 1/8] Python 版本...")
py_version = sys.version_info
if py_version.major == 3 and 8 <= py_version.minor <= 11:
    print(f"   ✅ Python {py_version.major}.{py_version.minor}.{py_version.micro} — 版本合适")
else:
    print(f"   ⚠️ Python {py_version.major}.{py_version.minor}.{py_version.micro} — 建议用 3.8~3.10")
print()

# 检查2: 便携运行环境
print("[检查 2/8] 便携运行环境（runtime）...")
runtime_py = os.path.join(PROJECT_ROOT, 'runtime', 'python.exe')
if os.path.exists(runtime_py):
    print(f"   ✅ 便携 Python 存在: runtime\\python.exe")
    print("      本系统无需安装任何软件，整个文件夹拷贝即可使用")
else:
    print(f"   ❌ 未找到 runtime\\python.exe —— 文件夹可能拷贝不完整！")
    all_ok = False
print()

# 检查3: 关键库
print("[检查 3/8] 关键 Python 库...")
libraries = {
    'ultralytics': 'YOLOv8 框架',
    'flask': 'Web 服务',
    'cv2': 'OpenCV 图像处理',
    'torch': 'PyTorch 深度学习',
    'numpy': '数值计算',
    'pandas': '数据处理',
    'openpyxl': 'Excel 报表导出',
    'PIL': '图像处理',
    'matplotlib': '绘图',
    'yaml': '配置文件解析',
    'psutil': '系统监控',
    'tqdm': '进度条',
}

all_ok = True
for lib, desc in libraries.items():
    try:
        mod = __import__(lib)
        version = getattr(mod, '__version__', '未知版本')
        print(f"   ✅ {desc:<12} ({lib:<15}) — {version}")
    except ImportError:
        print(f"   ❌ {desc:<12} ({lib:<15}) — 未安装！")
        all_ok = False
print()

# 检查4: CUDA可用性
print("[检查 4/8] GPU/CUDA 状态...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✅ CUDA 可用 — 显卡: {torch.cuda.get_device_name(0)}")
        print(f"      CUDA版本: {torch.version.cuda}")
    else:
        print("   ℹ️ 使用 CPU 推理（无独立显卡或CUDA未配置）")
        print("      桌面端检测速度会稍慢，但完全可用")
except:
    print("   ⚠️ 无法检测 GPU 状态")
print()

# 检查5: 项目文件
print("[检查 5/8] 项目核心文件...")
required_files = [
    ('core/Config.py', '全局配置'),
    ('web/app.py', 'Web端主程序'),
    ('web/templates/index.html', 'Web前端页面'),
    ('core/ct_preprocess.py', 'CT预处理'),
    ('scripts/train.py', '训练脚本'),
]
for fname, desc in required_files:
    if os.path.exists(fname):
        print(f"   ✅ {desc:<10} — {fname}")
    else:
        print(f"   ❌ {desc:<10} — {fname} 缺失！")
        all_ok = False
print()

# 检查6: 模型文件
print("[检查 6/8] 模型文件...")
import Config
model_path = Config.model_path
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✅ 模型文件存在: {model_path}")
    print(f"      大小: {size_mb:.1f} MB")
else:
    print(f"   ❌ 模型文件不存在: {model_path}")
    print(f"      请检查 Config.py 中的 model_path 设置")
    # 尝试找 models/ 下的 .pt 文件
    if os.path.exists('models'):
        pt_files = [f for f in os.listdir('models') if f.endswith('.pt')]
        if pt_files:
            print(f"      发现可用模型: {', '.join(pt_files)}")
            print(f"      建议修改 Config.py: model_path = 'models/{pt_files[0]}'")
    all_ok = False
print()

# 检查7: 数据集
print("[检查 7/8] 数据集...")
data_yaml = Config.data_ymal_path
if os.path.exists(data_yaml):
    print(f"   ✅ 数据集配置存在: {data_yaml}")
else:
    print(f"   ❌ 数据集配置不存在: {data_yaml}")
    print(f"      请确认 datasets/ 文件夹已完整复制")
    all_ok = False
print()

# 检查8: 简单推理测试
print("[检查 8/8] 模型推理测试...")
try:
    from ultralytics import YOLO
    import numpy as np
    
    print("   正在加载模型...")
    model = YOLO(Config.model_path, task='detect')
    
    print("   正在运行预热推理...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(dummy, verbose=False)
    
    print("   ✅ 模型加载和推理测试通过！")
except Exception as e:
    print(f"   ❌ 推理测试失败: {e}")
    all_ok = False
print()

# 总结
print("=" * 60)
if all_ok:
    print("  🎉 环境检查全部通过！可以开始使用了。")
    print()
    print("  启动方式: 双击「启动检测.bat」，浏览器访问 http://127.0.0.1:5000/")
else:
    print("  ⚠️ 有部分检查未通过，请根据上方提示修复后再试。")
    print("  常见问题:")
    print("    - runtime 缺失: 确认整个文件夹是完整拷贝的（含 runtime 文件夹）")
    print("    - 模型缺失: 确认 models/best.pt 存在")
    print("    - 数据集缺失: 确认 datasets/ 文件夹完整")
print("=" * 60)
