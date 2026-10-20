#coding:utf-8
"""
CT缺陷检测 — 交互式训练脚本（多模型版本管理版）

功能：
1. 自动扫描 datasets/ 下的所有数据集，让你选择用哪个训练
2. 每次训练起独立实验名，结果永不覆盖（除非你自己去 runs/ 里删）
3. 训练完成后报告 best.pt 位置，由你确认效果后手动复制到 models/启用/
   → 重启 Web 端即可在网页模型列表中选用
"""
import sys, os

# 把 core 目录加入路径，以便导入 Config
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(_SCRIPT_DIR, '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'core'))

# 切换到项目根目录，保证相对路径（datasets/ 等）正确
os.chdir(PROJECT_ROOT)

import glob
import datetime

import torch
from ultralytics import YOLO
import Config
from device_utils import pick_device


def find_datasets():
    """扫描 datasets/ 下所有含 data.yaml 的数据集"""
    result = []
    for yaml_path in sorted(glob.glob('datasets/*/data.yaml')):
        folder = os.path.dirname(yaml_path)
        # 顺便数一下训练图片数量
        n_train = len(glob.glob(os.path.join(folder, 'train', 'images', '*')))
        n_val = len(glob.glob(os.path.join(folder, 'val', 'images', '*')))
        result.append({'name': os.path.basename(folder), 'yaml': yaml_path,
                       'n_train': n_train, 'n_val': n_val})
    return result


def choose_dataset(datasets):
    """交互式选择数据集"""
    print('=' * 50)
    print('  可用数据集列表：')
    print('=' * 50)
    for i, d in enumerate(datasets, 1):
        print(f'  [{i}] {d["name"]:<25} 训练图:{d["n_train"]:>4}  验证图:{d["n_val"]:>4}')
    print()
    while True:
        choice = input(f'请选择数据集 (1-{len(datasets)}): ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(datasets):
            return datasets[int(choice) - 1]
        print('输入无效，请重新输入数字。')


def ask_experiment_name(default_name):
    """询问实验名称（作为训练结果文件夹名）"""
    print()
    print('给这次训练起个名字（会作为 runs 里的文件夹名）')
    print(f'直接回车使用默认名: {default_name}')
    name = input('实验名称: ').strip()
    return name if name else default_name


def choose_model_size():
    """选择预训练模型大小（扫描 models/pretrained/）"""
    avail = sorted(glob.glob('models/pretrained/*.pt'))
    desc = {'yolov8n': '最小最快，效果一般',
            'yolov8s': '更大更准，推荐（训练约15-30分钟）',
            'yolov8m': '再大一档，训练更久',
            'yolo11n': 'YOLO11 最小版'}
    print()
    print('=' * 50)
    print('  可用预训练模型（越大越准但越慢）：')
    print('=' * 50)
    for i, p in enumerate(avail, 1):
        base = os.path.basename(p).replace('.pt', '')
        print(f'  [{i}] {base:<12} {desc.get(base, "")}')
    print()
    while True:
        choice = input(f'请选择模型 (1-{len(avail)}): ').strip()
        if choice.isdigit() and 1 <= int(choice) <= len(avail):
            return avail[int(choice) - 1]
        print('输入无效，请重新输入数字。')


def ask_imgsz(default=1280):
    """询问训练分辨率"""
    print()
    print(f'训练分辨率（640 快；960 均衡，s模型推荐；1280 最清楚但 s 模型会非常慢；1920 仅 n 模型）')
    s = input(f'直接回车使用 {default}: ').strip()
    if not s:
        return default
    if s.isdigit() and int(s) % 32 == 0:
        return int(s)
    print('分辨率须为 32 的倍数，使用默认值')
    return default


def main():
    datasets = find_datasets()
    if not datasets:
        print('[错误] datasets/ 下没有找到任何数据集（缺少 data.yaml）')
        return

    # 1. 选择数据集
    ds = choose_dataset(datasets)
    print(f'\n✅ 已选择数据集: {ds["name"]}')

    # 2. 实验命名（默认：数据集名_月日_时分）
    default_name = f'{ds["name"]}_{datetime.datetime.now().strftime("%m%d_%H%M")}'
    exp_name = ask_experiment_name(default_name)

    # 3. 选择模型大小和分辨率
    model_pt = choose_model_size()
    imgsz = int(os.environ.get('CT_IMGSZ', 0)) or ask_imgsz()

    # 4. 训练参数（自动适配：GPU 可用则用 GPU，否则退回 CPU）
    device = pick_device(verbose=False)
    dev_desc = ('GPU: ' + torch.cuda.get_device_name(0)) if device == 0 else 'CPU（无可用显卡，训练会慢很多）'
    print(f'\n[训练设备] {dev_desc}')
    print(f'[预训练模型] {model_pt}')
    print(f'[分辨率] {imgsz}')
    print(f'[实验名称] {exp_name}')
    print(f'[输出目录] runs/detect/CT_pro/{exp_name}/')
    print()

    model = YOLO(model_pt)  # 加载本地预训练模型

    # batch 策略：GPU 模式默认按显存占比自动拉满（榨干显卡），可用环境变量覆盖：
    #   CT_BATCH=16    → 固定批大小
    #   CT_BATCH=0.85  → 按 85% 显存自动计算（更激进）
    #   CPU 训练时固定为 4
    batch_env = os.environ.get('CT_BATCH', '')
    if batch_env:
        batch = float(batch_env)
        batch = int(batch) if batch >= 1 else batch
    else:
        batch = 0.75 if device == 0 else 4
    print(f'[批大小] {"自动(显存75%)" if batch == 0.75 else batch}')

    results = model.train(
        data=ds['yaml'],
        epochs=int(os.environ.get('CT_EPOCHS', 300)),  # 可用环境变量 CT_EPOCHS 临时改轮数（调试用）
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=30,          # 早停：30 轮不提升则停
        project=os.path.join(PROJECT_ROOT, 'runs', 'detect', 'CT_pro'),  # 绝对路径，防止目录嵌套错乱
        name=exp_name,
        exist_ok=False,       # ★ 关键：不覆盖旧实验，自动建 train2、train3...
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        cos_lr=True,
        close_mosaic=10,
        cache={'0': False, '1': True, 'ram': True}.get(os.environ.get('CT_CACHE', 'disk').lower(), 'disk'),  # 默认缓存到磁盘，避免大数据集内存溢出；CT_CACHE=0 关闭
        workers=4,
        val=True,
        plots=True,
        # --- 数据增强（原 scripts/augment.yaml 内联版）---
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=7.5, cls=0.5, dfl=1.5,
        degrees=5.0, translate=0.05, scale=0.05,
        shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.1,   # 亮度扰动降到±10%，保护微弱亮斑（2026-10-13 调整）
        mosaic=1.0, mixup=0.0, copy_paste=0.0,  # 关掉 mixup：图像叠加会稀释微弱亮斑（2026-10-13 调整）
    )

    # 4. 训练完成：只报告模型位置，不自动启用
    # 网页可选的模型只来自 models/启用/，训练出的模型需人工确认效果后手动复制进去
    best_pt = None
    try:
        candidate = str(model.trainer.best)
        if os.path.exists(candidate):
            best_pt = candidate
    except Exception:
        pass
    if not best_pt:
        candidates = sorted(glob.glob(os.path.join(PROJECT_ROOT, 'runs', 'detect', 'CT_pro', f'{exp_name}*', 'weights', 'best.pt')),
                            key=os.path.getmtime)
        if candidates:
            best_pt = candidates[-1]
    print()
    print('=' * 50)
    if best_pt and os.path.exists(best_pt):
        print(f'  🎉 训练完成！')
        print(f'  新模型位于: {best_pt}')
        print(f'  📊 训练曲线/报告在: {os.path.dirname(os.path.dirname(best_pt))}')
        print()
        print('  👉 确认这个模型效果满意后，把上面的 best.pt 复制到:')
        print('     models\\启用\\  （可以改个名字，如 A型号第一批.pt）')
        print('     重启 Web 端后即可在网页模型列表中选用')
    else:
        print('  [警告] 未找到训练输出的 best.pt，请检查 runs/detect/CT_pro/ 目录')
    print('=' * 50)


if __name__ == '__main__':
    main()
