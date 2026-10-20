# -*- coding: utf-8 -*-
"""
CT图像预处理模块
专门用于软包电池CT缺陷检测的图像增强
"""

import cv2
import numpy as np


def denoise_gaussian(img, kernel_size=3, sigma=0.8):
    """高斯滤波去噪 - 消除CT颗粒噪声，保留边缘"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)


def clahe_enhance(img, clip_limit=2.0, tile_grid=(8, 8)):
    """
    CLAHE自适应直方图均衡化
    对CT图像的局部对比度增强效果极佳，缺陷区域灰度差异会被放大
    """
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        return clahe.apply(img)


def contrast_stretch(img, low_percentile=2, high_percentile=98):
    """
    对比度拉伸 - 将灰度范围压缩到全动态范围
    对CT图像特别有效：暗斑和亮斑的灰度差异会被拉大
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    lo, hi = np.percentile(img, [low_percentile, high_percentile])
    stretched = np.clip((img.astype(float) - lo) / (hi - lo + 1e-6) * 255, 0, 255)
    return stretched.astype(np.uint8)


def unsharp_mask(img, sigma=1.5, alpha=1.5, beta=-0.5):
    """
    Unsharp Masking锐化
    增强缺陷边缘，让裂纹和暗斑的边界更清晰
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, alpha, blurred, beta, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def laplacian_sharpen(img):
    """拉普拉斯锐化 - 增强细节"""
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    result = img.astype(np.float64) - laplacian
    return np.clip(result, 0, 255).astype(np.uint8)


def gamma_correction(img, gamma=1.2):
    """伽马校正 - 调整中间调亮度"""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)


def preprocess_ct_image(img, mode='combo'):
    """
    CT图像预处理主函数
    
    Args:
        img: 输入图像 (numpy array, BGR或灰度)
        mode: 预处理模式
            - 'none': 原图不做处理
            - 'clahe': 仅CLAHE
            - 'stretch': 仅对比度拉伸
            - 'sharpen': 仅锐化
            - 'denoise_clahe': 去噪 + CLAHE
            - 'combo': 组合预处理（推荐）
            - 'combo_v2': 组合预处理v2（更强增强）
    
    Returns:
        预处理后的图像 (BGR格式，保持原通道数)
    """
    # 保存原始通道信息
    is_color = len(img.shape) == 3
    original_dtype = img.dtype
    
    # 转为灰度处理（CT图都是灰度）
    if is_color:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    if mode == 'none':
        result = gray
    elif mode == 'clahe':
        result = clahe_enhance(gray, clip_limit=2.0, tile_grid=(8, 8))
    elif mode == 'stretch':
        result = contrast_stretch(gray, low_percentile=2, high_percentile=98)
    elif mode == 'sharpen':
        result = unsharp_mask(gray, sigma=1.5, alpha=1.5, beta=-0.5)
    elif mode == 'denoise_clahe':
        denoised = denoise_gaussian(gray, kernel_size=3, sigma=0.8)
        result = clahe_enhance(denoised, clip_limit=3.0, tile_grid=(8, 8))
    elif mode == 'combo':
        # 推荐组合：去噪 → CLAHE → 轻度锐化
        denoised = denoise_gaussian(gray, kernel_size=3, sigma=0.8)
        clahed = clahe_enhance(denoised, clip_limit=2.5, tile_grid=(8, 8))
        result = unsharp_mask(clahed, sigma=2.0, alpha=1.3, beta=-0.3)
    elif mode == 'combo_v2':
        # 强增强组合：对比度拉伸 → CLAHE → 锐化 → 伽马校正
        stretched = contrast_stretch(gray, low_percentile=1, high_percentile=99)
        clahed = clahe_enhance(stretched, clip_limit=4.0, tile_grid=(8, 8))
        sharpened = unsharp_mask(clahed, sigma=1.5, alpha=1.5, beta=-0.5)
        result = gamma_correction(sharpened, gamma=1.1)
    else:
        result = gray
    
    # 转回BGR格式（YOLO需要3通道）
    if is_color:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    else:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    
    return result


# ==================== 用于训练时的批量预处理 ====================

def preprocess_dataset_images(input_dir, output_dir, mode='combo'):
    """
    批量预处理数据集图像（用于训练前处理）
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出图像目录
        mode: 预处理模式
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for fname in os.listdir(input_dir):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img = cv2.imread(os.path.join(input_dir, fname))
            if img is not None:
                processed = preprocess_ct_image(img, mode=mode)
                cv2.imwrite(os.path.join(output_dir, fname), processed)


if __name__ == '__main__':
    # 测试：对单张图运行所有预处理模式并对比
    import os
    test_img = r"datasets/CT_DATASET/train/images/dark (1).jpg"
    if os.path.exists(test_img):
        img = cv2.imread(test_img)
        modes = ['none', 'clahe', 'stretch', 'denoise_clahe', 'combo', 'combo_v2']
        for m in modes:
            out = preprocess_ct_image(img, mode=m)
            cv2.imwrite(f"test_pre_{m}.jpg", out)
        print("已生成测试文件: test_pre_*.jpg")
    else:
        print(f"测试图片不存在: {test_img}")
