# -*- coding: utf-8 -*-
"""检测图像契约：内存统一HWC/BGR/uint8，PIL只负责上传流解码。
中文路径通过imdecode/fromfile和imencode/tofile读写，不隐式交换通道。
"""
import cv2
import numpy as np
from PIL import Image


def read_bgr_upload(stream):
    """保持原有PIL上传解码器，只纠正交给Ultralytics的通道顺序。"""
    with Image.open(stream) as image:
        return cv2.cvtColor(np.array(image.convert('RGB')), cv2.COLOR_RGB2BGR)


def read_bgr_disk(path):
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('无法解码图片')
    return image


def write_bgr(path, image, ext, params=None):
    """编码失败必须抛出；调用方不能把不存在的图片登记成成功。"""
    ok, buffer = cv2.imencode(ext, image, params or [])
    if not ok:
        raise OSError('图片编码失败: ' + ext)
    buffer.tofile(str(path))
