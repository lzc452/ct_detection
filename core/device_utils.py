# -*- coding: utf-8 -*-
"""
设备自动选择：适配各种电脑
- 有可用 NVIDIA 显卡（架构受当前 torch 支持）→ 返回 0（使用 GPU）
- 无显卡 / 显卡太旧（如 GTX 10 系及更早，cu128 不支持 sm<70）/ 驱动异常 → 返回 'cpu'

判断方式：不只看 torch.cuda.is_available()，而是真正在显卡上跑一次小张量运算，
任何异常（包括 "no kernel image found" 这类架构不支持错误）都安全退回 CPU。
"""
import torch

_device_cache = None


def pick_device(verbose=True):
    """返回 0（GPU）或 'cpu'，结果缓存，多次调用只探测一次。"""
    global _device_cache
    if _device_cache is not None:
        return _device_cache

    device = 'cpu'
    reason = '未检测到 NVIDIA 显卡'
    if torch.cuda.is_available():
        try:
            # 真正在显卡上做一次运算，验证架构/驱动确实可用
            x = torch.zeros(8, 8, device='cuda') + 1
            y = x @ x
            torch.cuda.synchronize()
            _ = float(y.sum())
            device = 0
            reason = torch.cuda.get_device_name(0)
        except Exception as e:
            device = 'cpu'
            reason = f'显卡存在但不可用（{type(e).__name__}: {str(e)[:80]}），已退回 CPU'

    _device_cache = device
    if verbose:
        if device == 0:
            print(f'[设备] 使用 GPU: {reason}')
        else:
            print(f'[设备] 使用 CPU —— {reason}')
    return device


def is_cuda_device(device):
    """识别 pick_device 返回的 GPU 编号与 CUDA 写法；不额外探测显卡。"""
    if isinstance(device, bool):
        return False
    text = str(device).strip().lower()
    return text.isdecimal() or text == "cuda" or (
        text.startswith("cuda:") and text[5:].isdecimal())
