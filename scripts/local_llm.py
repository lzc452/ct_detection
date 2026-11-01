# -*- coding: utf-8 -*-
"""本机 Qwen 部署只读扫描、方案生成与资源预检，不下载模型、不启动推理。

用法（项目根目录）：
    runtime/python.exe scripts/local_llm.py scan --output docs/local_llm_hardware_20261101.json
    runtime/python.exe scripts/local_llm.py plan --model qwen36-35b
    runtime/python.exe scripts/local_llm.py preflight --model qwen38-27b

仅使用标准库。适配 Windows 原生 llama.cpp / Strata，模型目录默认 E:/LocalLLM。
"""
import argparse
import csv
import datetime
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

PROFILES = {
    "qwen38-27b": {
        "engine": "llama-server", "family": None, "quant": "UD-Q4_K_M",
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "filename": "Qwen3.8-27B-UD-Q4_K_M.gguf",
        "download_gb": 16.5, "disk_budget_gib": 30, "free_ram_floor_gib": 24,
        "note": "稠密模型，8GB 显存使用自动分层卸载；不传 MoE 参数。",
    },
    "qwen36-35b": {
        "engine": "strata", "family": "qwen36", "quant": "UD-Q4_K_XL",
        "repo": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
        "filename": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "download_gb": 22.9, "disk_budget_gib": 40, "free_ram_floor_gib": 30,
        "note": "MoE 混合推理，使用含 MTP 的 GGUF；先以 8K、单并发、纯文本验证。",
    },
    "flash-next": {
        "engine": "strata", "family": "qwen", "quant": "IQ2_XS",
        "repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
        "filename": "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf",
        "download_gb": 68, "disk_budget_gib": 120, "free_ram_floor_gib": 44,
        "note": "8GB 属小显存实验路线；另需第二分片、MTP 及准备空间，不能只下载首分片。",
    },
}


def run_readonly(command):
    """执行固定的只读命令；不通过 shell、不安装依赖。"""
    try:
        result = subprocess.run(command, capture_output=True, timeout=30, shell=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "", str(exc)
    raw = result.stdout
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        output = raw.decode("utf-16", errors="replace")
    elif b"\x00" in raw[:200]:
        output = raw.decode("utf-16-le", errors="replace")
    else:
        try:
            output = raw.decode("utf-8")
        except UnicodeDecodeError:
            output = raw.decode("mbcs", errors="replace")
    if result.returncode:
        return "", "命令失败，退出码 %s" % result.returncode
    return output, None


def wmi_rows(alias, fields, errors, prefix=None):
    command = ["wmic"] + (prefix or [alias]) + ["get", fields, "/value"]
    output, error = run_readonly(command)
    if error:
        errors.append({"query": alias, "error": error})
        return []
    # 使用 key=value 避免 WMIC CSV 未转义厂商名称里的逗号；重复字段开始下一条记录。
    lines = [line.strip() for line in output.replace("\r", "\n").splitlines() if line.strip()]
    rows = []
    row = {}
    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in row:
            rows.append(row)
            row = {}
        row[key] = value.strip()
    if row:
        rows.append(row)
    return rows


def scan():
    errors = []
    result = {
        "scanned_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "cpu": wmi_rows("cpu", "Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed", errors),
        "memory": wmi_rows("memorychip", "Capacity,Speed,ConfiguredClockSpeed,Manufacturer,PartNumber", errors),
        "os": wmi_rows("os", "Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory", errors),
        "board": wmi_rows("baseboard", "Manufacturer,Product", errors),
        "disks": wmi_rows("diskdrive", "Index,Model,Size,InterfaceType", errors),
        "physical_disks": wmi_rows("physicaldisk", "FriendlyName,BusType,MediaType,Size,HealthStatus", errors,
                                  [r"/namespace:\\root\Microsoft\Windows\Storage", "path", "MSFT_PhysicalDisk"]),
        "volumes": wmi_rows("logicaldisk", "DeviceID,DriveType,FileSystem,Size,FreeSpace", errors),
        "tools_on_path": {name: shutil.which(name) for name in ("strata", "llama-server", "ollama", "nvcc", "wsl", "docker")},
        "python_version": sys.version.split()[0],
    }
    output, error = run_readonly([
        "nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
        "--format=csv,noheader,nounits",
    ])
    result["nvidia_gpus"] = []
    if error:
        errors.append({"query": "nvidia-smi", "error": error})
    else:
        for row in csv.reader(io.StringIO(output)):
            if len(row) != 5:
                continue
            result["nvidia_gpus"].append({
                "name": row[0].strip(), "total_mib": int(row[1]),
                "used_mib": int(row[2]), "free_mib": int(row[3]), "driver": row[4].strip(),
            })
    result["errors"] = errors
    return result


def make_plan(model, data_dir, binary=None):
    profile = PROFILES[model]
    directory = Path(data_dir) / model
    binary = binary or profile["engine"]
    if profile["engine"] == "llama-server":
        command = [binary, "--model", str(directory / profile["filename"]),
                   "--ctx-size", "8192", "--parallel", "1", "--gpu-layers", "auto",
                   "--fit-target", "1024", "--threads", "8", "--threads-batch", "8",
                   "--batch-size", "256", "--ubatch-size", "128", "--flash-attn", "on",
                   "--jinja", "--host", "127.0.0.1", "--port", "8081"]
        plan_command = None
    else:
        command = [binary, "launch", "--family", profile["family"], "--model", profile["quant"],
                   "--data-dir", str(directory), "--context", "8192", "--parallel", "1",
                   "--vision", "off", "--access", "local", "--port", "8081"]
        if model == "flash-next":
            command += ["--set", "STRATA_ROUTE_PRIOR=0", "--set", "STRATA_VRAM_FREE=device",
                        "--set", "vram-reserve-mib=1024"]
        plan_command = command + ["--plan", "--output", "json"]
        command += ["--yes", "--no-tui"]
    return {
        "model": model, "profile": profile, "directory": str(directory),
        "context_tokens": 8192, "parallel": 1, "vision": "off",
        "upstream_plan_argv": plan_command,
        "upstream_plan_command": subprocess.list2cmdline(plan_command) if plan_command else None,
        "launch_argv": command, "launch_command": subprocess.list2cmdline(command),
        "executed": False,
        "status": "仅生成方案；未下载、未启动，也未证明模型能成功推理。",
    }


def preflight(plan, hardware, min_free_vram_mib):
    """本机保守资源预检，不能代替引擎装载与真实请求验收。"""
    profile = plan["profile"]
    checks = []
    binary = plan["launch_argv"][0]
    binary_found = bool(shutil.which(binary) or Path(binary).is_file())
    checks.append({"name": "engine_available", "ok": binary_found, "detail": binary})
    gpus = hardware["nvidia_gpus"]
    free_vram = max((gpu["free_mib"] for gpu in gpus), default=0)
    checks.append({"name": "gpu_free_memory", "ok": free_vram >= min_free_vram_mib,
                   "detail": {"free_mib": free_vram, "required_mib": min_free_vram_mib}})
    ram_rows = hardware["os"]
    free_ram = int(ram_rows[0].get("FreePhysicalMemory") or 0) / 1024**2 if ram_rows else 0
    checks.append({"name": "ram_headroom", "ok": free_ram >= profile["free_ram_floor_gib"],
                   "detail": {"free_gib": round(free_ram, 2), "recommended_free_gib": profile["free_ram_floor_gib"]}})
    folder = Path(plan["directory"])
    existing = folder
    while not existing.exists() and existing.parent != existing:
        existing = existing.parent
    try:
        free_disk = shutil.disk_usage(existing).free / 1024**3
        disk_ok = free_disk >= profile["disk_budget_gib"]
        disk_detail = {"free_gib": round(free_disk, 2), "conservative_budget_gib": profile["disk_budget_gib"]}
    except OSError as exc:
        disk_ok, disk_detail = False, str(exc)
    checks.append({"name": "storage_headroom", "ok": disk_ok, "detail": disk_detail})
    # Strata 自己管理目录结构和下载；不根据猜测的内部文件路径断言缺少权重。
    if profile["engine"] == "llama-server":
        model_path = folder / profile["filename"]
        checks.append({"name": "model_file", "ok": model_path.is_file(), "detail": str(model_path)})
    return {"ready_for_engine_validation": all(check["ok"] for check in checks), "checks": checks,
            "inference_verified": False, "processes_stopped": False,
            "note": "若显存不足，请在原有 GPU 作业结束后重试。本工具不会结束其他进程。"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("scan", "plan", "preflight"))
    parser.add_argument("--model", choices=tuple(PROFILES), default="qwen36-35b")
    parser.add_argument("--data-dir", default=r"E:\LocalLLM")
    parser.add_argument("--binary", help="已安装的原生引擎路径；不会下载安装程序")
    parser.add_argument("--min-free-vram-mib", type=int, default=6144)
    parser.add_argument("--output", type=Path, help="将结果保存为 UTF-8 JSON")
    args = parser.parse_args()
    if args.action == "scan":
        result = scan()
    else:
        result = make_plan(args.model, args.data_dir, args.binary)
        if args.action == "preflight":
            result["preflight"] = preflight(result, scan(), args.min_free_vram_mib)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 2 if args.action == "preflight" and not result["preflight"]["ready_for_engine_validation"] else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
