# AGENTS.md — AI 协作指南

本文件面向在本仓库工作的 AI 编码助手，说明项目结构、运行方式与必须遵守的约定。

## 项目概述

电芯 CT 图像缺陷检测系统（YOLO / Ultralytics + Flask Web 端），便携版打包，目标环境为 **Windows**。
缺陷三类：`0=damage 极片损伤`、`1=dark 暗斑`、`2=light 亮斑`（定义在 `core/Config.py`，中英文映射必须保持一一对应）。

## 运行方式

- **不要自己安装 Python**：项目自带便携环境 `runtime/python.exe`，所有脚本用它运行。
  - 检测服务：`runtime/python.exe web/app.py`（或双击 `启动检测.bat`），端口 5000。
  - 训练：`runtime/python.exe scripts/train.py`（交互式选数据集）。
  - 收割：`runtime/python.exe scripts/harvest.py <源文件夹> [阈值]`。
- 在 Git Bash 中执行时，注意 Windows 路径与中文路径的转义。

## 目录职责

| 路径 | 职责 | 备注 |
| --- | --- | --- |
| `core/` | 全局配置 `Config.py`、预处理、设备选择 | 改配置先看这里 |
| `web/` | Flask 检测 Web 端，入口 `app.py` | 模板/静态文件在子目录 |
| `scripts/` | 训练、数据集构建、增强、收割、评估脚本 | 每个脚本头部有 docstring 说明用法 |
| `models/启用/` | 网页可选模型，`<名字>/best.pt` | 只放人工确认过的模型 |
| `models/pretrained/`、`weights/` | 基础权重 | 勿动 |
| `datasets/` | YOLO 数据集（含 `data.yaml`），按版本命名如 `CT_DATASET_FINAL_V4` | 大文件，已 gitignore |
| `runs/` | 训练输出 | 已 gitignore |
| `runtime/` | 便携 Python 环境 | **严禁修改/删除**，已 gitignore |

## 硬性约定（踩过的坑）

1. **推理 `imgsz` 必须为 1920**，与训练分辨率一致；640 下几乎检不出缺陷（2026-10-14 实测，`web/app.py`）。
2. **中文路径图片读写**必须用 `cv2.imdecode(np.fromfile(...))` / `cv2.imencode(...).tofile(...)`（参考 `scripts/harvest.py` 的 `imread_cn/imwrite_cn`），OpenCV 原生 `imread/imwrite` 遇到中文路径会静默失败。
3. Windows 环境变量 `PROCESSOR_ARCHITECTURE` 缺失会导致 ultralytics 依赖的 py-cpuinfo 崩溃；入口脚本已 `os.environ.setdefault` 兜底，新入口也要加。
4. `Config.py` 中是**相对路径**，入口脚本启动后需 `os.chdir(PROJECT_ROOT)`（参照 `web/app.py`、`scripts/train.py`）。
5. `.bat` 文件为 **GBK/ANSI 编码**，编辑时保持编码不变，否则中文乱码。
6. 源文件头部多为 `#coding:utf-8` / `# -*- coding: utf-8 -*-`，保持 UTF-8 编码与既有注释风格（中文注释）。

## 训练与实验约定

**开训练之前必须先读这三处，别凭记忆动手：**

| 想知道 | 去哪 |
| --- | --- |
| 这个报错/现象见过没 | `experiments/LESSONS.md`（踩坑库，18 条） |
| 某条结论还能不能当依据引用 | `experiments/FINDINGS.md`（结论台账，**已证伪的永不删除**） |
| 当初为什么这么定 | `experiments/DECISIONS.md`（决策日志，含复核触发条件） |

1. **开训前跑一次自检**：`runtime/python.exe scripts/kb.py preflight --data <data.yaml> --imgsz <画布>`。它会检查数据集、画布对齐、meta 的 `roi` 位是否与实际数据一致，并列出相关坑和**已证伪结论**。
2. **判据先预注册**（`DECISIONS` D-003）：训练启动前把门槛和可证伪预测写进 `FINDINGS.md`（状态=待验），结束后回填状态，不做结果回填式解释。
3. **一次只动一个变量**（`LESSONS` L-008）：开跑前 diff 两轮的 `args.yaml`，差异超过 1 项就不许开。
4. **新增坑/结论/决策用 CLI 记**，别只写在 commit message 里：
   `runtime/python.exe scripts/kb.py add-lesson|add-finding|add-decision ...`
   写完跑 `runtime/python.exe scripts/kb.py check` 校验格式。
5. **选型只看层 2/3**（`predict` 口径的 FP/千张、组级漏检、损伤召回），层 1（`model.val` 的 COCO mAP）只用于横向比（`DECISIONS` D-005）。
6. 生产模型 = **V5 全图@1920**（`DECISIONS` D-001）。换模型必须过 §4 预注册判据，代码不得自动上线。

## 数据与安全红线

- `datasets/`、`积攒/`、`练兵/`、`收割结果_*/`、`复核_*/`、`问题电芯图片汇总/` 等为原始生产数据，**体积巨大（数百 GB）且已 gitignore**：不要复制、移动、删除，不要递归遍历统计（`du` 会超时），按需精确访问。
- 模型权重 `*.pt` 不入库；新模型必须人工确认后才复制进 `models/启用/`，代码不得自动上线模型。
- 不要改动 `runtime/`、`models/pretrained/`。

## 代码风格

- Python 3，中文 docstring/注释，脚本开头写明功能与用法（参照现有 `scripts/*.py`）。
- 新增命令行脚本放在 `scripts/`，支持从项目根目录以相对路径运行。

## 验证方式

- 改 Web 端：用 `runtime/python.exe web/app.py` 启动后访问 `http://127.0.0.1:5000` 自测，验证完关闭进程。
- 改训练/数据脚本：先用小目录或小样本跑通，不要直接对全量数据操作。

## git代码提交规范

- 默认不要擅自创建分支和worktree, 只在当前分支或者worktree进行改动
- 每次改动都需要提交commit, 并推送远程仓库
- 可动态修改.gitignore来排除大文件夹和数据集进入git