# CT 缺陷检测系统

基于 YOLO（Ultralytics）的电芯 CT 图像缺陷检测系统，便携版打包：整个文件夹拷贝到任意 Windows 电脑即可双击使用，无需安装 Python。

## 缺陷类别

| ID | 英文（YOLO 内部） | 中文（界面显示） |
|----|------------------|------------------|
| 0  | damage           | 极片损伤          |
| 1  | dark             | 暗斑             |
| 2  | light            | 亮斑             |

类别定义见 `core/Config.py`。

## 快速开始

根目录 3 个开关，双击即用：

| 脚本           | 用途                                                |
| -------------- | --------------------------------------------------- |
| `启动检测.bat` | 日常检测，浏览器自动打开 <http://127.0.0.1:5000>     |
| `开始训练.bat` | 训练新模型（自动扫描 `datasets/` 让你选数据集）       |
| `检查环境.bat` | 环境体检，出问题时先点它                              |

另有 `产线收割.bat`：用当前 V4 模型高阈值跑海量产线图，导出高置信损伤原图供人工核对与标注（拖拽源文件夹到 bat 上，或双击后输入路径）。

> 有可用显卡自动用 GPU，老显卡/无显卡自动退回 CPU（`core/device_utils.py`）。

## 目录结构

```
CT Detection/
├─ core/            全局配置(Config.py)、CT图像预处理、设备选择
├─ web/             Web 检测系统（Flask，app.py 为入口）
├─ scripts/         训练、数据增强、收割、评估等工具脚本
├─ models/
│  ├─ 启用/         网页可选模型（人工确认后手动放入，子文件夹/best.pt）
│  └─ pretrained/   训练用基础模型（勿动）
├─ datasets/        训练数据集（YOLO 格式，含 data.yaml；大文件，不入库）
├─ runs/            训练记录（每次训练一个独立文件夹）
├─ runtime/         便携 Python 运行环境（勿动、勿删，不入库）
├─ weights/         基础权重
├─ 积攒/            产线积攒的原始图（数百 GB，不入库）
├─ 练兵/            练兵数据（不入库）
├─ 收割结果_*/      收割机导出结果
├─ 复核_*/          人工复核结果
├─ docs/ logs/ tools/  文档、日志、工具
```

## 工作流

**日常检测**：双击 `启动检测.bat` → 等 10–20 秒 → 浏览器自动打开。支持单张 / 批量文件夹 / 视频上传，结果可导出 Excel。

**训练新模型**：

1. 双击 `开始训练.bat`，按提示选数据集、起实验名
2. 产物在 `runs/detect/CT_pro/<实验名>/`（`weights/best.pt` + 曲线报告）
3. 确认效果后，把 `best.pt` 手动复制到 `models/启用/<名字>/best.pt`
4. 重启 Web 端即可在模型列表选用

> 网页可选模型只来自 `models/启用/`，训练结果不会自动上线。

**数据闭环（收割 → 标注 → 训练）**：`产线收割.bat` 高阈值筛出产线疑似损伤原图 → 人工复核（`复核_*`）→ X-AnyLabeling 标注（`tools/`）→ `scripts/build_final_dataset.py` 构建新版本数据集 → 重新训练。

## 关键约定

- **推理分辨率 1920**：必须与训练分辨率一致，640 下几乎检不出缺陷（2026-10-14 实测，见 `web/app.py`）。
- Windows 下需保证 `PROCESSOR_ARCHITECTURE` 环境变量存在，否则 py-cpuinfo 崩溃（代码已内置兜底）。
- 中文路径读写图片用 `imdecode/imencode`（见 `scripts/harvest.py` 的 `imread_cn/imwrite_cn`），OpenCV 原生 `imread` 不支持中文路径。
- 搬到新电脑：整个文件夹原样拷贝即可，`runtime/` 不能少。

## 技术栈

Python（便携 runtime）· Ultralytics YOLO · Flask · OpenCV · PyTorch · pandas/openpyxl

## 版本说明

本仓库用 Git 管理代码与文档；原始数据集、产线图片、模型权重、训练记录等大文件已通过 `.gitignore` 排除，不进版本库。
