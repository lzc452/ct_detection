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
| `开始训练.bat` | 训练新模型（只列有完整契约的 `datasets_v*` 版本）       |
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

1. 先审阅三本实验台账，按电芯/批次维护冻结清单；用 scripts/materialize_dataset.py 创建独立版本。物化只硬链接原图，禁止跨卷复制回退；已有输出即使传 --force 也不删除。原始 datasets/ 不作为新入口的可写派生目录。
2. 双击 开始训练.bat，或使用 scripts/train_run.py；两者默认1920并共用配置、严格preflight、日志、元信息与归档。旧无 data_contract.json 的数据集不会被猜测放行，须从冻结清单创建新版本；本轮没有自动重建生产数据。
3. 必须指定新 --name 与 --baseline-args。先加 --dry-run 输出计划指纹和配置差异（不会初始化模型/GPU）；除输出位置等非实验项外，变化超过1项拒绝。基线与默认值不符时，按差异显式指定参数，不把默认值当作已复制基线。
4. 用 kb.py add-finding --status 待验 --run 新实验名 登记预测、判据，并在 --evidence 中加入 dry-run 的 plan_sha256=...。再以同参数加 --finding F-XXX 开训。该门槛验证登记与配置绑定，判据是否有业务意义仍须实验设计审查。
5. 开训前自动执行 kb.py preflight，核对数据/标签哈希、类别、坐标及独占训练槽；异常也保留 failed 元信息和日志。--cutmix 会真实传入；--erasing 和无segments的 --copy-paste 不能作为本框检测任务的新干预。--resume-from 是微调来源，不恢复优化器；推荐写 --fine-tune-from。
6. 产物在 runs/detect/CT_pro/实验名/。通过正式验收后，人工把 weights/best.pt **连同绑定该权重的 _train_meta.json** 放入 models/启用/版本名/。未知无meta权重将被Web拒绝；旧1920兼容只限 core/model_spec.py 登记的三个既有权重SHA256。是否发布、切换或重启由人决定，代码不会自动执行。

数据契约只保证文件/几何/划分一致，不会把未复核图像变成工艺真值。母本缺标签且is_neg=1继承的是清单声明；没有独立真值时不能宣称已确认真实正常。高阈值收割还应补随机电芯、低置信与模型分歧抽检，再标注、分组与物化；不要沿旧按图随机切分流程复用test调参。当前遵守D-010，不再开ROI训练。

详细命令、兼容边界与验证结果见 [第四阶段报告](docs/ct_audit_stage4_20261023.md)。

## 关键约定

- **推理分辨率 1920**：必须与训练分辨率一致，640 下几乎检不出缺陷（2026-10-14 实测，见 `web/app.py`）。
- Windows 下需保证 `PROCESSOR_ARCHITECTURE` 环境变量存在，否则 py-cpuinfo 崩溃（代码已内置兜底）。
- 中文路径读写图片用 `imdecode/imencode`（见 `scripts/harvest.py` 的 `imread_cn/imwrite_cn`），OpenCV 原生 `imread` 不支持中文路径。
- 搬到新电脑：整个文件夹原样拷贝即可，`runtime/` 不能少。

## 技术栈

Python（便携 runtime）· Ultralytics YOLO · Flask · OpenCV · PyTorch · pandas/openpyxl

## 版本说明

本仓库用 Git 管理代码与文档；原始数据集、产线图片、模型权重、训练记录等大文件已通过 `.gitignore` 排除，不进版本库。
