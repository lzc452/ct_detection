# 开源选型报告：CT 缺陷检测能力补强

- 日期：2026-10-23
- 数据来源：GitHub REST API v3（`/repos/{owner}/{repo}`、`/releases/latest`、`/license`）、PyPI JSON API、各项目官方文档
- 抓取时间：**2026-10-23 10:50–11:05（UTC+8）**。Star / pushed_at 为快照值，复核时请重取
- 结论口径：匹配度只衡量"能否补上本项目的能力缺口"，许可证与维护状态单列为**准入闸门**，两者不混算

---

## 1. 需求侧：从代码里读出来的四条缺口

本项目技术栈（实测，非推测）：

| 项 | 实测值 | 来源 |
|---|---|---|
| Python | 3.10.11（便携 `runtime/python.exe`，固定不可升级） | `runtime/python.exe -V` |
| 框架 | ultralytics **8.4.146**、torch **2.8.0+cu128**、torchvision 0.23.0 | `pip list` |
| Web / 图像 / 报表 | Flask 3.1.3、opencv-python 4.8.1、pandas 2.3.3、openpyxl 3.1.5 | 同上 |
| 类别 | 0=damage（AR 4.13 长条）、1=dark、2=light | `core/Config.py` |
| 硬约束 | 推理 imgsz=1920；无 GPU 自动退 CPU；便携包整目录拷贝即用；中文路径走 imdecode | `AGENTS.md`、`README.md` |
| 已有自研 | `core/roi.py`、`core/model_spec.py`、`core/dataset_contract.py`、`scripts/kb.py`（LESSONS/FINDINGS/DECISIONS）、冻结划分清单 | `core/`、`scripts/` |

由此，外部项目能真正帮上忙的只有四处，其余（实验台账、数据契约、ROI 裁剪、评估口径）已经自研且比通用工具更贴合：

| 缺口 | 现状证据 | 需要外部能力 |
|---|---|---|
| **N1 小目标/长条召回** | 损伤 338 框、AR 4.13；test 损伤仅 73 框；V6 计划 P2 切片级联未做 | 切片/平铺推理，**零重训**可验证 |
| **N2 标注一致性与数据集审计** | P0-1 未做，且它决定天花板归属；93.9% test 泄漏是惨痛先例 | 错误分析台、近重复检测、可视化分桶 |
| **N3 稀疏类补充召回** | 损伤仅 9.6% 训练图、129 组、中位每组 2 张 | 无监督异常检测（做第二意见） |
| **N4 标注产能** | 现有 `labelImg 1.8.6`；损伤标注是样本量瓶颈 | AI 预标注 + 人工修正 |

明确**不需要**外部方案的部分（避免重复造轮子）：实验追踪、数据版本、Web 服务、Excel 导出、硬件回退——已由 `kb.py`、冻结清单、Flask、pandas、`core/device_utils.py` 覆盖。

---

## 2. 检索方法

三组关键词，GitHub 代码检索 + 官方文档核实，共 19 个仓库逐个取真实元数据（不做二手引用）：

1. `sliced inference small object detection` / `SAHI ultralytics tiled` → 切片推理
2. `industrial anomaly detection unsupervised patchcore` / `image dataset curation duplicate` → 异常检测与数据审计
3. `AI assisted labeling SAM auto annotation yolo` / `data versioning experiment tracking` → 标注与工程化

每个候选逐一核实：Star、Fork、`pushed_at`、Open Issues、SPDX License、`archived`、最新 release、PyPI 依赖下限。

---

## 3. 候选对比（按匹配度降序；同分按 Star 降序）

| # | 仓库 | Star | 匹配度 | 技术栈 | 最近提交 | 准入 | 核心命中 | 主要短板 |
|---|---|---|---|---|---|---|---|---|
| 1 | [obss/sahi](https://github.com/obss/sahi) | 5,558 | **96** | Python，框架无关；0.12.8(2026-09-30) | 2026-10-16 | ✅ MIT | 切片推理直打 N1，要求 ultralytics≥8.3.161（8.4.146 满足） | 不管训练/标注；跨块长条截断需自己处理 |
| 2 | [roboflow/supervision](https://github.com/roboflow/supervision) | 51,272 | **90** | Python，`sv.InferenceSlicer` + 检测工具链 | 2026-10-22 | ✅ MIT | 轻量切片（无 mmdet 依赖）+ 统一 Detections 口径 | 切片参数比 SAHI 少；不能替代评估脚本 |
| 3 | [visual-layer/fastdup](https://github.com/visual-layer/fastdup) | 1,915 | **82** | Python/C++，图像相似度与离群 | 2026-08-23 | ⛔ **CC BY-NC-ND 4.0** | 近重复/离群/泄漏检测，N2 强命中 | **禁商用、禁衍生**，不能进产线交付物 |
| 4 | [CVHub520/X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) | 10,770 | **80** | Python + ONNX Runtime，Windows exe | 2026-10-20 | ⚠️ GPL-3.0 | 预标注打 N4；原生支持 YOLO11/26、SAM、导出 YOLO txt | GPL 有传染性；不分发则无碍 |
| 5 | [voxel51/fiftyone](https://github.com/voxel51/fiftyone) | 11,216 | **78** | Python + MongoDB，`fiftyone.brain` | 2026-10-22 | ✅ Apache-2.0 | 错误分析台：按字段分桶、相似度、uniqueness | 依赖重（MongoDB），不能进便携包；694 open issues |
| 6 | [open-edge-platform/anomalib](https://github.com/open-edge-platform/anomalib) | 6,288 | **76** | PyTorch Lightning，PatchCore/EfficientAD | 2026-10-23 | ✅ Apache-2.0 | 无监督异常检测，打 N3 第二意见 | 需可信"正常样本"；最新 2.7.0 要 torch≥2.10 |
| 7 | [cleanlab/cleanvision](https://github.com/cleanlab/cleanvision) | 1,212 | **70** | Python，轻量图像体检 | 2026-01-08 | ✅ Apache-2.0 | 重复/模糊/尺寸异常/低信息量审计 | 9 个月无提交；不认电芯语义 |
| 8 | [cvat-ai/cvat](https://github.com/cvat-ai/cvat) | 16,979 | **66** | Django + Docker，标注平台 | 2026-10-22 | ✅ MIT | 标注 + QA + 团队协作 | 部署重，产线内网落地成本高 |
| 9 | [autodistill/autodistill](https://github.com/autodistill/autodistill) | 2,793 | **58** | Python，基础模型蒸馏到 YOLO | 2026-09-29 | ✅ Apache-2.0 | 伪标签扩样本 | 灰度 CT 细长目标上基础模型泛化存疑 |
| 10 | [open-mmlab/mmrotate](https://github.com/open-mmlab/mmrotate) | 2,177 | **30** | PyTorch/MMCV，旋转框 | **2024-09-28** | ⛔ 停更 2 年 | 理论上可解 AR 4.13 的框型误差 | 与 ultralytics 生态不兼容，MMCV 版本地狱 |

**准入被否决或重复投入，不进候选表**：`mlflow/mlflow`（28,502★，已有 `kb.py` 台账）、`treeverse/dvc`（15,939★，已有 `dataset_contract.json` + 冻结清单）、`HumanSignal/label-studio`（28,530★，重型平台，与 X-AnyLabeling+现有流程重叠）、`facebookresearch/sam2`（20,040★，仅作为 X-AnyLabeling 的内置引擎被间接使用）、`IDEA-Research/Grounded-Segment-Anything`（17,779★，**2024-09-05 后无提交**，已停更）。

---

## 4. 前四名逐项体检

### 4.1 SAHI — 匹配度 96

- **核心功能**：把大图切块 → 逐块推理 → 按 NMM/NMS 合并回全图坐标。官方口径是"framework agnostic sliced/tiled inference"，内置 `predict`/`coco slice`/`coco yolo`/`predict-fiftyone` 子命令，并带 COCO 误差分析图。
- **技术栈**：纯 Python，`ultralytics>=8.3.161`（本机 8.4.146 ✓），也支持 mmdet/HF/torchvision。MIT。
- **活跃度**：0.12.8 发布于 2026-09-30，最近提交 2026-10-16，仅 8 个 open issue → 维护紧凑、响应面干净。
- **适配性**：**不用改一行训练代码**，可在 CPU 上跑；能直接吃现有 `best.pt`。这是唯一能"今天下午就出对照结果"的候选。
- **与项目的差距**：① 只解决"看得到"，不解决"标得对"，对 N2/N4 零贡献；② 长条目标被块边界切断时，NMM 只能合并**重叠区内**的框，切开的两个半截框不会被拼回 → 损伤类恰好最容易踩这个坑；③ 切片数×单块推理，耗时线性上升，Web 端 3–5 秒的体验会破。

### 4.2 supervision — 匹配度 90

- **核心功能**：`sv.InferenceSlicer` 做平铺推理（`slice_wh`/`overlap_wh`/NMS·NMM·NONE 合并/多线程/`batch_size`），配套 `sv.Detections.from_ultralytics()` 统一结果结构，另有标注绘制、跟踪、指标工具。
- **技术栈**：MIT，51k Star，2026-10-22 仍在提交；官方文档明确给出 Ultralytics YOLO 的 callback 示例。
- **适配性**：比 SAHI 更轻（无 mmdet 生态），`batch_size>1 + thread_workers=1` 在 GPU 上更快。若 SAHI 与 ultralytics 8.4 出现签名冲突，这是首选替身。
- **与项目的差距**：切片参数比 SAHI 少（无 overlap 比例自动推导、无 COCO slice 工具）；OBB + 多线程会退化为串行；同样不解决跨块长条截断与标注质量。

### 4.3 fastdup — 匹配度 82，但**准入阻断**

- **核心功能**：亿级图像快速相似度聚类，找完全重复/近重复/离群/坏图，正是 N2 里"93.9% 泄漏"这类事故的探测器。
- **致命问题**：LICENSE 实为 **CC BY-NC-ND 4.0**（`raw.githubusercontent.com/.../LICENSE` 首行）；GitHub API 报 `NOASSERTION` 就是这个原因。NC=禁商业使用，ND=禁衍生作品。本项目是产线交付物，**不可用**。
- **替代路径**：FiftyOne 的 `compute_similarity` + `compute_uniqueness`（Apache-2.0），或 `imagehash` 感知哈希 + 自定义分组脚本。宁可慢一点，不留许可证尾巴。

### 4.4 X-AnyLabeling — 匹配度 80

- **核心功能**：桌面标注器，内嵌 ONNX/TensorRT 推理做 AI 预标注，支持 YOLO 全系（含 OBB）、SAM 1/2/3、Grounding DINO；导出 YOLO/COCO/VOC/DOTA。中文界面、Windows 原生 exe。
- **技术栈**：GPL-3.0，10,770★，2026-10-20 提交，仅 6 个 open issue（异常干净）。
- **适配性**：不需要装进 `runtime/`，标注员单独装 exe 即可；把 `best.pt` 导成 ONNX 塞进去，人工只做修正 → 直接抬高 N4 产能。
- **与项目的差距**：① GPL-3.0：仅内部工具不分发无传染，若随便携包分发则需法务评估（最稳妥是"标注工位专用，不进交付目录"）；② 它提升的是**标注速度**，标注**一致性**仍要靠审计流程（N2 不变）。

---

## 5. 可落地二次开发方案

### 方案 A（P0，主推）：SAHI 切片推理旁路

这是唯一同时满足"零训练成本、直接命中最大缺口、结论正负都有信息量"的方案（符合 `AGENTS.md` §8 的 Gate 精神：不该被标注审计拦住）。

**改造点**
- 新增 `scripts/sliced_infer.py`，只做推理旁路，**不动 `web/app.py`、不动训练链路**。
- 参数从模型 `_train_meta.json` 读取 `imgsz`/`roi_k`，保持 1920 画布口径，绝不新开一套隐式默认值（`LESSONS` L-012 同类风险）。
- 起始配置：`slice=960×960`、`overlap=192`（20%）、`postprocess=GREEDY_NMM`、`iou=0.5`、`batch=8`。
- 输出走现有 `scripts/eval_report.py` 同口径（FP/千张、组级漏检/170 组、损伤召回），禁止另造指标。

**预注册判据（写进 `FINDINGS.md`，状态=待验，先于开跑）**
- 主判据：组级漏检率相对 V5 **下降 ≥3pt**，且 FP/千张 **恶化不超过 10%**。
- 可证伪预测：亮斑/暗斑 AP 上升，损伤 AP **持平或下降**（因为切片切掉了 `R1` 指出的"上下文"）。若三类同时下降 → 上下文假设成立，切片路线当轮关闭。
- 损伤召回**只作方向参考**：test 仅 73 框，分辨率 ±11pt，不得作为通过条件。

**集成方式**：纯离线脚本 → 出对照报告 → 若达标再讨论是否进 Web（进 Web 时必须保留"全图/切片"双模式开关，默认全图，避免产线延迟突变）。

**风险**
1. 长条跨块截断：把 `overlap` 提到 ≥ 损伤中位长边，或在合并前按块边界做框拼接。这是本方案唯一的真实技术难点。
2. 延迟 3–6×：产线 Web 端不可直接替换。
3. 组级口径要重算：切片后同一电芯的框来源变多，去重规则必须显式写死，否则漏检率不可比。

**关闭依赖**：算力可关闭（CPU 半天出对照）。

---

### 方案 B（P0，并行）：FiftyOne 离线错误分析台

**改造点**：独立 venv（**不装进 `runtime/`**），导入 `reports/` 现有预测 JSON + `v5_split_manifest.csv`，按 `cell_id`、原生分辨率分桶查看 FP/FN 分布，用 uniqueness 查近重复（替代 fastdup）。

**集成方式**：只读分析，产出是"错题清单"（对标 `docs/错题清单_V3_20261016.csv`），不参与推理。

**风险**：MongoDB 依赖 + 安装 1–2 GB；`pymongo 4.9`/pydantic 与现有 numpy 1.26.1 可能冲突 → 必须隔离环境，绝不 `pip install` 进便携 runtime。

**关闭依赖**：算力可关闭（0 GPU，读盘即可）。

---

### 方案 C（P1，有前置阻断）：anomalib 无监督第二意见

**前置阻断（必须先拍板）**：PatchCore/EfficientAD 需要可信的"正常样本"。而项目里 `is_neg=1` 是从清单**声明**继承的，不是复核真值（`README.md` 明确警告）。在 P0-1 标注审计完成前，训练集"正常"标签不可信，**此时上异常检测等于把噪声当基准**。

**若前置解决**：pin `anomalib==2.2.x`（`torch>=2.4`，兼容本机 2.8.0）。**不要装 2.7.0**——官方 README 已要求 torch≥2.10，会撞死便携环境。产出为"电芯级异常分数"，仅用于复核排序，**不产出框、不进 Web 链路**。

**关闭依赖**：判断可关闭（需要人拍板正常样本真值归属）。

---

### 方案 D（P1，人力线）：X-AnyLabeling 预标注流水线

**改造点**：`best.pt` → `yolo export format=onnx imgsz=1920 opset=12` → 写自定义模型配置 → 标注员修正 → 导出 YOLO txt → 复用**已有**的 `scripts/import_new_labels.py` 入库。

**风险**：GPL-3.0 只允许内部工具用法；ONNX 在 1920 下体积/内存要实测（CPU 机器可能吃不住）。

**关闭依赖**：人力可关闭（标注作业量）。

---

### 方案 E（P2，兜底）：supervision `InferenceSlicer`

仅在方案 A 出现 SAHI 与 ultralytics 8.4 的兼容问题时启用；顺带可用 `sv.Detections` 统一评估口径。不单独排期。

---

## 6. 优先级与结论

| 优先级 | 动作 | 成本 | 关闭依赖 | 前置 |
|---|---|---|---|---|
| **P0** | A：SAHI 切片旁路 A/B | CPU 半天 | 算力 | 预注册判据 |
| **P0** | B：FiftyOne 审计台 | 1 人日 | 算力 | 隔离 venv |
| P1 | D：预标注提产能 | 持续 | 人力 | 无 |
| P1 | C：anomalib 第二意见 | 1 周 | 判断 | 正常样本真值 |
| P2 | E：supervision 兜底 | 2 小时 | 算力 | A 失败 |

**一句话结论**：只做一个，就做 **SAHI 切片推理旁路**——零训练成本、直接打损伤/小目标这个最大缺口、结论无论正负都能关掉一条路线。**最该避开的坑是 fastdup**：Star 不高但功能极对口，许可证却是 CC BY-NC-ND 4.0，进产线就是法务问题。

---

## 7. 未采纳项的明确理由

| 项目 | 理由 |
|---|---|
| MLflow / DVC | 能力重复：`kb.py` + `FINDINGS/DECISIONS` + `dataset_contract.json` + 冻结清单已覆盖，且更贴合"单变量 + 预注册"纪律 |
| Label Studio | 平台化部署（Docker/DB），与现有 labelImg→复核→入库流程冲突，重构成本高于收益 |
| mmrotate | 2024-09-28 后停更，MMCV 与 ultralytics 生态不兼容；要做长条框型应走 **ultralytics 内置 OBB**，无需外部项目 |
| Grounded-SAM | 2024-09-05 后停更；且开放词汇模型对灰度 CT 细长缺陷的零样本能力未经验证 |
| autodistill | 在 N4 上与 X-AnyLabeling 重叠，但引入"基础模型猜错→污染标注"的风险，需先小样本验证再谈 |
| cleanvision | 与 B 方案的重复/坏图检测功能重叠，且其"过暗/过亮"启发式对 CT 无意义；可作 B 的补充脚本，不单独立项 |
