# 本机配置清单与三条 Qwen 本地部署路线

查证日期：2026-11-01。硬件快照：`local_llm_hardware_20261101.json`，22:51（北京时间）。

结论：本机具备以 CPU/GPU 混合方式运行三种模型量化版的资源基础。优先验证 Qwen3.6-35B-A3B UD-Q4_K_XL；Qwen3.8-27B UD-Q4_K_M 用作另一条质量与速度对照路线；Flash-Next IQ2_XS 为更吃内存、显存和带宽的实验路线。这是容量与上游兼容性评估，**尚未完成本机模型装载、生成或速度验收**。

## 1. 实测配置

| 项目 | 本机结果 | 证据与说明 |
| --- | --- | --- |
| CPU | Intel Core i7-14700F，20 核 / 28 逻辑处理器 | WMIC 与系统信息一致 |
| 内存 | 64 GiB，金士顿 32 GiB × 2 | KF3600C18D4/32GX；系统可见约 63.8 GiB |
| 内存当前速度 | 2400 MT/s | 两根的 ConfiguredClockSpeed、Speed 均为 2400；不能把料号中的 3600 当作当前运行速度 |
| 主板 | Gigabyte B760M D2H DDR4 | WMIC 与系统信息一致 |
| 显卡 | **NVIDIA GeForce RTX 5060，8 GB 档** | nvidia-smi 报告 8151 MiB；用户补充为 4060 8G，以实测型号为本报告依据 |
| 显卡驱动 | 616.56，WDDM | nvidia-smi；CUDA UMD Version 13.4 是驱动支持信息，不代表已安装 CUDA Toolkit |
| 扫描时显存 | 已用约 7550 MiB，仅余约 347 MiB | 动态快照；不足以同时启动本报告的大模型路线 |
| 扫描时空闲物理内存 | 约 41.6 GiB | WMIC FreePhysicalMemory；会随现有作业变化 |
| 物理磁盘 | KINGSTON SNV3S2000G，约 2 TB，NVMe SSD | MSFT_PhysicalDisk：BusType=17、MediaType=4、HealthStatus=0 |
| 系统 | Windows 10 专业版，64 位，19045 | 系统信息与 WMIC |
| 项目 Python | 便携 Python 3.10.11 | 使用 runtime/python.exe；本次未安装 Python 或修改便携环境 |
| 现有 LLM 工具 | PATH 未找到 strata、llama-server、ollama | 不等价于全盘没有这些软件，本次没有遍历全盘找安装目录 |

| 分区 | 总容量（GiB，约） | 空闲（GiB，约） |
| --- | ---: | ---: |
| C: | 300.0 | 144.7 |
| D: | 782.0 | 164.1 |
| E: | 780.7 | 587.3 |

用户补充的约 1 TB 可用空间与三个分区空闲总量大体相符。模型建议放 `E:\LocalLLM`，避开 CT 仓库、数据集和 runtime。已枚举到一块物理盘；把模型从 D 换到 E 可改善空间安排，不代表得到独立磁盘的 I/O 带宽。

## 2. Qwen3.8-27B：优先以 GGUF + llama.cpp 部署

这是 27B **稠密模型**，不能套用 MoE 的“仅激活 3B/6B”计算量，也不能向它传 `--cpu-moe` 来解决内存问题。Qwen 官方模型卡说明其继承 Qwen3.5 架构，并支持视觉与可控思考。

| Unsloth GGUF | 发布文件大小（十进制 GB，约） | 本机用途 |
| --- | ---: | --- |
| UD-IQ2_XXS | 7.27 | 权重接近显存容量，缓存和工作区仍需空间；不作为质量基线 |
| UD-IQ3_XXS | 10.9 | 仍超过 8GB 显存，通常需要混合卸载 |
| UD-Q3_K_XL | 13.1 | 后续速度/质量折中对照 |
| **UD-Q4_K_M** | **16.5** | 首选验证档，CPU RAM + GPU 自动分层 |
| UD-Q4_K_XL | 17.6 | Q4 的另一候选，额外容量不代表本机任务必然更好 |
| UD-Q5_K_M | 19.8 | Q4 基线完成后再比较 |
| Q8_0 | 29.0 | RAM 能容纳权重，但不适合以当前 8GB 显存作为初次性能目标 |

文件大小不是运行显存。还需 KV、循环状态、计算工作区、驱动和桌面余量。64GB RAM 可以支撑 Q4 混合部署的容量需求；CPU 侧稠密计算与 DDR4 当前速度可能成为瓶颈，不能据此预报 tokens/s。

适配起点：Windows 原生、适配 RTX 50 的 llama.cpp CUDA 发行包；8K 上下文、1 并发、自动 GPU 层数、预留 1024 MiB、8 个 CPU 工作线程、较小 batch。先不用 MTP 和视觉投影，稳定后每次只改一个参数。工具版本需支持生成命令中的 `--gpu-layers auto` 和 `--fit-target`，安装后先核对 `--version`、`--help`。

Strata 当前主线模型目录未列 Qwen3.8-27B，不能从 Flash-Next 的支持推断出对稠密 27B 的支持。此路线用 llama.cpp。LM Studio/Ollama 可作图形界面或管理层的后续选择，不能据应用名称推断其内置引擎版本已经支持本模型。

来源：[官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)、[实际量化文件及大小](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/tree/main)、[llama.cpp server 参数](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)。

## 3. Qwen3.6-35B-A3B：优先验证 Strata 的 Q4 + MTP 路线

35B 总参数、每 token 约 3B 激活的 MoE；A3B 影响每步计算量，**全部专家仍需储存在 RAM/VRAM/磁盘中**。本机 64GB RAM 的优势在于能容纳量化专家，8GB GPU 负责其能容纳的权重与计算，其余由 CPU 完成。

| Strata 主线列出的量化档 | GGUF 大小（GB，约） | 上游安装器规划的系统 RAM | 本机建议 |
| --- | ---: | ---: | --- |
| **UD-Q4_K_XL** | **22.9** | **32 GB** | 第一条基线，选含 MTP 的仓库 |
| UD-Q6_K_XL | 32.6 | 44 GB | 容量可研究，待 Q4 验收后做质量对照 |
| Q8_0 | 37.8 | 52 GB | 当前还运行其他任务，不优先 |

上述 RAM 数是上游对系统容量的规划，不是实测常驻内存、空闲内存门槛或保证可启动。适配脚本中的 30 GiB 空闲 RAM 检查是本机首轮的保守预算。

上游还发布 UD-Q3_K_M（17.1GB）、UD-Q2_K_XL（12.6GB）、UD-IQ3_XXS（14.1GB）等更小文件；这不表示 Strata 主线菜单接受所有这些格式。若要研究更低位量化，可用支持该架构和量化的 llama.cpp；PolyStrata 的独立 Qwen3.6 路线也列出 Q3/Q2，需单独核验对应引擎与文件，不能混用两套启动参数。当前 64GB RAM 无需仅为“装下模型”先降到 Q2。

本机起点：`--family qwen36 --model UD-Q4_K_XL --context 8192 --parallel 1 --vision off`，监听 `127.0.0.1:8081`。若要对照 llama.cpp，比较相同 GGUF、相同上下文和提示词；先让 MoE 专家主要留在 CPU，可用其 `--cpu-moe`，随后仅调整 CPU 专家层数。该参数只用于此 MoE 路线。

上游有 RTX 3060 12GB 的 Q4 混合推理记录，也明确讨论了剩余显存约 7.5/5.5 GiB 时更小 prompt chunk。这提供兼容性研究依据，**不能把 12GB 卡的速度当成本机 8GB 卡的结果**。本机当前内存频率、CPU、GPU缓存容量、工作负载都不同。

视觉后续再启用：对应的 `mmproj-F16.gguf` 约 0.9GB；8GB 卡优先研究 CPU 视觉编码。CT 图片的具体缺陷判断仍需专门验证，不由通用模型部署可行性推出。

来源：[官方模型卡](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)、[MTP 量化文件](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/tree/main)、[Strata 模型目录](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)、[Qwen3.6 引擎说明](https://github.com/Niko1221/Strata/blob/main/docs/QWEN36.md)、[PolyStrata 模型目录](https://github.com/VecSzn/PolyStrata)。

## 4. Qwen3.8-Flash-Next：8GB 可以试验，优先级低于前两条

Strata 的 ISTA-DASLab 原版路线将专家放 RAM/GPU，将约 29GB n-gram 表留在 SSD。125B/6B 是主模型总量/激活量口径，不能据激活 6B 认为它只需 6B 权重的容量；官方另有大规模 n-gram 参数。

| 原版量化档 | 上游专家内存口径（GB，约） | 本机起点判断 |
| --- | ---: | --- |
| Q2_0 | 34 | 更小的备选 |
| **IQ2_XS** | **35.5** | 初次研究档；总下载约 68GB，另需 MTP/准备空间 |
| IQ3_XXS | 43 | 当前空闲内存不宽裕，后续再考虑 |
| IQ3_S | 50 | 不作为本机初次启动默认 |

这些数不是完整进程内存，需另留系统和辅助缓冲空间；项目自己的不同表格分别使用专家、RAM+VRAM和下载口径，本报告不将它们相加成一个精确实测值。为初次 IQ2_XS 留出约 120 GiB 磁盘预算、44 GiB 空闲 RAM 是本报告的保守建议，不是硬件最低要求。

上游 PR #2004 有 RTX 4060 8GB + 32GB RAM + Windows 10、IQ2_XS 的真实运行报告（旧引擎、低 RAM 模式），因此“8GB 绝对不能跑”不成立。那份报告短提示 decode 20.7 tok/s，同时长 prompt 有重分页与卡顿；本机是实测 RTX 5060、64GB RAM，不能把该数字移植为本机承诺。

适配：先 8K、单并发、纯文本，保持中文使用所需的完整词表，不为了省显存直接套英文 draft vocabulary。Windows 读取设备真实空闲显存、预留约 1GB，保持专家路由不变：`STRATA_ROUTE_PRIOR=0`。当前文档说明 0.2.8 已撤回小卡自动启用路由偏置的默认值；此处显式写 0 是为了固定质量基线。不要用优先选驻留专家的近似路由来冒充相同模型的纯性能提升。

来源：[官方 Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)、[Strata 内存口径](https://github.com/Niko1221/Strata/blob/main/docs/MODELS.md)、[引擎与 Windows 显存说明](https://github.com/Niko1221/Strata/blob/main/docs/DETAILS.md)、[8GB 显卡运行报告 PR #2004](https://github.com/Niko1221/Strata/pull/2004)。

## 5. 已完成的适配和使用方式

`scripts/local_llm.py` 只用项目便携 Python 和系统只读查询。没有模型下载、引擎安装、推理启动或结束进程的代码。

```text
runtime\python.exe scripts\local_llm.py scan --output docs\local_llm_hardware_20261101.json
runtime\python.exe scripts\local_llm.py plan --model qwen38-27b
runtime\python.exe scripts\local_llm.py plan --model qwen36-35b
runtime\python.exe scripts\local_llm.py plan --model flash-next
runtime\python.exe scripts\local_llm.py preflight --model qwen36-35b
```

`plan` 输出具体引擎命令、权重名称、存储路径和预算，但不执行。Strata 路线同时输出上游 `--plan --output json` 命令。`preflight` 查询当前资源，未达预算时退出码为 2；不是训练脚本 kb.py 的自检，也不代替真实推理验收。

Strata 路线应先查看生成的上游计划，再按计划准备引擎和模型。查证到的最新发行版为 Strata app 0.2.9、Control/engine 0.1.55（2026-10-31 发布）；部分安装文档标题仍写 0.2.5，应以发行说明和安装后的 `--version`、`--help` 为准。CUDA 13 包要求 NVIDIA 驱动 580+，本机 616.56 在这个范围内，但仍需对具体包做实际装载检查。

官方 Terminal 安装器会提供私有 Python。本次遵照仓库便携环境约定，未执行它、未执行 pip、未改 runtime；若实际采用该安装方式，需明确它自带环境的位置和管理边界。llama.cpp 原生发行包不需要另装 Python。两条路线都优先使用发行包，不因探索而安装编译器、CUDA Toolkit、WSL 或 Docker。

本机目前未在 PATH 找到推理引擎，且已有 GPU 进程占用大部分显存，所以**尚未安装和验证本地服务**。不得把生成了启动命令、下载成功或进程退出码 0 记为部署成功。

来源：[Strata 0.2.9 发行说明](https://github.com/Niko1221/Strata/releases/tag/v0.2.9)、[安装要求](https://github.com/Niko1221/Strata/blob/main/docs/INSTALL.md)、[Terminal 启动参数](https://github.com/Niko1221/Strata/blob/main/docs/TERMINAL_LAUNCH.md)、[AI 安装与验收说明](https://github.com/Niko1221/Strata/blob/main/docs/AI_SETUP.md)。

## 6. 后续验证与关闭依赖

| 优先级 | 任务 | 关闭依赖 | 当前状态 |
| --- | --- | --- | --- |
| P0 | 硬件清单、三份参数方案、资源预检 | 算力可关闭 | 已完成，无推理负载 |
| P0 | Qwen3.6 Q4：装载、真实生成和基本稳定性 | 算力可关闭 | 待现有 GPU 作业释放资源，并准备引擎/权重 |
| P1 | Qwen3.8-27B Q4：相同提示集对照 | 算力可关闭 | 方案已生成，未实跑 |
| P1 | Flash-Next IQ2_XS：小显存实验 | 算力可关闭 | 方案已生成，未实跑 |
| P1 | 根据中文/代码任务的质量与延迟选默认模型 | 判断可关闭 | 待三条路线的本机结果 |

首轮每条路线独立运行，保持相同 8K 上下文、1 并发、纯文本输入，记录冷启动、1K/4K prompt 的首 token 延迟、decode tokens/s、RAM/VRAM 峰值。使用模型官方推荐的采样方式，分别记录思考与最终回答，避免只统计 decode 而掩盖长思考延迟。先做 5 个中文任务、5 个代码任务及少量结构化输出/工具调用，随后再评估长上下文与视觉。

建议先登记的工程验收目标（本报告建议，不是用户已定 SLA）：无 OOM、无持续分页卡顿、重复多轮后无故障；4K prompt 首次响应不超过 60 秒，稳定输出不低于 10 tok/s，并且中文/代码内容达到实际用途要求。若速度达标但质量下降，不应仅按 tokens/s 选型。

真正服务验收时访问 `http://127.0.0.1:8081/health`、`/v1/models`，再发送真实 `/v1/chat/completions` 请求。单独验证中文、连续多轮、工具调用和思考字段；全部通过后才记录为本机部署完成。
