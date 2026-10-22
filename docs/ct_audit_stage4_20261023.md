# CT系统审阅：第四阶段训练、数据与规格门槛（2026-10-23）

关闭依赖：**算力**。本阶段关闭新训练/物化入口的可执行约束与Web未知规格回退。没有开真实训练、全量重建数据、复制原始生产图或修改现有服务；验证全部使用三张合成图、沙盘目录与假训练器。

## 1. 有代码依据的缺陷与修复

基线提交75974e3；以下旧行号均指该提交。代码可达风险不等同已经发生过生产事故。

| 问题与基线证据 | 修复位置（本阶段） | 可复核结果 |
|---|---|---|
| materialize_dataset.py:48-50、materialize_roi_dataset.py:73-81将用户root/name拼接后直接rmtree；原root/绝对name可绕出工程 | core/path_safety.py:11/24/40；core/materialize.py:42 | Windows名称、解析后路径/junction检查；仅工程datasets_v*或.cache派生目录；已有目录一律拒绝，--force不再删除。沙盘外部junction和用户哨兵文件均保留 |
| materialize_dataset.py:26-34硬链接失败退回复制；65-75缺图跳过、缺标默认为空；ROI脚本107-131只警告且忽略编码失败，148递归统计输出 | core/materialize.py:19/42；两个scripts/materialize_*入口共用实现 | 跨卷链接失败直接退出；按CSV命名样本精确读图与MD5/尺寸校验；缺标正例、未声明负例、坏JSON/类别/坐标、ROI无效/丢GT/编码失败均拒绝；不遍历生产母本、不计算递归目录体积 |
| kb.py:315-333把缺label.txt当0框；343-369依靠正则与固定布局，既不验类别顺序/坐标/内容漂移，也不读真实split配置 | core/dataset_contract.py:15/37；scripts/kb.py:315 | 新物化器最后生成data_contract.json，含输入标签哈希、图片身份、split/group、变换与逐产物SHA256；preflight核对YAML、图片/标签一一对应、哈希、类别、有限且不退化的框；显式空标签文件才是声明的负样本 |
| train.py:89默认1280、144-175独立训练且无完整meta；train_run.py:209接收cutmix但291-312没转发；204-205误称检测erasing生效 | scripts/train.py:58/62；scripts/train_run.py:131/186 | 两入口委托同一执行链，默认1920；cutmix进入实际train kwargs并经过已安装Ultralytics get_cfg校验；erasing/无segments CopyPaste保留历史参数，但拒绝为其变化启动无效检测实验 |
| preflight/预注册/配置对照原先只靠提醒；ROI规格还由目录名_M15猜测 | core/training_contract.py:28/48/62/80；scripts/train_run.py:186/256/278 | dry-run不创建模型/GPU；待验F条目必须关联run并登记精确plan_sha256；多于1项实验配置变化拒绝，device/workers亦参与比较，路径表示规范化；正式开训确实运行kb preflight、核对输入未变、检查已有训练并占独占锁；ROI由数据契约确定，D-010直接禁止新ROI训练 |
| train_run.py:263预算run路径，360才写meta；异常绕过日志收尾；只按预算路径归档 | scripts/train_run.py:111/249/278 | 独占新run且不覆盖；构造模型前写starting，回调读取trainer.save_dir/args/batch/accumulate，实际目录不符直接失败；best取trainer.best并校验包含关系；训练/构造异常写failed且还原原stdout/stderr、关闭并移除两处日志handler；归档失败另记archive_status，不谎报训练失败 |
| Web缺meta仍对所有未知权重回退全图1920，存在meta也没有验证其绑定的权重身份 | core/model_spec.py:12/27；web/app.py的_load_model_spec委托此模块 | 未知无meta权重拒绝；meta须绑定SHA256或历史MD5，拒绝failed/running状态和错误类别；ROI必须显式roi_k。仅三个原人工启用模型按精确SHA256保留旧1920兼容，不能用同文件名替换权重绕过 |
| dataset_split.py:141裸json.load(open()) | core/dataset_split.py:141 | 首轮沙盘实测ResourceWarning暴露句柄未关闭，改with后复验无此告警，数值转换未改 |

数据完成标记使失败半成品不能通过新训练入口；失败目录保留诊断，不擅自删除。全图只建硬链接，标签为独立生成文件；ROI工具保留历史只读复核/物化能力，但训练入口遵守D-010拒绝该路线。

## 2. 新流程与兼容边界

README已同步到新入口，旧数据目录/权重/标签均未补写。下一次拟开训时，可先按冻结清单创建新派生版本，再检查计划。例如：

~~~text
runtime/python.exe scripts/materialize_dataset.py --manifest reports/splits/v5_split_manifest.csv --root datasets_v5 --name CT_DATASET_VERIFIED_NEW --verify-only
runtime/python.exe scripts/materialize_dataset.py --manifest reports/splits/v5_split_manifest.csv --root datasets_v5 --name CT_DATASET_VERIFIED_NEW
runtime/python.exe scripts/train_run.py --data datasets_v5/CT_DATASET_VERIFIED_NEW/data.yaml --model models/pretrained/yolov8n.pt --name EXP_NEW --baseline-args <基线args.yaml> --dry-run
~~~

dry-run输出精确计划指纹；若基线与默认参数不同，应显式传入保持项，不能把示例当作单变量检查已通过。计划指纹绑定新数据/YAML、基础权重、基线文件、训练参数、入口/契约源码和Ultralytics版本。用kb add-finding把可证伪预测、验收门槛、关联run及plan_sha256登记为待验，再以同参数加--finding F-XXX才可开训。工程代码校验登记和绑定，不能判断所写业务判据是否合理。

run目录由入口独占创建，传exist_ok=True只是让Ultralytics使用该已预留目录；并不允许复用已有run。实际save_dir必须与独占目录一致，否则拒绝继续，避免隐式增后缀后错记产物。--fine-tune-from（兼容--resume-from）明确不恢复优化器状态。

Web只在新进程加载时应用严格规格。未知模型缺meta必须补充经过核对且绑定权重的元信息，不能靠文件夹名猜测。已登记v2mild/V3/V4的三个精确SHA256与现场文件只读核验一致；这仅保留既有兼容行为，不把历史泄漏指标升级为有效发布依据。旧全图meta缺k时因不裁剪取0；ROI缺k仍拒绝。正式新训练写出的completed元信息已在假训练器回归中直接通过共享规格读取。

## 3. 验证结果

| 验证 | 结果 |
|---|---|
| scripts/test_training_contract.py | 24项通过；真实CLI物化/preflight、junction逃逸、force哨兵保留、硬链接无复制回退、坏输入/哈希漂移、分组泄漏、单变量/预注册、异常收尾、实际目录漂移、归档失败区分 |
| scripts/test_model_spec.py | 5项通过；未知权重拒绝、精确旧哈希兼容、SHA256/MD5与父目录meta、坏规格/状态/类别、损坏meta不回退 |
| 既有Web模型状态/精度/台账 | 18项+4项及台账/导出/失败态套件通过；独立随机端口HTTP自测关闭，无5000重启 |
| 既有评测缓存/匹配 | 7项+5项通过；没有重跑V5/V53的676图计分 |
| AST/CLI/KB | 17个本阶段改动Python文件AST通过；两个训练与两个物化CLI --help通过；KB 26/32/12条（4条已证伪）格式通过 |

汇总证据：reports/system_audit/stage4_validation.json，包括测试名、源码指纹、旧权重身份及现场状态。预期的故障注入会打印拒绝信息，不是测试失败。测试没有训练真实YOLO，不能证明GPU优化器收敛、显存容量或模型性能已改善。

## 4. 关闭与后续

| 任务 | 关闭依赖 | 状态 |
|---|---|---|
| 新物化/训练入口的路径、坏数据、单变量与预注册约束 | 算力 | 沙盘关闭；没有批量重建生产数据 |
| 新训练元信息、异常日志收尾、实际best/目录与归档状态 | 算力 | 沙盘关闭；未启动长训练 |
| Web未知规格/旧兼容身份边界 | 算力 | 源码与沙盘关闭；生产进程尚未应用 |
| 完整FP32/FP16质量、真实GPU端到端时延、有界扰动 | 算力 | 下一阶段必要实测，不以假模型耗时代替 |
| 工艺真值、电芯溯源、双人重标、正式模型/精度发布 | 人力/判断 | 保留 |

边界：is_neg=1只是继承清单声明，不证明已独立复核为正常；group不等于完整真实cell_uid。新指纹从本次物化时冻结标签，不能追溯恢复历史run未保存的原始标注版本。单变量比较覆盖请求参数与新输入指纹；旧基线若缺数据内容快照，不能由args相等证明历史数据完全一致。文件哈希检查不提供训练期间对外部改写的文件系统隔离；硬链接JPG不可由其它工具回写。AutoBatch实际批量和累积在回调记录，但不声称它与每一步动态warmup都完全一致。进程互斥针对本工程入口与可识别训练命令，不是整机GPU调度器；崩溃遗留锁不会自动删除，先核对进程再处理。

只读现场复核仍为Web PID26316、current=models/启用/20261019_V4/best，没有训练进程。用户未跟踪V53目录保留。D-010继续生效，不新增架构训练；架构探针仍是后续建议，不构成本轮必须执行的工作。
