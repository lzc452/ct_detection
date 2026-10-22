# V5.1 预处理修复与训练调优记录

- 工作副本：`D:\CT Detection-yolo`（git worktree）
- 时间：2026-10-21
- 目标：把 `docs/preprocess_analysis_20261021.md` 里量化出的预处理缺陷落到一次可验证的训练里，产出与 V5 同口径（同一冻结 test、同一清单 md5）的对照数字
- 约束：**不动母本数据**（`问题电芯图片汇总` / `datasets/` / `积攒/` 全程只读）

---

## 1. 先纠正三个错误认知（源码 + 实测确认）

做完预处理分析后，我原计划的 V5.1 方案是"`imgsz=1184x2112` 矩形画布 + `copy_paste=0.3` 给损伤类增广"。**这两条都是错的**，实测后推翻：

| 原计划 | 实测结论 | 依据 |
|---|---|---|
| 训练用 `imgsz=(1184,2112)` 矩形画布 | **训练/验证的 imgsz 只能是方形 int**，传 tuple 会被静默降级成 `max()` | `trainer.py:340` `check_imgsz(..., max_dim=1)`；`validator.py:141` 同。实测 `check_imgsz((1184,2112), max_dim=1)` → `2112` |
| `copy_paste=0.3` 给 damage 增广 | **检测任务上是空操作**，设了无效 | `CopyPaste.__call__` 首行 `if len(labels["instances"].segments) == 0 or self.p == 0: return labels`；检测标注没有 segments |
| （隐含假设）矩形画布省 43% 算力可以落到训练 | 训练侧做不到；**但推理侧可以**（`predictor.py:274` `min_dim=2`） | 矩形画布的价值转移到线上推理，见 §5 |

另外确认了两个可用工具：

- `Mosaic.__init__` 里是 `self.border = (-imgsz // 2, -imgsz // 2)`，对 tuple 会直接 `TypeError` —— 这是 tuple 不能用于训练的第二个独立原因。
- **`cls_pw` 在检测任务上真实生效**：`DetectionTrainer.set_class_weights()` 用逆频率算权重、归一化到 mean=1，挂到 `model.class_weights` 上喂给分类损失。`cls_pw` 取值 0~1，1=完全逆频率。

---

## 2. V5 表现差的根因：训练与推理的缺陷尺度差 2.2 倍

这是本次调优的核心发现，比"参数没调好"更根本。

`imgsz=1920` + `mosaic=1.0` 时，mosaic 把每张图缩进 **960×960** 的格子（`imgsz/2`）。而评估时用的是 `LetterBox(new_shape=(1920,1920), scaleup=False)` —— **不放大、只按长边缩到 1920**。

| 阶段 | 高分辨率组 2129×1198 的处理 | 缩放率 | 亮斑（原生 16.8px）实际尺寸 |
|---|---|---|---|
| **V5 训练**（mosaic 格） | 缩进 960×960 格子 | 0.451 | **7.6 px** |
| **V5 评估/推理** | 长边缩到 1920 | 0.902 | **15.2 px** |
| **V5.1 训练**（mosaic=0，画布 2112） | 居中不缩放（±3%） | 1.000 | **16.8 px** |
| **V5.1 评估** | 长边缩到 2112（scaleup=False） | 0.992 | **16.7 px** |

**V5 在训练时看到 7.6px 的亮斑，推理时却是 15.2px —— 差 2.0 倍。** 对只有 3.3 个灰度级对比度、单像素 SNR<1 的亮斑来说，这等于换了目标；模型学到的"尺度先验"在推理时完全错位。

这也解释了为什么 V5 曲线在 `close_mosaic` 关掉之后（epoch 70 后）还略有上升（0.411→0.414）：那几轮才第一次见到接近推理尺度的样本，但只训了 4 轮就不够了。

低分辨率组受影响较小（mosaic 缩 0.913 vs 评估 1.0，错配 1.1 倍），所以问题集中在占 68% 的高分辨率组。

---

## 3. 画布扫描：先用 V5 权重做不改训练的预验证

在花 4 小时重训之前，先回答"更大画布是否真的更好"。做法：V5 权重（1920 训的）分别在 1920 与 2112 上评估同一冻结 test。

| 指标 | 1920 | 2112 | Δ |
|---|---|---|---|
| mAP50 | 0.8028 | 0.7826 | −0.0202 |
| **mAP50-95** | 0.4186 | **0.4238** | **+0.0052** |
| 亮斑 AP | 0.4869 | **0.5175** | **+0.0306（+6.3% 相对）** |
| 亮斑 AP50 | 0.9436 | **0.9556** | +0.0120 |
| 亮斑 R | 0.8963 | **0.9259** | **+0.0296** |
| 亮斑 P | 0.9237 | 0.8993 | −0.0244 |
| 暗斑 AP | 0.5326 | 0.5187 | −0.0139 |
| 损伤 AP | 0.2363 | 0.2350 | −0.0012 |
| FP/千张 | 88.8 | 122.8 | +34.0 |
| **组级漏检率** | 0.2647 | **0.2471** | **−0.0176** |

**怎么读这张表**：改善的恰好是"小目标类（亮斑）"和"漏检"，变差的恰好是"精确率/FP"。这是分辨率提高的典型指纹——看到更多细节 → 小目标检出率上去、TP 和 FP 同时增加。

注意这次是**跨分辨率推理**（1920 训、2112 评），存在错配，所以 mAP50 和暗斑反而掉了。真正的收益要在"同分辨率训练"下才干净，这正是 V5.1 要做的实验。

结论：画布方向正确，值得用 2112 重训。

---

## 4. V5.1 的完整配置与逐项依据

| 参数 | V5 | V5.1 | 依据 |
|---|---|---|---|
| `imgsz` | 1920 | **2112** | 原生宽 2129，2112 时缩放率 0.992≈1:1；亮斑 16.8px 保持（1920 时 15.2px）。「§3 实测亮斑 AP +0.031」 |
| `mosaic` | 1.0 | **0.0** | 消除训练/推理 2.0 倍尺度错配（§2）；且亮斑在 mosaic 后 52.6% 的框 <8px（P3 单格） |
| `close_mosaic` | 10 | **0** | mosaic 已全程关闭，此参数无意义 |
| `degrees` | 5.0 | **0.0** | 3 个灰度级的缺陷经任意角旋转会重采样，SNR 损失无补偿 |
| `translate` | 0.05 | **0.0** | 2129 宽的图在 2112 画布上已超 17px，再平移 ±5%(105px) 会把边缘缺陷移出画布 → 等价制造漏标 |
| `scale` | 0.05 | **0.03** | 保留轻微尺度多样性（±3%），最坏裁边 ≤1.9%，可接受 |
| `erasing` | **0.4（默认，从未显式设置）** | **0.0** | ultralytics 8.4 默认随机擦除 40%，可能正好盖住 15px 亮斑 → **等于给模型喂错标** |
| `fliplr` | 0.5 | 0.5 | 保持。CT 缺陷无固定左右语义，水平翻转安全 |
| `hsv_v` | 0.1 | 0.1 | 保持。亮度只扰动 ±10%，保护微弱亮斑 |
| `cls_pw` | 0.0 | **0.5** | train 中 damage 338 框 / dark 1092 框（3.2:1）。`cls_pw=0.5` 约给 damage 1.28× 分类权重，温和纠偏 |
| `copy_paste` | 0.0 | 0.0 | 检测任务空操作（§1），不加 |
| `lr0` / `optimizer` | 1e-3 / AdamW+cos | 同 | 新训而非续训，与 V5 一致才可比 |
| `epochs` / `patience` | 80 / 20 | 同 | V5 已证明 80 ep 收敛（续训 120 ep 仅 +0.0015），无需加预算 |

**为什么敢把增强削弱到这么弱**：因为 §2 证明了"增强（mosaic）本身"就是尺度错配的来源。削弱几何增强不是偷懒，是把训练分布对齐到推理分布。代价是过拟合风险上升，用 `weight_decay=5e-4` + `fliplr=0.5`（等效 2 倍数据）+ patience 早停兜底。

**训练数据分布（`reports/splits/v5_split_manifest.csv`）**：

| split | 图 | damage 框/图 | dark 框/图 | light 框/图 | 负样本 |
|---|---|---|---|---|---|
| train | 3195 | 338 / 307 | 1092 / 1021 | 630 / 604 | **1337（41.8%）** |
| val | 721 | 73 / 69 | 234 / 225 | 136 / 133 | 312 |
| test | 676 | 73 / 67 | 234 / 213 | 135 / 128 | 287 |

平均每图 0.64 个框、41.8% 是纯背景图——这是 FP/千张 偏高的结构性原因，不是调参能完全解决的。

---

## 5. 推理侧的额外收益（不改训练就能拿）

虽然训练不能用矩形画布，**推理可以**（`predictor.py` 用 `check_imgsz(min_dim=2)`）。同一张图在 `imgsz=(1184,2112)` 与 `2112` 方形下的检测框坐标一致（≤2px，2026-10-21 实测），但画布算力从 4.46 MP 降到 2.50 MP（**−44%**）。

> **2026-10-22 更新**：ROI 方案确定后，这一条已不再作为主推方向。ROI 裁剪本身把画布降到 647×991，推理只需要 1536 方形（2.36 MP），比矩形全图方案更低，且精度更高（见评测报告）。`web/app.py` 已改为「裁剪 ROI → 检测 → 坐标映射回原图」，由模型旁的 `_train_meta.json` 决定是否走 ROI 分支；无 meta 的旧模型仍按全图 `INFER_SIZE` 推理，行为不变。

`scripts/eval_report.py` 现在会把 `imgsz_requested` / `imgsz_layer1` / `imgsz_predict` 分别记进 JSON，避免"以为用了矩形其实被降级"这类记录失真。

---

## 6. 测量口径必须说清的两个坑（源码级确认）

这两条都会直接影响数字怎么读，必须写在报告前面。

### 6.1 `train_batch0.jpg` 看起来像 mosaic，其实不是

`mosaic=0` 的 run，其 `train_batch0.jpg` 仍是 2×2 拼接版式，容易被误读成 mosaic 生效。判据：图里**保留着每张图的原始文件名**（`9_0724.jpg`、`1_1287.jpg`…），且格子数 = batch size（1536 画布下 batch=3）。真正的 mosaic 会重采样子图并清掉文件名。以 `args.yaml` 的 `mosaic:` 值为准，不看预览图。

### 6.2 训练会放大低分辨率图，验证不会——所以 val mAP 系统性偏低

- 训练：`ultralytics/data/dataset.py:317` → `v8_transforms()` → `LetterBox(..., scaleup=True)`（默认），**小图被放大**到画布尺寸。
- 验证：`ultralytics/data/dataset.py:319` → `Compose([LetterBox(new_shape=(imgsz, imgsz), scaleup=False)])`，**小图不放大**，只补边。
- 推理：`predictor.py` 走 `LetterBox(..., scaleup=True)`，与训练一致。

对 ROI 数据集影响很具体：低分辨率组 ROI 只有 **319×484**（占 32%）。训练把它放大到约 1010×1530（3.1×），验证却只放进 1536 画布的一角（有效像素仍是 319×484），推理又按 3.1× 放大。

后果与处理：

1. `layer1_coco`（走 `model.val`，`scaleup=False`）**低估**低分辨率组的表现，整体 val mAP 也因此偏低；
2. 同一份报告里 `layer1` 与 `layer2/3`（走 `predict`，`scaleup=True`）在小图上口径不同；
3. 但这个偏差对 V3/V4/V5/V5.1 **完全一样**，所以**跨模型对比仍然公平**；而**产线决策应看 `layer2_3` 工作点数字**，因为它与线上推理路径一致。
4. 未在本次修改中绕过该行为（改 `site-packages` 不可接受，且会破坏与其他模型的同基准可比性），仅登记为测量口径缺陷。

---

## 7. 训练执行

### 7.1 实验一：全图 @2112（已证伪，保留作反例）

```powershell
runtime\python.exe scripts\train_run.py --data "datasets_v5\CT_DATASET_V5_GROUPED\data.yaml" `
    --model "models\pretrained\yolov8n.pt" --imgsz 2112 --epochs 80 --patience 20 `
    --mosaic 0 --close-mosaic 0 --degrees 0 --translate 0 --scale 0.03 `
    --erasing 0 --cls-pw 0.5 --lr0 0.001 --name V51_PREPROC_1021
```

结果：77 轮 / 5.36 h / 251 s 每轮，test mAP50-95 **0.346**（V5 0.419）。**失败**，根因是关掉 mosaic 后失去主要正则项 → 过拟合（train box_loss 0.543 vs val 1.101，2.0×），叠加 32% 低分辨率组在 2112 方形画布上只占 13.7% 面积。

### 7.2 实验二：ROI 裁剪 @1536（当前方案）

```powershell
# 1) 物化 ROI 数据集（真实裁剪+重编码，不是硬链接）
runtime\python.exe scripts\materialize_roi_dataset.py

# 2) 训练
runtime\python.exe scripts\train_run.py `
    --data "datasets_v51\CT_DATASET_V51_ROI\data.yaml" `
    --model "models\pretrained\yolov8n.pt" --imgsz 1536 --epochs 80 --patience 20 `
    --mosaic 0 --close-mosaic 0 --degrees 0 --translate 0 --scale 0.05 `
    --erasing 0 --cls-pw 0.5 --lr0 0.001 --name V51_ROI_1536
```

关于 `--scale 0.05` 而不是实验一的 `0.03`：ROI 之后画布利用率从 13.7%~56% 提升到约 99%，缩放增强不会再把目标推出画面，可以给一点尺度抖动来补回关掉 mosaic 后损失的正则性；但仍保持很小，避免破坏"缺陷绝对像素尺度"这一信号。

产物（`runs/detect/CT_pro/V51_ROI_1536/`）：

| 文件 | 内容 |
|---|---|
| `train.log` | 全量 stdout+stderr（含每轮指标行、Class weights 打印） |
| `results.csv` | 每轮 19 项指标 |
| `_train_meta.json` | 数据集/权重 md5、全部超参、`hyperparams_effective`（ultralytics 实际生效值）、`roi`/`roi_box` |
| `results.png`、`Box*_curve.png`、`confusion_matrix*.png`、`val_batch*_{labels,pred}.jpg` | 训练曲线与验证可视化 |

### 7.3 训练过程图表（自建对比图）

```powershell
runtime\python.exe scripts\plot_training.py `
    --run V51_ROI_1536=runs/detect/CT_pro/V51_ROI_1536 `
    --run V51_PREPROC_1021=runs/detect/CT_pro/V51_PREPROC_1021 `
    --run V5_GROUPED_1021_80ep=runs/detect/CT_pro/V5_GROUPED_1021_80ep `
    --outdir reports/train
```

产出 `reports/train/curves_{metrics,loss,lr,overfit}.png` 与 `summary.json`（best epoch、最优指标、泛化间隙）。

---

## 8. 结果与验收（2026-10-22）

完整评测见 **`docs/v51_eval_report_20261022.md`**。这里只放结论：

| 指标（冻结 test 676 张，conf 0.25 / IoU 0.5） | V5 基线 | V5.1-A 全图@2112 | V5.1-B ROI@1536 |
|---|---|---|---|
| mAP50 | **0.803** | 0.713 | 0.752 |
| mAP50-95 | **0.4186** | 0.346 | 0.4015 |
| FP/千张 | **88.8** | 182.0 | 127.2 |
| 组级漏检 | **26.5%** | 30.0% | 28.2% |
| 损伤召回 | **0.562** | 0.562 | 0.397 |
| 单张推理 | 13.4 ms | 16.1 ms | **11.1 ms** |

**验收判定：不通过，V5.1-B 不替换生产模型。**

但有两条要保留下来：

1. **ROI 裁剪本身有效**：在完全相同的超参下（V5.1-A → V5.1-B），只换裁剪与画布，mAP50-95 `0.346 → 0.402`（+0.056），FP/千张 `182.0 → 127.2`（−30%），推理 `16.1 → 11.1 ms`（−31%），训练 `5.36 → 1.89 h`（−65%）。收益集中在低分辨率组（FP/千张 `98.6 → 14.1`，−86%）。
2. **去掉 mosaic/旋转/erasing 是净损失**，且比 ROI 的增益更大。本次一次改了 5 个变量，无法做单变量归因——这是本实验最大的方法缺陷，必须在下一轮修正。

**下一轮（V5.2）**：`ROI@1536 + mosaic 1.0 + close_mosaic 15 + degrees 5 + translate 0.05 + erasing 0.4 + cls_pw 0.7`。
关键洞察：ROI 把亮斑从 7.6px 抬到 26.0px，因此又有余量去支付 mosaic 的代价——mosaic 子图下亮斑仍有 13.0px，不低于 V5 推理时的 15.2px；再用 `close_mosaic 15` 在最后 15 轮回到真实尺度校准。这是「用几何收益换正则化」的正确用法，等确认后执行。


