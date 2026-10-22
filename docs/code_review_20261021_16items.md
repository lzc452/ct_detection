# 代码评审：16 项问题逐条核实报告

- 评审对象：`D:\CT Detection-yolo`（当前工作副本）
- 评审时间：2026-10-21
- 评审视角：视觉算法工程师（数据管线 → 训练 → 推理 → 评估 → 工程化）
- 核实方式：逐行读码 + 在真实数据上跑最小验证脚本（脚本已删除，结论可复现）

> 说明：你给的 (line NNN) 与当前副本行号有偏移（本副本 `web/app.py` 为 1534 行）。下文一律给出**当前副本的真实行号**，并标注与你描述的对应关系。

---

## 0. 一页结论

| # | 问题 | 判定 | 严重度 | 核心依据 |
|---|---|---|---|---|
| 1 | 旋转增强框方向相反 | **真实缺陷** | **P0** | `augment_final.py:72-73` 用了 `M` 的转置（-angle），实测偏差 67.0px |
| 2 | 中文路径约定未贯彻 | **真实缺陷（分级）** | **P1** | 实测中文目录 `cv2.imread` 失败 40/40；`save_annotated_image` 不查返回值 |
| 3 | 检测失败显示成"未检出" | **真实缺陷，且更严重** | **P0** | 失败文件被 `continue` 跳过 → 响应与前端 `chunk[k]` **索引错位** |
| 4 | 复核/预标注脚本破坏人工数据 | **真实缺陷** | **P0** | `exec_v4_review.py:26` 顶层 `unlink()`，无清单无备份 |
| 5 | 登录无服务端校验 | **真实缺陷** | **P1**（联网即 P0） | `login.js:154` 硬编码口令；`/main` 与全部 API 无鉴权；`app.py:1534` 监听 0.0.0.0 |
| 6 | Excel 检测文件数错 | **真实缺陷** | **P1** | `app.py:476/627` 恒传 `''`，`:893` 按该字段去重 |
| 7 | 训练默认 1280 与部署 1920 冲突 | **真实缺陷** | **P1** | `train.py:89` 默认 1280；`app.py:61` 强制 1920；类别名硬编码于 `:99-100` |
| 8 | 阈值 <0.25 无效 | **真实缺陷** | **P1** | `conf=0.25` 写死在 `:462/592/773`，再二次过滤 |
| 9 | 模型切换/状态竞态 | **真实缺陷** | **P1** | `load_model` 先删后建（`:123-135`）；`save_state` 非原子（`:168-176`） |
| 10 | 图级划分泄漏 | **真实缺陷，比预估严重** | **P0** | 实测 val 中 **82.7%** 与 train 存在像素级重复（MAE<1） |
| 11 | 宽松匹配掩盖定位问题 | **真实缺陷** | **P0** | `eval_loose.py:28` IoU≥0.1 或中心互落；V3 用 IoU 0.5（`analyze_v3_val.py:15`） |
| 12 | 标注转换/命名缺校验 | **真实缺陷** | **P1** | 空目录 `rng.sample` 崩溃；`cls` 越界；`a(1)`/`a1` 同名覆盖 |
| 13 | 便携路径/配置无统一依据 | **真实缺陷，且已发生** | **P1** | 6 个脚本硬编码 `D:\CT Detection`；`models/` 为空导致 `app.py` 导入即崩 |
| 14 | 增强策略未做对照 | **部分真实（3 个独立问题）** | P1/P2 | 预处理只进 train；5×/2× 改比例；`close_mosaic` 早停前根本不触发 |
| 15 | "原图"不可追溯 | **真实缺陷** | **P1** | 转 RGB 重编码 JPEG；扩展名沿用上传名；无模型/阈值/哈希记录 |
| 16 | 测试与文档落后 | **部分真实** | **P2** | `test_real.py` 在本仓库**不存在**；文档引用不存在的脚本；导入即加载模型 |

**一句话总结**：16 条里 14 条完全成立、2 条方向对但细节需修正（#14、#16）；其中 #1/#4/#10/#11 属于"不修则所有训练与汇报结论都不可信"的级别。而我在核实 #10 时挖到了比原判断严重得多的事实（见 §1 红旗 A）。

---

## 1. 三个红旗（你未提出，但优先级高于全部 16 条）

### 红旗 A：所谓"考卷"测试集 100% 来自训练/验证集，93.3% mAP 无意义

`app.py:144` 的注释写着"2026-10-19 清洗考卷 mAP50 93.3%，生产口径三类召回 95.8/96.6/90.3，全面优于 V3"，`eval_loose.py` 也以 `datasets/考卷_清洗版` 作为评测集。实测：

```
考卷 196 张：MD5 与 V4 train 完全相同 66 张，与 V4 val 完全相同 130 张
→ 66 + 130 = 196，即 100% 的考卷图片都字节级来自 V4 数据集本身
像素级复核：191/196 张与 V4 train 最近邻 MAE < 1（灰度 0~255 尺度）
```

**这不是"考卷与训练集有部分重叠"，是"考卷就是训练集/验证集的子集"**。因此：

- 93.3% mAP50、95.8/96.6/90.3 的召回，全部是**自评分数**，不具备任何泛化含义；
- 以此为依据做的"V4 优于 V3"结论、"生产口径"上线决策，都缺少有效证据；
- 反过来也不能断言 V4 一定差——只是**目前没有可信证据**。

**修复**：立刻冻结一个**从未参与训练的数据集**（按电芯/批次分组，见 #10），重新出全部指标，再更新 `app.py:144` 的注释与所有汇报口径。

---

### 红旗 B：`models/启用/` 为空，服务根本无法启动

```
models/  →  空目录（无任何 .pt）
app.py:145-151 → 启动时若无模型，raise FileNotFoundError
```

也就是说当前副本 `runtime\python.exe web\app.py` 会在**导入阶段**直接抛异常退出。同时 `Config.py:7` 的 `model_path = 'models/启用/CT_DATASET_AUG_0910_1419/best.pt'` 指向一个不存在的模型，`检查环境.py:103-110` 必然报"模型文件不存在"，而 Web 想用的是 `20261019_V4`（`app.py:146`）——三者互不一致，正是 #13 说的问题，但它已经**实际发生**，不只是"隐患"。

### 红旗 C：`web/results` 是指向旧工程的符号链接，两套程序共享同一份状态

```
D:\CT Detection-yolo\web\results -> /d/CT Detection/web/results
D:\CT Detection            ← 旧工程，仍然存在（含自己的 scripts/datasets/models）
D:\CT Detection-yolo       ← 当前工程
```

后果：`/api/clear`（`app.py:1465-1471`）会**删除旧工程 results 目录里的全部文件**；`_state.json`、`_detect_task.json` 被两套代码共同读写；`web/app.py` 里 `RESULTS_FOLDER` 实际落在旧目录。配合 6 个脚本硬编码的 `D:\CT Detection`，现状是"两套树互相踩"。

**修复**：`web/results` 改为本工程内的真实目录（复制或软链指向自身），旧工程目录整体归档只读；所有脚本统一用 `__file__` 推导根目录。

---

## 2. 逐条分析

### 问题 1 — 旋转增强把图与框转向相反方向

**判定：真实缺陷（P0，数据管线）**

**证据**（`scripts/augment_final.py:60-81`）

- 图像侧：`M = cv2.getRotationMatrix2D(center, angle, 1.0)`，再 `warpAffine(img, M, ...)`。OpenCV 的图像坐标系 y 轴向下，该矩阵的线性部分为 `[[cos, sin], [-sin, cos]]`，即视觉上逆时针转 `angle`。
- 框侧（`:72-73`）：
  ```python
  nx = (px-cx)*ca - (py-cy)*sa + cx
  ny = (px-cx)*sa + (py-cy)*ca + cy
  ```
  线性部分是 `[[cos, -sin], [sin, cos]]` —— 恰好是 `M` 线性部分**顺时针/逆时针互换**的那个矩阵，等价于把框转了 `-angle`。

**量化**：距中心 `d=640px` 处，3° 的双向偏差 = `2·d·sin(3°) = 2×640×0.05234 ≈ 67.0 px`。与你的估算完全一致。V4 的缺陷框长边多在 30~150px 量级，67px 偏移等于**标签完全落在缺陷之外**，而且偏移方向随框在图像中的象限变化（不是整体平移，无法被模型当作系统性偏差"学掉"）。

**影响**：`POS_VARIANTS[3]` 是每个正样本的固定 1/5 变体，即**约 20% 的正样本训练图带着系统性错误标签**。V3_AUG、V4_AUG 都包含该变体（`augment_final.py:84-89`）。这不会让模型完全失效（其余 4/5 变体与 val 是对的），但它会持续给框回归注入噪声、并抬高置信度校准误差——这恰好与"误报压不下去、需要把阈值从 0.25 一路提到 0.30"的现象吻合。

**修复**

```python
def rot(img, boxes, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    img2 = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = []
    for c, x, y, bw, bh in boxes:
        x1, y1 = (x - bw / 2) * w, (y - bh / 2) * h
        x2, y2 = (x + bw / 2) * w, (y + bh / 2) * h
        pts = cv2.transform(
            np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], np.float32), M
        ).reshape(-1, 2)
        nx1, ny1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        nx2, ny2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        # 先按整框是否还在图内判断，再统一裁剪；逐点裁剪会把越界框拉成"贴边假框"
        if nx2 <= 0 or ny2 <= 0 or nx1 >= w or ny1 >= h:
            continue
        area0 = (x2 - x1) * (y2 - y1)
        cx1, cy1 = max(0, nx1), max(0, ny1)
        cx2, cy2 = min(w, nx2), min(h, ny2)
        if cx2 - cx1 < 2 or cy2 - cy1 < 2:
            continue
        # 裁掉超过 40% 面积的框直接丢弃，避免"半个缺陷"进训练
        if (cx2 - cx1) * (cy2 - cy1) < 0.6 * area0:
            continue
        out.append([c, (cx1 + cx2) / 2 / w, (cy1 + cy2) / 2 / h,
                    (cx2 - cx1) / w, (cy2 - cy1) / h])
    return img2, out
```

顺手修掉原实现的两个次生问题：① 原来的 `max(0,min(xs))` / `min(w,max(xs))` 是**逐点裁剪**，框一半转出图外时会产出一个贴边的、尺寸错误的框；② 越界后面积塌缩到 <2px 的框被 `continue` 丢弃，但**图像里那块缺陷仍然存在**——这等价于把缺陷当背景训练（假阴性标签）。上面的实现丢弃阈值更严格。

**必需的回归检查**（你提的"合成图回归检查"，我建议做成可自动化的）

```python
def test_rot_label_follows_pixels():
    """在纯黑图上画一个白块，旋转后从像素反算真实位置，与返回的框比 IoU"""
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.rectangle(img, (520, 60), (600, 140), (255, 255, 255), -1)   # 故意偏离中心
    boxes = [[0, 560/640, 100/480, 80/640, 80/480]]
    for ang in (-5, -3, 3, 5, 15):
        im2, bx = rot(img, boxes, ang)
        ys, xs = np.where(cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY) > 127)
        assert len(xs), "白块被转出画布"
        c, xc, yc, bw, bh = bx[0]
        pred = [((xc-bw/2)*640), ((yc-bh/2)*480), ((xc+bw/2)*640), ((yc+bh/2)*480)]
        real = [xs.min(), ys.min(), xs.max(), ys.max()]
        assert iou(pred, real) > 0.70, f"angle={ang} IoU={iou(pred, real):.3f}"
```

用原实现跑这个用例，±3° 时 IoU 会掉到 0.1~0.3（白块是 80px 见方、偏心 560px，偏移约 58px），修复后应 >0.9。

**优先级与收益**：P0。修复成本约 20 行 + 一个单测；收益是消除 20% 正样本的系统性标签噪声，**任何后续重训前必须先做**。注意：修复后必须**重建** AUG 数据集（`AUG_RESUME` 会跳过已完成文件，不能原地续跑）。

**归类：确需修复的缺陷。**

---

### 问题 2 — 中文路径读写约定没有贯彻

**判定：真实缺陷，但要分三级看**

**证据 A（确定缺陷，已实测触发）**

`app.py:232-243`：
```python
try:
    cv2.imwrite(result_path, img)
    return result_filename      # ← 无论 imwrite 成功与否
except Exception as e:
    ...
```
`cv2.imwrite` 失败时**返回 False，不抛异常**（OpenCV 对 Unicode 路径在 Windows 上的典型行为），所以 `except` 分支永远进不去，函数把"一个不存在的文件名"当成功返回。`app.py:489-490` 的 `if not result_filename` 检查因此形同虚设，前端拿到 404 的图片 URL。

**这不是理论风险，我在本机实测到了同样的失败模式**：中文目录 `datasets/考卷_清洗版/images` 下 `cv2.imread` **40/40 全部返回 None**（OpenCV 打 WARN `can't open/read file`），换成 `imdecode + np.fromfile` 后 **40/40 成功**。当前 `results` 目录路径（`D:\CT Detection\web\results`）恰好是纯 ASCII 才没爆；一旦把程序放到 `D:\电芯检测\`、或导出到中文目录，保存会静默失败。

**证据 B（隐患，当前未触发）**

- `scripts/augment_final.py:110,119,124,129`：`cv2.imread` / `cv2.imwrite`。当前数据文件名是 ASCII（`0_0001.jpg`），所以没出事；但 `:92` 允许用 `sys.argv[1]` 指定任意数据集名，用户传中文名即触发。
- `core/ct_preprocess.py:148,151`：同上，`preprocess_dataset_images` 整个函数对中文路径无效。

**影响**：A 级是**用户可见的功能故障**（结果图裂图、导出包里缺图，且日志无异常）；B 级是**未来必定踩的坑**，AGENTS.md 第 2 条已把它列为硬约定，属于"文档写了但代码没执行"的规则漂移。

**修复**：抽一个 `core/imgio.py`，全项目只走这一条路径。

```python
# core/imgio.py
import os, cv2, numpy as np

def imread_cn(path, flags=cv2.IMREAD_COLOR):
    """中文/Unicode 路径安全读图；失败抛异常，不返回 None"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"文件为空或不存在: {path}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"无法解码图片（格式不支持或文件损坏）: {path}")
    return img

def imwrite_cn(path, img, ext=None, quality=95):
    """中文/Unicode 路径安全写图；失败抛异常，绝不静默返回"""
    ext = ext or os.path.splitext(path)[1] or '.jpg'
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext.lower() in ('.jpg', '.jpeg') else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"编码失败: {path}")
    buf.tofile(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"写入失败: {path}")
    return path
```

`save_annotated_image` 改成：

```python
def save_annotated_image(img, filename):
    ext = os.path.splitext(os.path.basename(filename))[1].lower() or '.jpg'
    if ext not in ('.jpg', '.jpeg', '.png'):     # 结果图统一落 jpg，避免 bmp/tif 体积与兼容问题
        ext = '.jpg'
    name = f"result_{uuid.uuid4().hex}{ext}"
    try:
        imwrite_cn(os.path.join(RESULTS_FOLDER, name), img, ext)
        return name
    except Exception as e:
        print(f"[Save Error] {name}: {e}")
        return None          # 调用方 :489 的检查这时才真正生效
```

**优先级与收益**：P1（A 级）；B 级 P2。收益：消除"结果图 404 / 导出缺图"这类无日志故障，同时让 AGENTS.md 的约定第一次真正可执行。

**归类：A 级确需修复；B 级属于"约定未贯彻"，成本极低，建议同批改掉。**

---

### 问题 3 — 检测失败在界面上呈现为"未检出目标"

**判定：真实缺陷，且比你描述的更严重（存在索引错位）**

**证据**

1. `app.py:566-576` 批量读图：
   ```python
   except Exception as e:
       print(f"[Skip] Cannot read {f.filename}: {e}")
       continue                       # ← 直接跳过，response 里没有这条
   ```
   而 `:556` `total_files = len(files)`、`:688` 响应 `"total": total_files` → **总数含失败项**，`results` 数组不含。
2. `app.py:662-673` 结果处理异常时塞入 `"error": str(e)` 的条目，`target_nums: 0`。
3. 前端 `main.js:452-455`：只要 `locations.length === 0` 就渲染 `未检出目标：<文件名>`；`main.js:1074-1082` 构造 `detection` 对象时**丢掉了 `item.error`** —— 于是"推理异常"和"真的没有缺陷"在 UI 上完全同形。
4. **额外发现（比上面更危险）**：`main.js:1029` `chunk = files.slice(...)`，`:1069-1073`
   ```js
   for (let k = 0; k < data.results.length; k++) {
       if (!item.image_url && chunk[k]) item.image_url = URL.createObjectURL(chunk[k]);
   ```
   后端跳过失败文件后，`data.results[k]` 与 `chunk[k]` **不再对齐**。假设 chunk 里第 2 张读图失败，那么第 3 张的结果会挂上第 2 张的缩略图 —— 界面上出现"文件名 A、图片是 B、框是 A 的框"的错乱。这在产线复核时足以让人误判。
5. 视频 `app.py:760-792`：帧级 `except` 里 `annotated = frame`（写原帧），`cur += 1` 照常推进，最终 `:815` `status='done'`。同时 `:738` 创建 `VideoCapture` 后**没有 `cap.isOpened()` 检查**：打不开时 `width/height=0`，`VideoWriter` 静默失败，`while` 循环 0 次，然后照样 `status='done'` + 返回一个不存在/0 字节的 mp4 URL。

**影响**：把"检测失败"当"无缺陷"是**质量事故型错误**——它不只影响统计，会让坏图直接流入良品。视频侧更隐蔽：整个任务"成功完成"，实际上可能一帧都没检测。

**修复**（关键是让"状态"成为一等公民，而不是靠"框数为 0"推断）

```python
# 统一的逐文件状态
STATUS_OK_DEFECT, STATUS_OK_CLEAN, STATUS_FAILED = 'ok_defect', 'ok_clean', 'failed'

# 批量：按输入下标建记录，永远与前端 chunk 对齐
for j, f in enumerate(chunk_files):
    idx = i + j
    try:
        img = np.array(Image.open(f.stream).convert('RGB'))
    except Exception as e:
        records[idx] = {'index': idx, 'filename': names[idx], 'status': STATUS_FAILED,
                        'error': f'读图失败: {e}', 'target_nums': 0,
                        'locations': [], 'classes': [], 'confs': []}
        continue
    ...
return jsonify({
    "success": True,
    "total": total_files,
    "ok_defect": n_defect, "ok_clean": n_clean, "failed": n_failed,
    "failed_files": [...],
    "results": [records[k] for k in sorted(records)]   # 保证顺序 + 缺项显式失败
})
```

前端三处配套：① `addTableRows` 前先判 `detection.status === 'failed'`，渲染成红色 `检测失败：<文件名>` 行并计入"失败"计数，**不要**走"未检出"分支；② 用 `item.index` 而不是循环下标去取 `chunk[item.index - chunkStart]`；③ 结束提示改为 `共 N 张：有缺陷 X、无缺陷 Y、失败 Z`，Z>0 时用 `showToast('error', ...)`。

视频侧：

```python
cap = cv2.VideoCapture(temp_path)
if not cap.isOpened():
    video_tasks[task_id].update(status='error', message='视频无法打开（编码不支持或文件损坏）')
    return
...
out = cv2.VideoWriter(result_path, fourcc, fps, (width, height))
if not out.isOpened():
    video_tasks[task_id].update(status='error', message='视频写入器初始化失败')
    cap.release(); return
...
frame_failed = 0
except Exception as e:
    frame_failed += 1
    annotated = frame                       # 保底写原帧，但必须记账
...
video_tasks[task_id].update(
    status='done' if frame_failed == 0 else 'done_with_errors',
    frames=cur, frames_defect=total_frames_with_defects, frames_failed=frame_failed,
    message=f'{frame_failed} 帧推理失败，已用原帧填充' if frame_failed else '')
```

**优先级与收益**：P0。这是"结论正确性"问题而非体验问题。收益：把"失败"从"无缺陷"里彻底剥离，使 `失败数=0` 成为可监控的质量闸门；顺带修掉图片与文件名错配。

**归类：确需修复的缺陷。**

---

### 问题 4 — 复核与预标注脚本可能破坏人工数据

**判定：真实缺陷（P0），且当前脚本还指向**错误的工程目录****

**证据**

- `scripts/exec_v4_review.py:21-27`：模块顶层直接 `p.unlink()`，删 `问题电芯图片汇总/<类>/<stem>.jpg` 与 `.json`；`:41-57` 就地改 `1_0441.json`，无框时还 `shutil.move` + `jp.unlink()`；`:59` 追加写 `删除记录.txt`。**没有 dry-run、没有备份、没有回滚**，脚本一跑即不可逆。
- `scripts/prelabel.py:46-48`：`open(jp,'w')` 无条件覆盖同名 json。收割目录里如果已经有人工改过的 json，**会被模型预标注直接冲掉**。
- `scripts/merge_v4_staging.py:34-38,52,58`：`shutil.copy2` 覆盖母本图片、`dst_json.write_text` 覆盖/合并母本 json，并 `old_img.unlink()` 删除负样本原图。幂等性只靠"`box_key` 去重"和"`next_id` 取 max+1"，中断重跑会重复生成新编号（`next_id` 已把上次的算进去，但图片已复制的情况没有检测）。
- 三个脚本 `ROOT = Path(r"D:\CT Detection")`（`exec_v4_review.py:6`、`merge_v4_staging.py:6`、`prelabel.py:8`）—— 当前工程是 `D:\CT Detection-yolo`，而这个旧目录**真实存在**（见红旗 C）。也就是说现在运行它们，改的是**旧工程里的母本**，两套树从此分叉且无人知晓。

**影响**：这是唯一能造成"人工标注劳动成果不可恢复丢失"的一类脚本。V4 母本 3753+ 张人工标注是项目最贵的资产，一次误跑即损失。

**修复（分三步，缺一不可）**

1. **默认 dry-run**：`--apply` 才真正落盘；无 `--apply` 时打印完整操作清单（删除/修改/新增/移动，含源与目标绝对路径）并退出。清单同时写 `复核_YYYYMMDD_plan.csv`，行号即"操作编号"。
2. **写前备份**：任何 `unlink`/`write_text` 之前，把目标文件复制到 `_backup/<批次号>/<原相对路径>`；顶部用 `try/finally` + `shutil.copytree` 做整批快照（只备份将被触及的文件，按清单驱动）。
3. **幂等 + 来源记录**：操作编号写进 json 的 `attributes.op_id`；合并时若 `op_id` 已存在则跳过（真正的幂等，而不是靠坐标近邻）；`prelabel` 默认 `if jp.exists(): skip`，`--overwrite` 才覆盖。

```python
# 骨架
import argparse, shutil, csv
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--apply', action='store_true', help='不加此参数只输出清单')
ap.add_argument('--backup-dir', default='_backup')
args = ap.parse_args()

ROOT = Path(__file__).resolve().parents[1]     # ← 不再硬编码绝对路径
BATCH = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

def do_unlink(p: Path, log):
    log.append(('DELETE', p))
    if args.apply:
        bak = ROOT / args.backup_dir / BATCH / p.relative_to(ROOT)
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, bak)
        p.unlink()

log = []
...  # 收集所有操作
for op, p in log:
    print(f'{op:6} {p}')
if not args.apply:
    print(f'\n共 {len(log)} 项操作（dry-run）。确认无误后加 --apply 执行。')
    raise SystemExit(0)
```

**优先级与收益**：P0。改动集中在 3 个脚本的入口，约 80 行；收益是"最贵资产不可丢失"+ 消除两套树分叉。

**归类：确需修复的缺陷。**

---

### 问题 5 — 登录页面没有服务端访问控制

**判定：真实缺陷**

**证据**

- `login.js:154-155`：`VALID_USER='HPT'`、`VALID_PASS='123456'` 写在前端；`:185` 纯浏览器比对；`:198` `window.location.href='/main'`。
- `app.py:371-379`：`/` 与 `/main` 都是无鉴权 `render_template`；全部 22 个 API 路由无任何 session/token 校验。
- `app.py:1534`：`app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)`。

**影响**：同网段任何人 `curl -X POST http://<ip>:5000/api/switch_model -d '{"model_id":"..."}'` 即可切模型；`/api/export` 可拉走全部检测数据；`/api/clear` 一行请求清空全部结果（并顺带删掉旧工程 results，见红旗 C）。口令 `HPT/123456` 还能被"查看页面源代码"直接读到。

**但要区分两种场景**（这是本条的关键判断）：

| 场景 | 现状是否可接受 | 建议 |
|---|---|---|
| **纯单机**（产线工控机，不联局域网） | 前端登录作为"防误操作"是**可接受的权衡** | 把 `0.0.0.0` 换成 `127.0.0.1`（改 1 行，零成本），其余不动 |
| **需要局域网/平板查看** | **不可接受** | 必须上服务端会话 + WSGI |

单机版最小改动（5 分钟）：

```python
HOST = os.environ.get('CT_HOST', '127.0.0.1')   # 需要局域网时显式 CT_HOST=0.0.0.0
app.run(host=HOST, port=PORT, debug=False, threaded=True)
```
并在启动时打印 `http://127.0.0.1:5000`，避免用户不知道只能本机访问。

需要联网时的最小方案（不引入数据库）：

```python
import secrets, functools
from flask import session, redirect, url_for, request, jsonify

app.secret_key = os.environ['CT_SECRET']          # 首次启动生成并写入本机配置文件
PASS_HASH = os.environ['CT_PASS_HASH']            # PBKDF2/scrypt 哈希，不存明文

def login_required(f):
    @functools.wraps(f)
    def w(*a, **kw):
        if not session.get('uid'):
            if request.path.startswith('/api/'):
                return jsonify({"success": False, "message": "未登录"}), 401
            return redirect(url_for('login'))
        return f(*a, **kw)
    return w

@app.post('/api/login')
def api_login():
    d = request.get_json(silent=True) or {}
    if not verify(d.get('password', ''), PASS_HASH):
        return jsonify({"success": False, "message": "密码错误"}), 401
    session['uid'] = 'op'; session.permanent = True
    return jsonify({"success": True})
```
再给 `/main` 和全部 `/api/*` 加 `@login_required`，用 `waitress`（Windows 友好）替代 `app.run`：
`waitress-serve --host=0.0.0.0 --port=5000 web.app:app`。

**优先级与收益**：单机 P1（一行改动）；联网 P0。**不要**在没有服务端校验的情况下把端口开到局域网——这是本次最容易被忽视的安全面。

**归类：单机场景属于"可用但应加固"；多机场景是确需修复的缺陷。**

---

### 问题 6 — Excel"检测文件数"存在确定的统计错误

**判定：真实缺陷（P1）**

**证据**

- `app.py:476`（单图）与 `:627`（批量）：`parse_results_to_global(result, filename, '', batch_id)` —— **第三个参数 `filepath` 恒为 `''`**（路径直读模式 `:1286` 才传 `ap`）。
- `app.py:893`：`["检测文件数", len(set(item["文件全路径"] for item in rows))]` → 上传模式下所有行的该字段都是 `''`，去重后**恒为 1**。
- 更根本的问题：`parse_results_to_global`（`:318-321`）在 `len(boxes)==0` 时直接 `return 0`，**无缺陷图片根本不产生任何行**。所以 `rows` 只含"有缺陷的图×框"，无法作为总检测量的分母。
- 顺带：`_build_report_excel` 的表头（`:865`）不含"文件全路径"，用户也看不到这一列，无法自知统计口径。

**影响**：对外报表的核心指标错误。举例：一批 500 张、12 张有缺陷 → 报"检测文件数 1、检测目标总数 N"。如果拿这个表算漏检率/良率，分母完全错。

**修复**：引入文件级台账，与框级明细分离。

```python
# 新增：文件级记录（一次检测每个输入文件恰好一条）
file_records = []      # {批次ID, 文件路径, 文件全路径, 状态, 缺陷数, 模型, 阈值, 检测时间}

def add_file_record(batch_id, path, full_path, status, n_boxes, model_id, conf):
    file_records.append({
        "批次ID": batch_id, "文件路径": path, "文件全路径": full_path,
        "状态": status,                       # ok_defect / ok_clean / failed
        "缺陷数": n_boxes,
        "模型": model_id, "阈值": conf,
        "检测时间": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
```

汇总统计改为：

```python
n_files  = len(rows_files)                                    # 台账行数
n_fail   = sum(1 for r in rows_files if r["状态"] == "failed")
n_defect = sum(1 for r in rows_files if r["状态"] == "ok_defect")
n_clean  = sum(1 for r in rows_files if r["状态"] == "ok_clean")
summary = [
    ["检测文件数", n_files],
    ["　其中：有缺陷", n_defect],
    ["　其中：无缺陷", n_clean],
    ["　其中：检测失败", n_fail],
    ["检测目标总数", len(rows)],
    ["缺陷图片占比", f"{n_defect / max(1, n_files - n_fail) * 100:.2f}%"],
    ...
]
```

`file_records` 一并进 `_state.json`，历史批次也能正确重算。

**优先级与收益**：P1。收益：报表从"必然错误"变为可核对；同时为 #3（失败计数）、#11（电芯级漏检率）、#15（追溯）提供同一份数据底座。建议与 #3 合并成一次改造。

**归类：确需修复的缺陷。**

---

### 问题 7 — 训练默认分辨率与部署条件冲突

**判定：真实缺陷（P1），两个独立问题**

**证据**

1. `train.py:89` `def ask_imgsz(default=1280)`，`:92` 的提示语还写着"1280 最清楚但 s 模型会非常慢；1920 仅 n 模型"。而 `app.py:61` `INFER_SIZE = 1920`，注释明确"640 下几乎检不出缺陷"——正确结论只写进了推理端，没回写到训练端。
2. `train.py:118`：`imgsz = int(os.environ.get('CT_IMGSZ', 0)) or ask_imgsz()`，回车即 1280。**交互式脚本里"回车走默认"是最容易被误触的路径**，一次误触 = 数小时训练 + 训出的模型在 Web 上不可用（且不会报错，只是效果差），这是最坏的一类故障：**静默降级**。
3. `app.py:99-100`：`'names': Config.names, 'CH_names': Config.CH_names` —— 对所有模型**硬编码同一套类别名**，从未读取 `YOLO(path).names`。`:262`、`:339`、`:508` 全部按 `Config.CH_names[cls_id]` 翻译。

**影响**

- 1280 训练的模型在 1920 推理：不崩，但小缺陷尺度失配，召回明显掉，且**没有任何报错**。
- 类别顺序问题更隐蔽：若某模型训练时 `names = {0:'dark',1:'damage',2:'light'}`，网页会把暗斑显示成"极片损伤"，报表、导出文件名（`_export_worker` 的 `it['缺陷']`）、分组汇总全部跟着错。这是**会误导人工复核结论**的错误。

**修复**

```python
# train.py
DEPLOY_IMGSZ = 1920                  # 与 web/app.py 的 INFER_SIZE 单一来源，建议放 core/Config.py
def ask_imgsz(default=DEPLOY_IMGSZ):
    print(f'训练分辨率（必须与部署一致，当前部署为 {DEPLOY_IMGSZ}；改小会导致 Web 端明显漏检）')
    ...

# 训练结束/启用时校验并落盘训练元信息
meta = {"imgsz": imgsz, "model_pt": os.path.basename(model_pt), "epochs": args_epochs,
        "batch": batch, "names": names_ordered, "close_mosaic": 10,
        "git": git_rev(), "time": datetime.datetime.now().isoformat()}
json.dump(meta, open(os.path.join(os.path.dirname(best_pt), 'train_meta.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
```

```python
# app.py: 模型注册时读取真实类别表并校验
CANON = ['damage', 'dark', 'light']
EXPECTED_IMGSZ = 1920

def inspect_model(pt_path):
    m = YOLO(pt_path, task='detect')
    names = m.names                                   # {0:'damage', ...}
    order = [names[i] for i in sorted(names)]
    meta_path = os.path.join(os.path.dirname(pt_path), 'train_meta.json')
    meta = json.load(open(meta_path, encoding='utf-8')) if os.path.exists(meta_path) else {}
    warn = []
    if order != CANON:
        warn.append(f'类别顺序与系统约定不一致: {order} != {CANON}（界面翻译会错）')
    if meta.get('imgsz') and meta['imgsz'] != EXPECTED_IMGSZ:
        warn.append(f'训练分辨率 {meta["imgsz"]} ≠ 推理分辨率 {EXPECTED_IMGSZ}')
    if not meta:
        warn.append('缺少 train_meta.json，无法核对训练条件')
    return {'names': order, 'imgsz': meta.get('imgsz'), 'warnings': warn}
```
把 `warnings` 通过 `/api/models` 返回，前端在模型下拉里标红；**类别顺序不一致的模型直接拒绝启用**（而不是"翻译错但照跑"）。

**优先级与收益**：P1。收益：堵住"静默降级训练"和"类别错译"两条最难排查的故障路径，属于投入 30 行换长期可维护性。

**归类：确需修复的缺陷。**

---

### 问题 8 — 阈值滑块低于 0.25 时无法提高召回

**判定：真实缺陷（P1）**

**证据**

- `app.py:462`（单图）、`:592`（批量）、`:773`（视频）、`:1280`（路径直读）全部写死 `conf=0.25` 传给模型。
- 之后再用用户阈值二次过滤：`result_guolv_yolov8(results, conf_threshold)`（`:469`、`:608`、`:778`、`:1282`）。
- 因此有效阈值 = `max(0.25, 用户值)`。滑块设 0.15 → 实际 0.25，**低于 0.25 的候选框在 NMS 阶段就被丢弃了，不可能找回**。
- 附带：`conf_threshold = float(request.form.get('conf', CONF_DEFAULT))`（`:436/542/711/1199`）**没有任何校验**。传 `nan` → `boxes.conf >= nan` 恒 False → 全部框被过滤，界面显示"未检出"（又一次被 #3 的语义坑掩盖）；传 `inf` → 全部通过；传 `abc` → 500。

**影响**：低于 0.25 是排查漏检、做 PR 曲线、评估"放宽阈值能捞回多少"的**唯一手段**，现在这条路径被物理封死。而 `sweep_conf.py` 这类脚本是绕过 Web 直接调模型的，所以离线能扫、线上不能复现 —— 阈值决策与实际行为脱节。

**修复**（最小改动，去掉"双阈值"结构）

```python
# app.py
import math

def parse_conf(v, default=CONF_DEFAULT):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return round(min(1.0, max(0.0, f)), 4)

conf_threshold = parse_conf(request.form.get('conf'), CONF_DEFAULT)

# 直接交给模型，不再二次过滤（ultralytics 的 conf 同时作用于 NMS 与输出）
results = current_model(img, imgsz=INFER_SIZE, half=USE_FP16, device=DEVICE,
                        max_det=MAX_DET, verbose=False, conf=conf_threshold)[0]
```

若确实想保留"低阈值进 NMS、高阈值出结果"的行为（对密集小目标有利于 NMS 排序），则必须固定 `conf_nms = 0.01` 这类**低于任何用户可选值**的底线，且把两个值都记进结果元信息，避免再次出现"看起来调了、其实没调"。但**推荐前者**，简单且不会欺骗用户。

前端配套：滑块 `min` 改 0.05，并在标签上显示"低于 0.25 会显著增加误报"的提示，避免产线操作员把 0.05 当常规工作点。

**优先级与收益**：P1。收益：恢复"阈值—召回"这条可控曲线，让 `sweep_conf` 的离线结论可以在线复现；顺带修掉 `nan/inf` 导致的"全过滤/全通过"这类难以归因的异常。

**归类：确需修复的缺陷。**

---

### 问题 9 — 模型切换、检测、清空和状态保存存在竞态

**判定：真实缺陷（P1），但严重度低于你的描述**

**证据**

- `app.py:112-142` `load_model`：`:123-129` 先 `del current_model; current_model = None`，然后才 `YOLO(model_path)` 加载（大模型 3~10 秒）。窗口期内任何请求读到 `None`：
  - 单图 `:455` `current_model(...)` → `TypeError: 'NoneType' object is not callable` → 500；
  - 批量 `:585`、视频 `:766` 同理。
- 视频/批量不冻结模型引用：`:766` 每帧读全局 `current_model`，`:585` 每 chunk 读一次 → **一个视频可能混用两个模型**，统计口径无法解释。（路径直读模式 `:1260` 已经做了 `model = current_model` 捕获，说明这个坑作者已经踩过一次，只是没回推到其他入口。）
- `save_state`（`:168-176`）直接 `open(STATE_FILE,'w')` 覆盖写；`clear_results`（`:1465-1471`）在另一个线程 `os.remove(STATE_FILE)` 并删文件。两者无锁 → 可能写出半截 JSON（下次启动 `load_state` 静默失败，`except` 里只 print），或 clear 之后被一个在途的 `save_state` 复活部分数据。
- 附带性能问题：`save_state` **每次全量重写**，路径直读模式每 16 张落一次盘（`:1312`）。实测当前 `_state.json` 240 行/159KB；按 V4 规模（3753 张、假设 20% 有缺陷、每图 2 框）估算可到 2~5MB，3000+ 张时写入量是 O(n²) 级，长任务后期会明显拖慢。

**影响**：单机单用户场景下概率不高（切换模型时正好有人在检测），但一旦发生就是"500 报错"或"视频结果混模型"，且**日志里看不出来**。

**最小可行修复**（不上队列、不上 Redis，保持单进程）

```python
MODEL_LOCK = threading.RLock()
STATE_LOCK = threading.Lock()

def load_model(model_id, warmup=True):
    global current_model, current_model_id
    with MODEL_LOCK:
        if current_model_id == model_id and current_model is not None:
            return current_model
        new_model = YOLO(MODEL_REGISTRY[model_id]['path'], task='detect')   # 先在锁外准备
    if warmup:
        new_model([np.zeros((INFER_SIZE, INFER_SIZE, 3), np.uint8)] * BATCH_SIZE,
                  imgsz=INFER_SIZE, half=USE_FP16, device=DEVICE, verbose=False)
    with MODEL_LOCK:
        old = current_model
        current_model, current_model_id = new_model, model_id     # 原子换指针
    del old                                                        # 换完再释放
    ...
```
关键顺序：**先加载+预热，再切指针，最后释放旧模型**。原有的"先卸载"是为了省显存，但换来一个必然的空窗期；正确做法是短暂接受两份权重共存（n 模型 ~6MB，s ~22MB，完全可接受），换完再释放。

各入口统一"开工前取快照"：

```python
with MODEL_LOCK:
    model, mid = current_model, current_model_id
if model is None:
    return jsonify({"success": False, "message": "模型正在切换，请稍后重试"}), 503
```

状态原子写：

```python
def save_state():
    with STATE_LOCK:
        tmp = STATE_FILE + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump({...}, f, ensure_ascii=False)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, STATE_FILE)        # 原子替换，Windows 同盘可用
```
`clear_results` 也持 `STATE_LOCK`，并**不要** `os.remove` 结果文件（见红旗 C 的符号链接风险），改为先 `os.replace` 到 `results/_trash/<时间戳>/`。

另外建议按 #3 的改造把"检测任务"和"全局状态"解耦：任务级数据只追加，落盘节流为"每 N 批或每 M 秒"，避免 O(n²)。

**优先级与收益**：P1。收益：消除静默 500 与"混模型统计"；顺带解决长任务越跑越慢。真要做多用户并发再考虑受控推理队列，**单机不必**。

**归类：确需修复的缺陷，但不需要按你提的"受控推理队列"那么重**——加锁 + 快照 + 原子写就能覆盖 95% 的场景，属于"技术选型可以更轻"的部分。

---

### 问题 10 — 图级随机划分不能隔离相邻 CT 切片

**判定：真实缺陷，但性质比"相邻切片"更严重（P0）**

**证据（以及我做的实测）**

代码层面（`build_final_dataset.py`）：
- `:57-77` 按**单张图片**分层抽样进 val；`:19-20` 的 `md5` 只能排除**字节完全相同**的文件；
- `:74-75` `k = max(1, int(round(len(items)*VAL_RATIO)))`、`rng.sample(range(len(items)), k)` —— 完全没有"电芯/批次"这一维；
- `:51` `seen` 是全局内容哈希集合，正负样本一起去重，但同样只认字节相等。

**实测结论（重点）**：我对 `datasets/CT_DATASET_FINAL_V4` 做了两级核对：

1. `MD5(train) ∩ MD5(val) = 0` —— 没有字节完全相同的文件，这解释了为什么"MD5 去重"给了虚假的安全感；
2. 但用 256×256 归一化灰度做最近邻检索、再回原分辨率算 MAE：

```
646 张 val，与其最近 train 图（尺寸一致，645/646）
  全分辨率 MAE 中位数 = 0.39（灰度 0~255 尺度）
  MAE < 1: 534 张 (82.7%)   ← 仅剩 JPEG 重编码噪声，即同一切片
  MAE < 2: 584 张 (90.4%)
  MAE < 5: 629 张 (97.4%)
```

`MAE≈0.1~0.4` 在 8bit 灰度上等价于"同一张图被存了两次"（对比：`0_0010` 与 `0_0015` 的真实不同切片 MAE=11.5、P99=137）。最相似的一组：`val/neg_9_0652.jpg` ≈ `train/neg_9_0651.jpg`，MAE=0.05 —— **编号连续、内容重复**。

所以真实情况不是"相邻切片可能泄漏"，而是：**同一张切片以不同文件名同时存在于 train 与 val，占比 82.7%**。这直接解释了 V4 val 上指标的乐观偏差，也是红旗 A 的成因之一（考卷就是从这个池子里抽的）。

**影响**：val mAP/召回被系统性高估；早停（`patience=30`）会在"验证集已见过"的条件下选择 epoch，等价于**用训练集做模型选择**；`best.pt` 的选择过程本身失真。

**修复**

```python
# 1) 分组键：优先用电芯/扫描批次，其次用文件名前缀+源目录
def group_key(img_path, src_folder):
    """返回不可跨组泄漏的分组标识。同一电芯的所有切片必须同组。"""
    # 例：积攒/20260810-P1出样CT/组A/正面/x.jpg -> "20260810-P1出样CT/组A"
    rel = os.path.relpath(img_path, src_folder).replace('\\', '/')
    parts = [p for p in rel.split('/') if p]
    return '/'.join(parts[:-1]) or 'root'      # 去掉文件名，保留目录层级=批次/电芯

# 2) 按组划分，且按类别分层，保证 val 里三类都有
from collections import defaultdict
groups = defaultdict(list)
for it in items:
    groups[group_key(it['img'], SRC)].append(it)

gkeys = sorted(groups)
rng.shuffle(gkeys)
k = max(1, int(round(len(gkeys) * VAL_RATIO)))
val_groups = set(gkeys[:k])
for g in gkeys:
    split = 'val' if g in val_groups else 'train'
    ...

# 3) 兜底：分组后仍要做内容级查重（跨 split 的近重复直接剔除或并入同一侧）
#    用 256x256 z-score 余弦 + 原分辨率 MAE<2 双阈值，脚本化，结果落 manifest
```

同时**必须产出一份 split 清单**（这是可复核的前提）：

```csv
文件名,源路径,组键,split,md5,内容指纹(256x256 hash),框数,各类框数
0_0001.jpg,问题电芯图片汇总/0-极片损伤/0_0001.jpg,0-极片损伤/train,0123abc...,...,1,1/0/0
```

并且**冻结一个独立测试集**：从**从未进入 train/val 的电芯/批次**里划出（例如按生产日期整批留出），其 MD5 与内容指纹对所有训练数据做白名单校验，加进 CI —— 任何"考卷被复制进训练集"的操作都会被拦下。

**优先级与收益**：P0。这是本次评审里**唯一能直接推翻现有全部指标**的问题。修复成本主要在"定义分组键"（需要业务确认电芯编码规则）+ 一次数据集重建；收益是让 mAP 第一次具备参考价值。

**归类：确需修复的缺陷（比原判断更严重）。**

---

### 问题 11 — 宽松匹配指标容易掩盖定位问题

**判定：真实缺陷（P0），且与 #10 叠加后指标完全失效**

**证据**

- `eval_loose.py:27-28`：
  ```python
  def loose_match(pb, gb):
      return iou(pb, gb) >= 0.1 or center_in(pb, center(gb)) or center_in(gb, center(pb))
  ```
  `center_in(gb, center(pb))` 这一项允许"预测框中心落在 GT 框内"——一个覆盖整幅 1929×1198 的预测框，只要中心落在某个 30px 的暗斑框内就算命中。`analyze_v4_val.py:30` 同样实现。
- 匹配是**按预测顺序贪心**（`eval_loose.py:50-54`），没有按 `conf` 降序，也没做全局最优（匈牙利）——同一个 GT 可能被低分框先占掉，导致 FP/FN 的分配随模型输出顺序波动，**不可复现**。
- 口径不可比：`analyze_v3_val.py:15` 用 `IOU = 0.5`，V4 用 0.1/中心互落。两者的 P/R 不能并列比较，而汇报材料里恰恰是并列的（`app.py:144` 注释："全面优于 V3"）。

**影响**：IoU≥0.1 对"极片损伤"这类长条形缺陷尤其宽松（长条框轻微错位时 IoU 仍可能 >0.1）；"中心互落"对大框更宽松。结果是指标对**定位质量不敏感**，无法发现"框偏了一半"这类真实退化。叠加 #10 的 82.7% 重复，现有 P/R 既不可比也不可信。

**修复**：分三层报告，宽松指标降级为辅助。

```python
# 层 1：标准 COCO 指标（直接复用 ultralytics，避免自研口径）
m = YOLO(weights)
r = m.val(data='data_frozen.yaml', imgsz=1920, conf=0.001, iou=0.6,
          split='test', plots=False)     # 输出 mAP50 / mAP50-95 / 每类 AP
# 层 2：固定工作点（上线阈值）下的分类 P/R + 每千张误报
#   匹配规则：IoU>=0.5、同类、按 conf 降序贪心（或用 scipy.optimize.linear_sum_assignment 求全局最优）
# 层 3：业务口径
#   - 电芯级漏检率 = 有缺陷电芯中"至少一个缺陷未被检出"的比例
#   - 每千张误报数 FP/1000img（产线最关心的成本指标）
#   - 宽松命中率（保留，但明确标注"仅业务参考，不可与 V3 比较"）
```

报告模板建议固定成一张表（同一冻结测试集、同一 imgsz、同一 conf）：

| 模型 | mAP50 | mAP50-95 | 损伤 P/R | 暗斑 P/R | 亮斑 P/R | FP/千张 | 电芯漏检率 |
|---|---|---|---|---|---|---|---|

并把这张表写进 `runs/` 与 `models/启用/<模型>/eval.json`，取代散落在注释里的数字。

**优先级与收益**：P0（与 #10 同批）。收益：让"V3→V4 到底有没有变好"这个问题第一次有答案；同时把"每千张误报"引入决策，这是产线唯一真正在意的成本项。

**归类：确需修复的缺陷（评估体系层面）。**

---

### 问题 12 — 标注转换和文件命名缺少严格校验

**判定：真实缺陷（P1），四个子问题都成立**

**证据**（`build_final_dataset.py`）

1. `:22-42` `json_to_yolo`：
   - `cls = int(s['label'])`，`except (ValueError, KeyError): continue` —— 若标注文件里 `label` 是 `"damage"` 这类英文，会**静默丢弃该框**，不计数不报警（`:128` 只统计"无框"张数，看不出丢框）；
   - **没有类别范围校验**：`cls=9`（负样本误入）会被写进 YOLO 标签，随后 `:89` `cls_boxes[split][cls] += 1` 直接 `IndexError` 崩在半路（留下一个不完整的输出目录，下次重跑又被 `:45` 的"已存在"挡住）；
   - 没有 `shape_type` 校验：polygon/rotation 形状会被 `min/max` 当外接矩形处理（对旋转框是错的），`line`/`point` 也会被当退化框；
   - 没有 `imageWidth/Height` 与真实图片尺寸一致性校验（`:25` 直接信任 json，若标注时图片被替换，坐标全错）；
   - 没有 `w/h > 0` 之外的上限与越界比例检查。
2. `:74-75` 空类别目录：`k = max(1, ...)`，`len(items)==0` 时 `k=1`，`rng.sample(range(0), 1)` → `ValueError: Sample larger than population`。今天没崩只是因为三个目录都非空；一旦新数据集某一类暂时为空（很常见的中间状态），脚本直接失败。
3. `:81-82` 命名清洗：`stem.replace(" ","_").replace("(","").replace(")","")` —— `a(1)` 与 `a1` 会映射成同一个 `safe`，随后 `:84-88` **静默覆盖**，图片和标签一起丢，且不计数。
4. `import_new_labels.py:146-147`：`remap = len(ids) == 1` → 单类别时**无条件改写**为该文件夹对应的类别。设计意图是"防标注软件编号漂移"，但副作用是：一张实际标错类别的图（比如暗斑被标成 damage）会被**自动"修正"成文件夹类别**，错误被永久固化、且不产生 warning。

**影响**：这些都是"静默数据损坏"，比报错更危险——数据集看起来构建成功，实际少框/错类/丢图。

**修复**

```python
class LabelError(Exception): pass

def json_to_yolo(jpath, nc=3, strict=True):
    d = json.load(open(jpath, encoding='utf-8'))
    W, H = d['imageWidth'], d['imageHeight']
    boxes, errs = [], []
    for i, s in enumerate(d.get('shapes', [])):
        st = s.get('shape_type', 'rectangle')
        if st not in ('rectangle',):                     # 明确拒绝，而不是静默退化
            errs.append(f'{jpath}#{i} 不支持的 shape_type={st}'); continue
        try:
            cls = int(s['label'])
        except (TypeError, ValueError):
            errs.append(f'{jpath}#{i} label 不是类别序号: {s.get("label")!r}'); continue
        if not (0 <= cls < nc):
            errs.append(f'{jpath}#{i} 类别越界: {cls}'); continue
        pts = s.get('points') or []
        if len(pts) < 2:
            errs.append(f'{jpath}#{i} 点数不足'); continue
        xs = [float(p[0]) for p in pts]; ys = [float(p[1]) for p in pts]
        if min(xs) < -1 or min(ys) < -1 or max(xs) > W + 1 or max(ys) > H + 1:
            errs.append(f'{jpath}#{i} 坐标越界 [{min(xs):.1f},{min(ys):.1f},{max(xs):.1f},{max(ys):.1f}] vs {W}x{H}')
        x1, x2 = max(0.0, min(xs)), min(float(W), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(H), max(ys))
        if x2 - x1 < 1 or y2 - y1 < 1:
            errs.append(f'{jpath}#{i} 退化框'); continue
        boxes.append((cls, (x1+x2)/2/W, (y1+y2)/2/H, (x2-x1)/W, (y2-y1)/H))
    if errs and strict:
        raise LabelError('\n'.join(errs))
    return W, H, boxes
```

配套：
- `if not items: print(f'{folder}: 空目录，跳过'); continue`（修 `:74` 崩溃）；
- 稳定唯一名：`safe = f'{folder}_{stem}'` 并**不依赖清洗去重**，在写入前统一检查 `dst in written: raise`（`:81` 的碰撞）；
- 图片尺寸与 json 的 `imageWidth/Height` 必须相等，不等则进错误清单（这条能抓住"标注后替换图片"）；
- `import_new_labels.py` 的类别改写降级为 `--assume-single-class` 显式开关，默认保留原编号 + 输出 warning 清单。

**优先级与收益**：P1。收益：把"静默丢框/丢图/错类"变成可见错误，数据集构建从"看起来成功"变成"可验证成功"。

**归类：确需修复的缺陷。**

---

### 问题 13 — 便携路径、环境和配置缺少统一依据

**判定：真实缺陷，而且已经发生（P1）**

**证据**

- 硬编码 `D:\CT Detection`（而非当前 `D:\CT Detection-yolo`）：
  `harvest.py:11`、`prelabel.py:8`、`exec_v4_review.py:6`、`merge_v4_staging.py:6`、`eval_loose.py:8`、`analyze_v4_val.py:8`、`analyze_v3_val.py`（同款）。
  且**旧目录真实存在**（`ls "/d/CT Detection"` 有完整工程树），所以这些脚本不会报错，而是**默默操作另一套数据**——这是最坏的情况：不报错、结果看似正常、实际改错地方。
- 生成的 YAML 用绝对路径：`build_final_dataset.py:118` `path: {OUT}`、`augment_final.py:144` 同款。数据集一旦移动/拷给别人即失效（YOLO 会用 `path` 定位，绝对路径不存在直接报错）。
- `Config.py:7,10` 与实际不符：`model_path` 指向不存在的 `CT_DATASET_AUG_0910_1419/best.pt`，`data_ymal_path` 指向 `CT_DATASET_AUG`，而 Web 用的是 `20261019_V4`（`app.py:146`）。`检查环境.py:103,122,138` 依赖这两个字段 → 体检结论与真实运行状态无关。
- `models/` 为空（红旗 B）、`web/results` 为跨工程符号链接（红旗 C）。

**影响**：换机、改名、拷给同事、归档，任一动作都会踩到。已经造成"环境体检报错但服务其实能跑""脚本改了另一套数据"这类难以解释的现象。

**修复**

```python
# 所有脚本统一（core/paths.py）
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = PROJECT_ROOT / 'datasets'
MODELS_ENABLED = PROJECT_ROOT / 'models' / '启用'
RESULTS = PROJECT_ROOT / 'web' / 'results'
```
- `data.yaml` 用相对路径（`path: .` + `train: train/images`），或干脆不写 `path`（ultralytics 支持以 yaml 所在目录为基准的相对路径）；
- `Config.py` 只保留**一份**默认模型来源：`DEFAULT_MODEL_ID = 'models/启用/20261019_V4/best'`，`app.py:146` 和 `检查环境.py` 都从这里取，删掉 `model_path` 这个影子字段；
- 发布包附带 `release_manifest.json`：Python/torch/ultralytics 版本、`pip freeze` 摘要、每个权重的 `sha256` + `train_meta.json`（imgsz/epochs/batch/names）、数据集清单（每个 split 的文件数、各类框数、内容指纹）；启动时校验一遍并打印不一致项。
- 修红旗 C：`web/results` 换成本工程内真实目录。

**优先级与收益**：P1。收益：消除"改错目录"这类无日志事故；让发布包可复现（这也是 #15 追溯能力的基础）。

**归类：确需修复的缺陷。**

---

### 问题 14 — 增强策略需要通过对照实验确认

**判定：部分真实 —— 拆成三个独立问题，其中一个是确定缺陷**

**(a) 强预处理只作用于训练图 → 训练/推理分布不一致（确定缺陷，P1）**

`augment_final.py:113-114` 在 `PREPROCESS` 时对训练图做 `combo_v2`，而 `:133-140` 的 val 是**逐字节复制**原图，Web 端（`app.py`）也完全不做预处理。于是模型学的是"CLAHE+拉伸+锐化后的图"，推理时喂的是原图——这属于训练/服务特征不一致（training-serving skew），是导致线上效果远低于离线指标的最常见原因之一。历史记录说"该路线效果不佳"，从代码看，**原因很可能不是增强本身无效，而是只在训练侧增强**。

修法二选一，且必须显式记录：
1. **不要**在离线增强里做像素级强预处理，改由训练时 `hyp`（`hsv_v`、`clahe` 等）统一施加，验证也走同一 transform；
2. 若确实需要 CLAHE 类增强，则把它固化到**推理链路**（`app.py` 在读图后调用同一个 `preprocess_ct_image`），并写进 `train_meta.json` 的 `preprocess` 字段，让"该模型需要预处理"成为模型属性而非口头知识。

**(b) 正样本 5×、负样本 2× 改变样本比例（可接受的权衡，但需声明）**

`augment_final.py:117-131`：有框图 → 原图 + 4 变体（5×）；无框图 → 原图 + 1 翻转（2×）。若原始正:负 = 1:3，增强后变成 5:6。这**本身是有意为之**（缓解正样本稀缺），属于可接受权衡；但它会改变"每千张误报"这类业务指标的先验，所以必须在 `train_meta.json` 和评估报告里写明增强后的比例，且**评估必须在未增强的原始分布上做**（现在 val 未增强，这点是对的）。

**(c) `close_mosaic` 在早停前根本不会触发（确定缺陷，P2）**

`train.py:160` `close_mosaic=10`、`:146` `epochs=300`、`:150` `patience=30`。ultralytics 的实现是"在 `epoch == epochs - close_mosaic` 那一刻关闭 mosaic"，即**第 290 轮**。你引用的"第 75 轮提前结束"正是典型情形：best 出现在 ~45 轮，75 轮触发早停 → mosaic **从未关闭**。也就是说这个参数在当前配置下是**无效配置**，而 `docs/preprocess_guide.md:68` 还把它当作"关键改进参数"。历史训练若都在 100 轮内早停，则"关闭 mosaic 稳定训练"这个假设从未被验证过。

修法：用回调按**实际进度**关，而不是按总轮数。

```python
from ultralytics.utils import callbacks

def close_mosaic_cb(trainer):
    total = trainer.epochs
    if trainer.epoch == max(1, int(total * 0.85)) - 1:      # 相对 85% 位置
        trainer.train_loader.dataset.close_mosaic(hyp=trainer.args)
        print(f'[Mosaic] 已关闭 @ epoch {trainer.epoch + 1}')

model.add_callback('on_train_epoch_start', close_mosaic_cb)
```

**更值得讨论的一点（超出你的清单，但影响更大）**：`train.py:174` `mosaic=1.0` + `imgsz=1920`。mosaic 会把 4 张图各缩到约 960×960 再拼，**等效把缺陷尺寸缩小一半**——这与"必须 1920 才能检出小缺陷"的核心结论直接冲突。对这类小目标缺陷，`mosaic=1.0` 很可能是净负收益。建议把 `mosaic` 纳入对照矩阵（见下），而不是只对照"无增强/温和增强"。

**(d) 对照实验设计（固定测试集之后才能做）**

| 实验 | 离线增强 | mosaic | 预处理 | 观测指标 |
|---|---|---|---|---|
| E0 基线 | 无 | 1.0 | 无 | mAP50-95 / FP千张 / 电芯漏检率 |
| E1 | 温和（修好#1后） | 1.0 | 无 | 同上 |
| E2 | 温和 | 0.0 | 无 | 同上 |
| E3 | 温和 | 0.0 | 训练+推理同构 | 同上 |
| E4 | 无 | 0.0 | 无 | 同上 |

每档固定随机种子、固定 epoch 预算（建议 120，而不是 300+早停，便于公平比较）、固定测试集。用同一份 `eval.json` 模板出表。

**优先级与收益**：P1（a）、P1（c 的修法 + mosaic 复核）、P2（b 仅需文档化）。收益：把"增强有没有用"从猜测变成可复现结论；预计能解释一部分"离线好、线上差"的差距。

**归类：(a) 确需修复的缺陷；(b) 可接受的权衡（需声明）；(c) 配置失效，属确定缺陷；(d) 属于方法论要求。**

---

### 问题 15 — 导出的"原图"无法完整追溯输入

**判定：真实缺陷（P1）**

**证据**

- `app.py:246-255`：
  ```python
  orig_filename = f"orig_{uuid.uuid4().hex}.jpg"
  Image.fromarray(rgb_img).save(orig_path, 'JPEG', quality=95)
  ```
  `rgb_img` 来自 `:448` `np.array(Image.open(file.stream).convert('RGB'))`。所以"原图"实际是**解码 → 转 RGB → 再 JPEG 压缩**的副本：不是原始字节，且**多了一次有损压缩**（quality=95 仍有损失，标签复核时看的是二次压缩图）。
- `app.py:970-988`（打包）：
  ```python
  base, ext = os.path.splitext(it['文件名'])      # ext 来自"上传时的文件名"
  ...
  zf.write(p, f"{zip_dir}/{name}_原图{ext}")       # 但 p 的内容永远是 JPEG
  ```
  上传 `x.bmp` / `x.png` → 导出包里的 `x_原图.bmp` 实际是 JPEG 数据。第三方工具按扩展名解析会失败或显示异常。
- 结果行字段（`:332-347`）只有 `组别/文件路径/类别/置信度/坐标/时间`，**没有模型标识、阈值、imgsz、预处理模式、输入哈希**。同一批结果无法回答"这是哪个模型、什么工作点下得到的"。
- 高位深输入（我在本机实测）：

```
16bit PNG → PIL mode I;16 → convert('L') / convert('RGB')
原值 [0, 1000, 4000, 30000, 65535]
转换后 [0, 255, 255, 255, 255]      ← 直接截断，不是 >>8 缩放
```
也就是说：**任何 12/16bit 的 CT 原图，经当前读图链路会变成几乎全白的图**（1000 以上全部饱和到 255），且不报错。当前产线图是 8bit JPEG 所以未触发，但"CT 原始导出常见 16bit"这件事意味着这是一颗定时炸弹。

**影响**：复核时看到的"原图"不是原始数据；导出包扩展名与内容不符；一旦需要回溯"某批结果是用哪个模型跑的"就无从查起（这在客户投诉/内部复盘时是刚需）；16bit 输入直接失效。

**修复**

```python
# 1) 保留原始字节（上传侧）
raw = file.read()                        # 一次性读入（注意 MAX_CONTENT_LENGTH 已设 10GB，建议按需分块落盘）
h_in = hashlib.sha256(raw).hexdigest()
with open(os.path.join(RESULTS_FOLDER, f"orig_{h_in[:16]}{ext}"), 'wb') as f:
    f.write(raw)                          # 字节级一致，扩展名来自真实文件头

# 2) 扩展名以文件头为准，不信任上传名
def sniff_ext(raw):
    if raw[:3] == b'\xff\xd8\xff': return '.jpg'
    if raw[:8] == b'\x89PNG\r\n\x1a\n': return '.png'
    if raw[:2] == b'BM': return '.bmp'
    if raw[:4] in (b'II*\x00', b'MM\x00*'): return '.tif'
    return '.bin'
```
（注意：上传的是 `file.stream`，当前 `:448` 只读一次就丢弃了原始字节；要保留必须改成"先落盘/读入 bytes，再从 bytes 解码"，同时保持内存占用可控。）

```python
# 3) 每张图/每批次记录检测条件（一次写入，不要每行重复）
result_meta = {
    "batch_id": batch_id,
    "model_id": current_model_id,
    "model_sha256": MODEL_SHA[current_model_id],     # 启动扫描时算好
    "imgsz": INFER_SIZE, "conf": conf_threshold,
    "half": USE_FP16, "device": str(DEVICE),
    "preprocess": None,                               # 若启用预处理，写明 mode
    "input_bit_depth": 8,
    "input_sha256": h_in,
    "upload_name": file.filename,
}
```

```python
# 4) 高位深显式归一化（若将来接入 16bit）
def decode_image(raw, bit_depth=None):
    arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None: raise ValueError('解码失败')
    if arr.dtype == np.uint16:
        # 用 16->8 位缩放（>>8）而非截断；并把位深记录进 meta
        arr = (arr >> 8).astype(np.uint8)
        bit_depth = 16
    elif arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr, bit_depth
```
**关键**：`IMREAD_UNCHANGED` + 显式缩放，替代 `PIL.convert('RGB')` 的隐式截断；并把 `bit_depth` 与归一化方式计入 `result_meta`，让"灰度/位深转换规则"成为可审计的记录而不是隐式行为。

**优先级与收益**：P1。收益：导出物与原始输入字节一致（复核可信）、可按模型/阈值重放任意历史批次、为 16bit 输入预留正确通路（避免未来的"莫名全白"事故）。

**归类：确需修复的缺陷。**

---

### 问题 16 — 现有测试与文档已经落后于接口

**判定：部分真实（P2），其中一条与现状不符**

**证据**

- **`test_real.py` 在本仓库不存在**（`scripts/` 只有 `test_fp16.py`、`test_web_history.py`、`test_web_pathmode.py`；在 `D:\CT Detection-yolo` 全库与旧目录 `D:\CT Detection` 中搜索 `test_real`，只命中 `runtime/site-packages` 里的第三方测试）。所以"`test_real.py:37` 仍期待 `task_id`"这条**无法在当前代码上验证**——要么该文件已被删除，要么在别的副本里。但结论方向仍成立：`/api/export`（`app.py:1035-1051`）现在**同步返回 `download_url`**，而打包接口 `/api/download-all` 走异步 `task_id`（`:1492`）。两个导出路径语义不同、命名相近，接口层面已发生过语义漂移。
- 现有两个测试（`test_web_history.py`、`test_web_pathmode.py`）确实是真实集成测试（`:74` 起 `subprocess.Popen` 起真服务 + 真模型），覆盖批次迁移、逐条删除、路径模式。这是加分项。
- 但它们**必须依赖真实模型**：`:75-77` 直接 `web/app.py`，而 `app.py:145-151` 在导入阶段就 `load_model()`。当前 `models/` 为空 → 测试连服务都起不来（`raise SystemExit('server died')`）。也就是说**测试在当前状态下全部不可运行**。
- 文档错误（`docs/preprocess_guide.md`）：
  - `:12` `python preview_preprocess.py`、`:25` `python augment_dataset.py --factor 4` —— 两个脚本都**不存在**（`scripts/` 下无此文件，实际为 `augment_final.py` 且参数语义不同）；
  - `:48` `0: crease` —— 与 `Config.py:16,19`（`damage/极板损伤`）不一致，`damage` 从未叫过 `crease`；
  - `:74-83` 指导在 `MainProgram.py` 里加预处理，而该文件已不存在（Web 已重构为 `web/app.py`）。

**影响**：新人（或未来的你）按文档操作会直接失败；测试不可运行等于没有回归网；接口语义漂移没有被任何测试锁定。

**修复**

1. **先让测试能跑**：把 `app.py` 重构成工厂函数，模型可注入。

```python
def create_app(model_path=None, model_dir=None):
    app = Flask(__name__, ...)
    ...注册路由...
    mid = init_model(model_path or pick_default_model(model_dir))
    if mid is None:
        app.config['MODEL_ERROR'] = '无可用模型'
    return app

app = create_app() if os.environ.get('CT_EAGER_APP') == '1' else None

if __name__ == '__main__':
    create_app().run(...)
```
再提供一个 3 秒就能训出来的极简权重（或直接 `YOLO('yolov8n.yaml')` 随机初始化）供测试注入，测试从此不依赖生产权重。

2. **补五组小测试**（对应你列的清单，我按"能抓住回归"的标准写验收条件）：

| 测试 | 输入 | 断言 | 能抓住的回归 |
|---|---|---|---|
| 旋转框一致性 | 合成黑图 + 白块，±3/±5/±15° | 由像素反算的框与返回值 IoU>0.9 | #1 符号错误、越界裁剪 |
| 中文路径读写 | 临时目录名含中文 + 图 | `imread_cn/imwrite_cn` 成功；`save_annotated_image` 在只读目录返回 `None` 而非假文件名 | #2 |
| 非法标签 | json 含 `label=9` / `label="damage"` / polygon / 越界坐标 | 抛 `LabelError` 且错误清单含该文件名 | #12 |
| 批量失败状态 | 3 张图其中 1 张损坏 | 响应 `failed==1`、`ok_clean+ok_defect==2`、`results` 与 `chunk` 下标对齐 | #3、#6 |
| 导出统计 | 5 张图（2 缺陷 / 2 干净 / 1 失败） | Excel `检测文件数==5`、`失败==1`、`缺陷图==2` | #6 |

3. **文档修正**：`preprocess_guide.md` 的脚本名、类别名、`MainProgram.py` 全部替换为当前实现；把"旋转增强""阈值语义""导出接口返回 `download_url`"写进接口约定。

**优先级与收益**：P2（但"测试可运行"应提前到 P1 一起做，否则后面的修复没有回归保护）。收益：把上面 15 条修复固化下来，避免下次接手时重新踩一遍。

**归类：文档/测试属"确需修复"（成本低）；`test_real.py` 一条描述与现状不符，需要你确认是否指旧副本。**

---

## 3. 哪些是确需修复，哪些是可接受的权衡

### 必须修（按批次）

| 批次 | 条目 | 理由 |
|---|---|---|
| **第 1 批：让结论可信** | #10 分组划分 + 冻结测试集 → #11 标准指标 → 红旗 A 重出全部数字 | 不做这步，后面的优化都无法判断是否有效 |
| **第 2 批：止血** | #3 失败态 + #6 文件台账（同一次改造）、#4 复核脚本 dry-run+备份、#1 旋转、#8 阈值 | 数据安全 + 结论正确性 + 训练数据质量 |
| **第 3 批：工程化** | #2 imgio、#7 训练/类别校验、#9 加锁+原子写、#13 路径与 manifest、#15 原图与追溯 | 消除静默故障与换机故障 |
| **第 4 批：方法论** | #14 对照实验（含 mosaic 复核）、#12 标注校验、#16 测试与文档 | 长期可维护性 |

### 可接受的权衡（建议在文档中显式声明，而不是"就这样"）

1. **前端登录 + 纯单机**：作为"防误操作"可接受，但**必须把监听地址改成 127.0.0.1**（1 行）。保持 0.0.0.0 且无服务端校验，在任何"能 ping 到"的网络里都不接受。
2. **干净图不落盘**（`app.py:631-641` 注释"省 100~300ms/张"）：这是很好的性能权衡，产线多数是干净图。前提是**必须有文件级计数**（#6）来补回信息，否则"没落盘"就退化成"没检测"。
3. **单进程 + 全局状态，不上 Redis/队列**：单机完全够用。加锁 + 引用快照 + 原子写即可；#9 里"受控推理队列"对当前规模是过度设计。
4. **贪心一对一匹配**：可接受，但必须**按 conf 降序**（现在是按输出顺序），否则 FP/FN 分配不稳定、结果不可复现。宽松命中率本身可以留作业务参考，只要不与 V3 的 IoU 0.5 并列。
5. **只导出损伤类**（`harvest.py:16` `EXPORT_CLS={0}`）：业务选择，可接受；但导出 CSV 里已记录 `>=0.30` 的全部类别（`:15,74`），信息没丢，做法是对的。
6. **val 不做增强 / 不做像素扰动**：**这是正确的**。问题不在 val，而在训练侧单独加了强预处理却没同步到推理侧（#14a）——别把正确做法当成缺陷改掉。
7. **二次阈值过滤本身**：可以保留（例如为了"NMS 用低阈值排序、输出用高阈值"），但底线必须低于用户可选范围，且两个阈值都要记录。当前的 0.25 实质是"用户不可见的硬下限"，这不可接受。

### 你判断需要修正的两处

- **#10**：不是"相邻切片可能泄漏、需取数据验证"，而是**已实测 82.7% 的 val 与 train 是同一切片的重复副本**；且"考卷"100% 来自 V4。泄漏已发生且程度严重。
- **#16**：`test_real.py` 在本仓库不存在，该条依据需要你确认出处；但"接口返回 `download_url`、文档引用不存在的脚本、导入即加载模型导致测试不可运行"这三项均成立。

---

## 4. 建议的验收顺序（每步都有可观测产物）

1. `test_rot_label_follows_pixels` 由红转绿 → 重建 `CT_DATASET_FINAL_*_AUG`（`AUG_RESUME` 不能续跑，需清空重跑）。
2. 数据集重建后输出 `split_manifest.csv`，并断言 `train/val/test` 的组键两两不相交、内容指纹两两不相交；`考卷` 与任何 split 的 MD5/指纹交集为 0。
3. 在冻结测试集上跑 `eval_report.json`（mAP50、mAP50-95、三类 P/R@工作点、FP/千张、电芯漏检率），与旧数字并列一表存档 —— **预期旧数字会明显下降，这是正常的**。
4. 批量/视频的失败态测试：注入 1 张损坏图 + 1 个损坏视频，断言 `failed==1`、`ok_clean+ok_defect==2`、视频任务 `status=='error'`。
5. 导出 Excel 断言四个计数（总/有缺陷/无缺陷/失败）与输入一致。
6. 复核脚本 dry-run 输出清单与 `--apply` 后备份目录同时存在，`git status` 无意外改动。
7. （仅当需要联网时）未登录访问 `/api/clear` 返回 401。

---

## 附：本次实测命令与结果（可复现）

```
# 中文路径（实测 40/40 失败 vs 40/40 成功）
cv2.imread('datasets/考卷_清洗版/images/0_0023.jpg')                → None  (WARN can't open/read file)
cv2.imdecode(np.fromfile(..., np.uint8), cv2.IMREAD_COLOR)          → 正常

# 考卷污染（MD5，字节级）
考卷 196 张：= V4 train 66 张，= V4 val 130 张，合计 196 = 100%

# V4 val 泄漏（256×256 最近邻 + 原分辨率 MAE）
646 张 val，645 张尺寸一致；MAE 中位数 0.39
MAE<1: 534 (82.7%) | <2: 584 (90.4%) | <5: 629 (97.4%)
对照：两张真实不同切片 MAE=11.5, P99=137

# 16bit 截断
I;16 [0,1000,4000,30000,65535] --convert('L')--> [0,255,255,255,255]
```
