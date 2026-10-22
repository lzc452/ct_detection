# 批 A · 损伤标注一致性审计作业单（test 全量 73 框）

> 本单为**盲审**：不提供原标注的坐标、尺寸、数量。请独立判断。

## 一、要做什么

对每张裁剪图，标注出图中**所有**你认为属于「极片损伤 / 划痕」的区域，按你判断的**完整边界**画矩形框。

**不要**参考任何已有标注，**不要**与他人讨论。两个人各自独立完成。

## 二、怎么操作

1. `labelme` 打开裁剪图目录（见下表 `裁剪图` 列所在的文件夹）；图幅偏小属**正常**（刻意只放大 2×，避免把标注精度人为抬高），用滚轮放大查看细节即可
2. 逐张画框，**标签名按下表规范填**（这一步比画框本身更重要）
3. 若某张图确实看不到任何损伤，**不要画框**，直接保存空 json
4. 全部做完后把整个 json 目录交回

### 标签名规范（用来区分「边界模糊」和「标注马虎」）

| 标签名 | 什么时候用 |
|---|---|
| `damage` | 痕迹清楚，边界能确定，换个角度看也是这条 |
| `damage_soft` | 能看到痕迹，但边界只能靠判断，差几个像素很正常 |
| `damage_unsure` | 几乎看不出，凭经验认为这里有问题 |

> 千万不要为了让框"好看"而把边界画得比实际更整齐。这份数据要测的正是边界有多不确定，画得越像标准答案，结果越没用。

> 坐标换算由脚本完成（裁剪偏移见下表），你**不需要**换算或填写坐标。

## 三、这份数据用来判什么

| 指标 | 含义 | 门槛 |
|---|---|---|
| IoU(A,B) 中位 | 两个人对同一划痕边界的一致程度 | ≥0.75 通过 |
| | | <0.65 停止模型侧优化，先修标注规范 |

判读逻辑：如果人工彼此都画不到 0.65，那么模型 mAP 的天花板里有一部分属于「标注定义本身模糊」，不是模型缺陷。

## 四、为什么 test 要全审而不是抽

test 的损伤框共 73 个，是 mAP/召回/混淆矩阵的**唯一**样本。抽 30 个算中位数，95% 置信区间约 ±0.07，正好盖住 0.65 与 0.75 这道分界，等于问了也答不了。73 个全审的工作量约 1 小时/人，性价比远高于抽样。

## 五、逐框清单（73 框）

| 编号 | 裁剪图 | 裁剪偏移(原图 x0, y0) | 放大 | 画好了 |
|---|---|---|---|---|
| A001 | `reports/audit/crops/A001.jpg` | 890, 468 | 2× | ☐ |
| A002 | `reports/audit/crops/A002.jpg` | 850, 293 | 2× | ☐ |
| A003 | `reports/audit/crops/A003.jpg` | 857, 295 | 2× | ☐ |
| A004 | `reports/audit/crops/A004.jpg` | 850, 294 | 2× | ☐ |
| A005 | `reports/audit/crops/A005.jpg` | 854, 278 | 2× | ☐ |
| A006 | `reports/audit/crops/A006.jpg` | 1130, 333 | 2× | ☐ |
| A007 | `reports/audit/crops/A007.jpg` | 856, 333 | 2× | ☐ |
| A008 | `reports/audit/crops/A008.jpg` | 416, 114 | 2× | ☐ |
| A009 | `reports/audit/crops/A009.jpg` | 416, 102 | 2× | ☐ |
| A010 | `reports/audit/crops/A010.jpg` | 415, 116 | 2× | ☐ |
| A011 | `reports/audit/crops/A011.jpg` | 412, 114 | 2× | ☐ |
| A012 | `reports/audit/crops/A012.jpg` | 410, 114 | 2× | ☐ |
| A013 | `reports/audit/crops/A013.jpg` | 411, 114 | 2× | ☐ |
| A014 | `reports/audit/crops/A014.jpg` | 411, 114 | 2× | ☐ |
| A015 | `reports/audit/crops/A015.jpg` | 413, 117 | 2× | ☐ |
| A016 | `reports/audit/crops/A016.jpg` | 495, 79 | 2× | ☐ |
| A017 | `reports/audit/crops/A017.jpg` | 852, 229 | 2× | ☐ |
| A018 | `reports/audit/crops/A018.jpg` | 820, 224 | 2× | ☐ |
| A019 | `reports/audit/crops/A019.jpg` | 846, 223 | 2× | ☐ |
| A020 | `reports/audit/crops/A020.jpg` | 872, 235 | 2× | ☐ |
| A021 | `reports/audit/crops/A021.jpg` | 866, 222 | 2× | ☐ |
| A022 | `reports/audit/crops/A022.jpg` | 414, 138 | 2× | ☐ |
| A023 | `reports/audit/crops/A023.jpg` | 414, 130 | 2× | ☐ |
| A024 | `reports/audit/crops/A024.jpg` | 415, 122 | 2× | ☐ |
| A025 | `reports/audit/crops/A025.jpg` | 421, 133 | 2× | ☐ |
| A026 | `reports/audit/crops/A026.jpg` | 419, 102 | 2× | ☐ |
| A027 | `reports/audit/crops/A027.jpg` | 417, 103 | 2× | ☐ |
| A028 | `reports/audit/crops/A028.jpg` | 418, 101 | 2× | ☐ |
| A029 | `reports/audit/crops/A029.jpg` | 417, 112 | 2× | ☐ |
| A030 | `reports/audit/crops/A030.jpg` | 416, 68 | 2× | ☐ |
| A031 | `reports/audit/crops/A031.jpg` | 888, 681 | 2× | ☐ |
| A032 | `reports/audit/crops/A032.jpg` | 833, 247 | 2× | ☐ |
| A033 | `reports/audit/crops/A033.jpg` | 828, 242 | 2× | ☐ |
| A034 | `reports/audit/crops/A034.jpg` | 528, 404 | 2× | ☐ |
| A035 | `reports/audit/crops/A035.jpg` | 527, 405 | 2× | ☐ |
| A036 | `reports/audit/crops/A036.jpg` | 534, 434 | 2× | ☐ |
| A037 | `reports/audit/crops/A037.jpg` | 1169, 962 | 2× | ☐ |
| A038 | `reports/audit/crops/A038.jpg` | 1166, 653 | 2× | ☐ |
| A039 | `reports/audit/crops/A039.jpg` | 1166, 607 | 2× | ☐ |
| A040 | `reports/audit/crops/A040.jpg` | 1166, 584 | 2× | ☐ |
| A041 | `reports/audit/crops/A041.jpg` | 1166, 522 | 2× | ☐ |
| A042 | `reports/audit/crops/A042.jpg` | 1102, 456 | 2× | ☐ |
| A043 | `reports/audit/crops/A043.jpg` | 1121, 456 | 2× | ☐ |
| A044 | `reports/audit/crops/A044.jpg` | 1130, 458 | 2× | ☐ |
| A045 | `reports/audit/crops/A045.jpg` | 1125, 460 | 2× | ☐ |
| A046 | `reports/audit/crops/A046.jpg` | 1130, 656 | 2× | ☐ |
| A047 | `reports/audit/crops/A047.jpg` | 1145, 956 | 2× | ☐ |
| A048 | `reports/audit/crops/A048.jpg` | 843, 309 | 2× | ☐ |
| A049 | `reports/audit/crops/A049.jpg` | 851, 304 | 2× | ☐ |
| A050 | `reports/audit/crops/A050.jpg` | 855, 303 | 2× | ☐ |
| A051 | `reports/audit/crops/A051.jpg` | 1085, 930 | 2× | ☐ |
| A052 | `reports/audit/crops/A052.jpg` | 1159, 912 | 2× | ☐ |
| A053 | `reports/audit/crops/A053.jpg` | 1164, 972 | 2× | ☐ |
| A054 | `reports/audit/crops/A054.jpg` | 1150, 972 | 2× | ☐ |
| A055 | `reports/audit/crops/A055.jpg` | 1146, 971 | 2× | ☐ |
| A056 | `reports/audit/crops/A056.jpg` | 1144, 971 | 2× | ☐ |
| A057 | `reports/audit/crops/A057.jpg` | 1161, 592 | 2× | ☐ |
| A058 | `reports/audit/crops/A058.jpg` | 1147, 563 | 2× | ☐ |
| A059 | `reports/audit/crops/A059.jpg` | 1157, 532 | 2× | ☐ |
| A060 | `reports/audit/crops/A060.jpg` | 1144, 469 | 2× | ☐ |
| A061 | `reports/audit/crops/A061.jpg` | 1144, 448 | 2× | ☐ |
| A062 | `reports/audit/crops/A062.jpg` | 1131, 412 | 2× | ☐ |
| A063 | `reports/audit/crops/A063.jpg` | 1148, 557 | 2× | ☐ |
| A064 | `reports/audit/crops/A064.jpg` | 1152, 554 | 2× | ☐ |
| A065 | `reports/audit/crops/A065.jpg` | 1134, 474 | 2× | ☐ |
| A066 | `reports/audit/crops/A066.jpg` | 1139, 472 | 2× | ☐ |
| A067 | `reports/audit/crops/A067.jpg` | 1136, 472 | 2× | ☐ |
| A068 | `reports/audit/crops/A068.jpg` | 1137, 472 | 2× | ☐ |
| A069 | `reports/audit/crops/A069.jpg` | 1193, 312 | 2× | ☐ |
| A070 | `reports/audit/crops/A070.jpg` | 878, 313 | 2× | ☐ |
| A071 | `reports/audit/crops/A071.jpg` | 879, 310 | 2× | ☐ |
| A072 | `reports/audit/crops/A072.jpg` | 897, 390 | 2× | ☐ |
| A073 | `reports/audit/crops/A073.jpg` | 895, 364 | 2× | ☐ |
