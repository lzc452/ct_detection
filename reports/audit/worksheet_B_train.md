# 批 B · 损伤标注一致性审计作业单（train 分层 100 框）

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

## 四、这批什么时候做

**条件触发**：先做批 A。若批 A 判定为「通过」，说明标注测量本身可信，此批用于判断**训练标签**是否也需要清洗（决定是否花 1 天重标 338 框）；若批 A 判定「不通过」，先修标注规范，本批顺延重做。

## 五、逐框清单（100 框）

| 编号 | 裁剪图 | 裁剪偏移(原图 x0, y0) | 放大 | 画好了 |
|---|---|---|---|---|
| B001 | `reports/audit/crops/B001.jpg` | 414, 123 | 2× | ☐ |
| B002 | `reports/audit/crops/B002.jpg` | 410, 135 | 2× | ☐ |
| B003 | `reports/audit/crops/B003.jpg` | 1125, 953 | 2× | ☐ |
| B004 | `reports/audit/crops/B004.jpg` | 1176, 519 | 2× | ☐ |
| B005 | `reports/audit/crops/B005.jpg` | 1010, 249 | 2× | ☐ |
| B006 | `reports/audit/crops/B006.jpg` | 1176, 578 | 2× | ☐ |
| B007 | `reports/audit/crops/B007.jpg` | 1138, 960 | 2× | ☐ |
| B008 | `reports/audit/crops/B008.jpg` | 412, 150 | 2× | ☐ |
| B009 | `reports/audit/crops/B009.jpg` | 412, 147 | 2× | ☐ |
| B010 | `reports/audit/crops/B010.jpg` | 415, 110 | 2× | ☐ |
| B011 | `reports/audit/crops/B011.jpg` | 1031, 170 | 2× | ☐ |
| B012 | `reports/audit/crops/B012.jpg` | 831, 914 | 2× | ☐ |
| B013 | `reports/audit/crops/B013.jpg` | 1063, 243 | 2× | ☐ |
| B014 | `reports/audit/crops/B014.jpg` | 411, 106 | 2× | ☐ |
| B015 | `reports/audit/crops/B015.jpg` | 408, 109 | 2× | ☐ |
| B016 | `reports/audit/crops/B016.jpg` | 416, 110 | 2× | ☐ |
| B017 | `reports/audit/crops/B017.jpg` | 513, 435 | 2× | ☐ |
| B018 | `reports/audit/crops/B018.jpg` | 1094, 250 | 2× | ☐ |
| B019 | `reports/audit/crops/B019.jpg` | 386, 406 | 2× | ☐ |
| B020 | `reports/audit/crops/B020.jpg` | 853, 247 | 2× | ☐ |
| B021 | `reports/audit/crops/B021.jpg` | 839, 244 | 2× | ☐ |
| B022 | `reports/audit/crops/B022.jpg` | 414, 113 | 2× | ☐ |
| B023 | `reports/audit/crops/B023.jpg` | 828, 240 | 2× | ☐ |
| B024 | `reports/audit/crops/B024.jpg` | 504, 81 | 2× | ☐ |
| B025 | `reports/audit/crops/B025.jpg` | 413, 162 | 2× | ☐ |
| B026 | `reports/audit/crops/B026.jpg` | 415, 164 | 2× | ☐ |
| B027 | `reports/audit/crops/B027.jpg` | 510, 81 | 2× | ☐ |
| B028 | `reports/audit/crops/B028.jpg` | 837, 243 | 2× | ☐ |
| B029 | `reports/audit/crops/B029.jpg` | 893, 370 | 2× | ☐ |
| B030 | `reports/audit/crops/B030.jpg` | 1154, 728 | 2× | ☐ |
| B031 | `reports/audit/crops/B031.jpg` | 1154, 931 | 2× | ☐ |
| B032 | `reports/audit/crops/B032.jpg` | 876, 751 | 2× | ☐ |
| B033 | `reports/audit/crops/B033.jpg` | 1152, 844 | 2× | ☐ |
| B034 | `reports/audit/crops/B034.jpg` | 849, 894 | 2× | ☐ |
| B035 | `reports/audit/crops/B035.jpg` | 882, 343 | 2× | ☐ |
| B036 | `reports/audit/crops/B036.jpg` | 922, 889 | 2× | ☐ |
| B037 | `reports/audit/crops/B037.jpg` | 543, 394 | 2× | ☐ |
| B038 | `reports/audit/crops/B038.jpg` | 840, 283 | 2× | ☐ |
| B039 | `reports/audit/crops/B039.jpg` | 400, 110 | 2× | ☐ |
| B040 | `reports/audit/crops/B040.jpg` | 398, 84 | 2× | ☐ |
| B041 | `reports/audit/crops/B041.jpg` | 400, 122 | 2× | ☐ |
| B042 | `reports/audit/crops/B042.jpg` | 1141, 949 | 2× | ☐ |
| B043 | `reports/audit/crops/B043.jpg` | 840, 261 | 2× | ☐ |
| B044 | `reports/audit/crops/B044.jpg` | 1137, 963 | 2× | ☐ |
| B045 | `reports/audit/crops/B045.jpg` | 868, 318 | 2× | ☐ |
| B046 | `reports/audit/crops/B046.jpg` | 543, 346 | 2× | ☐ |
| B047 | `reports/audit/crops/B047.jpg` | 556, 431 | 2× | ☐ |
| B048 | `reports/audit/crops/B048.jpg` | 1053, 941 | 2× | ☐ |
| B049 | `reports/audit/crops/B049.jpg` | 1119, 941 | 2× | ☐ |
| B050 | `reports/audit/crops/B050.jpg` | 1114, 940 | 2× | ☐ |
| B051 | `reports/audit/crops/B051.jpg` | 1132, 708 | 2× | ☐ |
| B052 | `reports/audit/crops/B052.jpg` | 1129, 954 | 2× | ☐ |
| B053 | `reports/audit/crops/B053.jpg` | 1134, 700 | 2× | ☐ |
| B054 | `reports/audit/crops/B054.jpg` | 1146, 952 | 2× | ☐ |
| B055 | `reports/audit/crops/B055.jpg` | 1148, 793 | 2× | ☐ |
| B056 | `reports/audit/crops/B056.jpg` | 1181, 332 | 2× | ☐ |
| B057 | `reports/audit/crops/B057.jpg` | 1163, 955 | 2× | ☐ |
| B058 | `reports/audit/crops/B058.jpg` | 1199, 722 | 2× | ☐ |
| B059 | `reports/audit/crops/B059.jpg` | 1195, 706 | 2× | ☐ |
| B060 | `reports/audit/crops/B060.jpg` | 920, 707 | 2× | ☐ |
| B061 | `reports/audit/crops/B061.jpg` | 925, 716 | 2× | ☐ |
| B062 | `reports/audit/crops/B062.jpg` | 1172, 701 | 2× | ☐ |
| B063 | `reports/audit/crops/B063.jpg` | 1196, 812 | 2× | ☐ |
| B064 | `reports/audit/crops/B064.jpg` | 878, 931 | 2× | ☐ |
| B065 | `reports/audit/crops/B065.jpg` | 1184, 693 | 2× | ☐ |
| B066 | `reports/audit/crops/B066.jpg` | 1163, 903 | 2× | ☐ |
| B067 | `reports/audit/crops/B067.jpg` | 1057, 242 | 2× | ☐ |
| B068 | `reports/audit/crops/B068.jpg` | 1161, 838 | 2× | ☐ |
| B069 | `reports/audit/crops/B069.jpg` | 1154, 972 | 2× | ☐ |
| B070 | `reports/audit/crops/B070.jpg` | 1148, 974 | 2× | ☐ |
| B071 | `reports/audit/crops/B071.jpg` | 1092, 229 | 2× | ☐ |
| B072 | `reports/audit/crops/B072.jpg` | 1134, 961 | 2× | ☐ |
| B073 | `reports/audit/crops/B073.jpg` | 1129, 748 | 2× | ☐ |
| B074 | `reports/audit/crops/B074.jpg` | 1124, 790 | 2× | ☐ |
| B075 | `reports/audit/crops/B075.jpg` | 1141, 970 | 2× | ☐ |
| B076 | `reports/audit/crops/B076.jpg` | 1148, 384 | 2× | ☐ |
| B077 | `reports/audit/crops/B077.jpg` | 1076, 377 | 2× | ☐ |
| B078 | `reports/audit/crops/B078.jpg` | 1153, 599 | 2× | ☐ |
| B079 | `reports/audit/crops/B079.jpg` | 1134, 952 | 2× | ☐ |
| B080 | `reports/audit/crops/B080.jpg` | 1147, 452 | 2× | ☐ |
| B081 | `reports/audit/crops/B081.jpg` | 795, 905 | 2× | ☐ |
| B082 | `reports/audit/crops/B082.jpg` | 1101, 334 | 2× | ☐ |
| B083 | `reports/audit/crops/B083.jpg` | 1115, 872 | 2× | ☐ |
| B084 | `reports/audit/crops/B084.jpg` | 1140, 315 | 2× | ☐ |
| B085 | `reports/audit/crops/B085.jpg` | 1144, 839 | 2× | ☐ |
| B086 | `reports/audit/crops/B086.jpg` | 1140, 840 | 2× | ☐ |
| B087 | `reports/audit/crops/B087.jpg` | 1143, 438 | 2× | ☐ |
| B088 | `reports/audit/crops/B088.jpg` | 531, 293 | 2× | ☐ |
| B089 | `reports/audit/crops/B089.jpg` | 893, 326 | 2× | ☐ |
| B090 | `reports/audit/crops/B090.jpg` | 891, 356 | 2× | ☐ |
| B091 | `reports/audit/crops/B091.jpg` | 888, 306 | 2× | ☐ |
| B092 | `reports/audit/crops/B092.jpg` | 897, 352 | 2× | ☐ |
| B093 | `reports/audit/crops/B093.jpg` | 900, 359 | 2× | ☐ |
| B094 | `reports/audit/crops/B094.jpg` | 1130, 781 | 2× | ☐ |
| B095 | `reports/audit/crops/B095.jpg` | 878, 307 | 2× | ☐ |
| B096 | `reports/audit/crops/B096.jpg` | 883, 328 | 2× | ☐ |
| B097 | `reports/audit/crops/B097.jpg` | 871, 326 | 2× | ☐ |
| B098 | `reports/audit/crops/B098.jpg` | 883, 341 | 2× | ☐ |
| B099 | `reports/audit/crops/B099.jpg` | 404, 301 | 2× | ☐ |
| B100 | `reports/audit/crops/B100.jpg` | 403, 235 | 2× | ☐ |
