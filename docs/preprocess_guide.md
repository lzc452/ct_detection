# -*- coding: utf-8 -*-
"""
CT缺陷检测 — 预处理与增强使用说明
================================

针对你的情况（样本少、CT图缺陷不明显），建议按以下步骤执行：


一、先运行预处理预览，选最佳方案
--------------------------------

    python preview_preprocess.py

运行后会在当前目录生成：
    - preprocess_preview.jpg   （6种预处理对比图）
    - preview_*.jpg             （每种模式的单独图）

你肉眼对比哪个模式下缺陷最清晰，后续就用哪个。
推荐先用 **combo**（组合预处理）或 **denoise_clahe**（去噪+CLAHE）。


二、离线增强数据集（从140张 → 500+张）
----------------------------------------

    python augment_dataset.py --factor 4

这个命令会：
    1. 读取 datasets/CT_DATASET/train 和 val
    2. 对每张图先用 combo 预处理
    3. 生成4种变体（亮度/噪声/翻转/旋转）
    4. 同步变换YOLO标注框坐标
    5. 输出到 datasets/CT_DATASET_AUG/train 和 val

执行完后：
    - train 约 140 × 5 = 700 张
    - val 约 42 × 3 = 126 张


三、修改 data.yaml 指向增强数据集
--------------------------------

编辑 datasets/CT_DATASET_AUG/data.yaml：

    train: ./train/images
    val: ./val/images
    nc: 3
    names:
      0: crease
      1: dark
      2: light

（或直接复制原 data.yaml 过来）


四、开始训练
------------

train.py 已经更新，自动使用 augment.yaml 的增强配置：

    python train.py

关键改进参数：
    - epochs: 300（更多轮数）
    - patience: 30（早停，防止过拟合）
    - optimizer: AdamW（更适合小样本）
    - hyp: augment.yaml（CT优化的增强策略）
    - cos_lr: True（余弦退火学习率）
    - close_mosaic: 10（最后10轮关闭Mosaic稳定训练）


五、检测时预处理（可选）
------------------------

如果你想在 MainProgram.py 检测时也加上预处理，修改检测逻辑：

在 MainProgram.py 的 detect_one_img 或 detact_batch_imgs 中：

    from ct_preprocess import preprocess_ct_image
    
    # 检测前预处理
    img = cv2.imread(img_path)
    img = preprocess_ct_image(img, mode='combo')  # 加这一行
    self.results = self.model(img)[0]

这样检测时也会增强缺陷对比度。


六、各预处理模式说明
--------------------

| 模式 | 效果 | 适用场景 |
|------|------|---------|
| none | 原图 | 对比baseline |
| clahe | 局部对比度增强 | 缺陷区域亮度不均 |
| stretch | 全局对比度拉伸 | 整体动态范围窄 |
| denoise_clahe | 去噪+CLAHE | 噪声多且对比度低 |
| combo | 去噪+CLAHE+锐化 | **推荐默认** |
| combo_v2 | 强增强组合 | 缺陷极不明显时 |


七、注意事项
----------

1. 增强后的标注文件一定要检查！运行 augment_dataset.py 后，用 detect_tools.py 的 draw_yolo_data() 抽查几张。

2. 如果你的CT图是 DICOM (.dcm) 格式，需要先用 pydicom 读取并转为jpg/png，再跑这些脚本。

3. 样本扩充到500+后，如果验证mAP仍上不去，再考虑换 yolov8s 或加注意力机制。

4. 训练时建议打开 TensorBoard 看曲线：
   
       tensorboard --logdir runs/detect/CT_pro
