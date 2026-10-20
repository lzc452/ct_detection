#coding:utf-8
# ============================================
#  CT缺陷检测系统 — 全局配置
# ============================================

# 模型路径（Web端加载的检测模型，环境体检脚本用）
model_path = 'models/启用/CT_DATASET_AUG_0910_1419/best.pt'

# 训练用数据集配置（train.py 使用）
data_ymal_path = 'datasets/CT_DATASET_AUG/data.yaml'

# 训练输出目录（train.py 使用）
project_path = "./runs/detect/CT_pro"

# 缺陷类别：英文（YOLO内部使用，ID从0开始）
names = {0: 'damage', 1: 'dark', 2: 'light'}

# 缺陷类别：中文（界面显示用，与英文一一对应）
CH_names = ['极片损伤', '暗斑', '亮斑']
