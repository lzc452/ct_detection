#coding:utf-8
"""交互式训练选择器；所有校验、参数、日志和归档委托train_run.py。
用法：runtime/python.exe scripts/train.py；传命令行参数时与train_run.py完全一致。
默认1920。尚无预注册时只输出dry-run计划，不启动模型或GPU。
"""
import os
import sys
import glob
import json
import datetime
from pathlib import Path
os.environ.setdefault('PROCESSOR_ARCHITECTURE', 'AMD64')
PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))
import train_run


def find_datasets():
    """仅列数据契约已完成的派生版本，不扫描原始图片目录。"""
    found = []
    for path in sorted(glob.glob('datasets_v*/*/data.yaml')):
        marker = Path(path).with_name('data_contract.json')
        if not marker.is_file():
            continue
        contract = json.loads(marker.read_text(encoding='utf-8'))
        if contract.get('schema') == 'ct-dataset/1' and contract.get('state') == 'complete':
            found.append(dict(name=Path(path).parent.name, yaml=path))
    return found


def choose_dataset(datasets):
    for index, dataset in enumerate(datasets, 1):
        print('[%d] %s' % (index, dataset['name']))
    while True:
        value = input('数据集编号: ').strip()
        if value.isdigit() and 1 <= int(value) <= len(datasets):
            return datasets[int(value)-1]


def choose_model_size():
    models = sorted(glob.glob('models/pretrained/*.pt'))
    if not models:
        raise FileNotFoundError('没有本地预训练权重，不自动下载')
    print('模型容量是待验证因素，较大模型不保证本任务更准。')
    for index, model in enumerate(models, 1):
        print('[%d] %s' % (index, model))
    while True:
        value = input('模型编号: ').strip()
        if value.isdigit() and 1 <= int(value) <= len(models):
            return models[int(value)-1]


def ask_experiment_name(default_name):
    return input('新实验名称（回车%s）: ' % default_name).strip() or default_name


def ask_imgsz(default=1920):
    return int(input('训练画布（32对齐，回车%d）: ' % default).strip() or default)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        return train_run.main(argv)
    datasets = find_datasets()
    if not datasets:
        raise FileNotFoundError('没有完成严格物化的数据契约；请按冻结清单用materialize_dataset.py创建独立版本')
    data = choose_dataset(datasets)
    name = ask_experiment_name(data['name']+'_'+datetime.datetime.now().strftime('%m%d_%H%M'))
    model = choose_model_size()
    size = os.environ.get('CT_IMGSZ') or str(ask_imgsz())
    baseline = input('基线args.yaml路径（必填，用于单变量对照）: ').strip().strip('"')
    finding = input('待验预注册编号F-XXX（留空只输出dry-run计划）: ').strip()
    options = ['--data', data['yaml'], '--model', model, '--name', name, '--imgsz', size,
               '--baseline-args', baseline, '--epochs', os.environ.get('CT_EPOCHS', '80'),
               '--batch', os.environ.get('CT_BATCH', '0.75'), '--device', os.environ.get('CT_DEVICE', '0'),
               '--cache', os.environ.get('CT_CACHE', 'disk')]
    options += ['--finding', finding] if finding else ['--dry-run']
    return train_run.main(options)


if __name__ == '__main__':
    sys.exit(main())
