# -*- coding: utf-8 -*-
"""收割复核入库：收割结果_积攒_V4/极片损伤 → 问题电芯图片汇总（母本）
规则：
  有标注 → 按主类别入 0/1/2 文件夹（损伤>暗斑>亮斑），全类别框保留，补全缺失的 description
  空标注 → 负样本文件夹（仅图，9_xxxx）
  已删图 → 不管（用户已删）
  更新 重命名对照表.csv；源文件夹保留不动，人工抽查后可删
"""
import os, glob, json, shutil, csv

ROOT = r'D:\CT Detection'
SRC = os.path.join(ROOT, '收割结果_积攒_V4', '极片损伤')
MASTER = os.path.join(ROOT, '问题电芯图片汇总')
PREFIX_DIR = {'0': '0-极片损伤', '1': '1-暗斑', '2': '2-亮斑'}
DESC = {'0': 'damage', '1': 'dark', '2': 'light'}
CSV_MAP = os.path.join(MASTER, '重命名对照表.csv')

def next_id(folder, prefix):
    nums = []
    for p in glob.glob(os.path.join(MASTER, folder, prefix + '_*.jpg')):
        s = os.path.splitext(os.path.basename(p))[0].split('_')[1]
        if s.isdigit():
            nums.append(int(s))
    return max(nums) + 1 if nums else 1

def main():
    log = []
    n_pos, n_neg = 0, 0
    for jp in sorted(glob.glob(os.path.join(SRC, '*.json'))):
        stem = os.path.splitext(os.path.basename(jp))[0]
        img = os.path.join(SRC, stem + '.jpg')
        if not os.path.exists(img):
            continue
        doc = json.load(open(jp, encoding='utf-8'))
        shapes = doc.get('shapes', [])
        if not shapes:
            # 负样本
            nid = next_id('负样本', '9')
            new_stem = '9_%04d' % nid
            shutil.copy2(img, os.path.join(MASTER, '负样本', new_stem + '.jpg'))
            log.append(['负样本', stem + '.jpg', new_stem + '.jpg'])
            n_neg += 1
            continue
        # 补全 description
        for s in shapes:
            if not s.get('description'):
                s['description'] = DESC.get(s['label'], '')
        # 主类别：损伤优先，其次暗斑，再次亮斑
        labels = {s['label'] for s in shapes}
        primary = '0' if '0' in labels else ('1' if '1' in labels else '2')
        folder = PREFIX_DIR[primary]
        nid = next_id(folder, primary)
        new_stem = '%s_%04d' % (primary, nid)
        doc['imagePath'] = new_stem + '.jpg'
        shutil.copy2(img, os.path.join(MASTER, folder, new_stem + '.jpg'))
        with open(os.path.join(MASTER, folder, new_stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False)
        log.append([folder, stem + '.jpg', new_stem + '.jpg'])
        n_pos += 1

    with open(CSV_MAP, 'a', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerows(log)
    print('入库样本 %d 张（正样本 %d + 负样本 %d），对照表已追加 %d 行' % (len(log), n_pos, n_neg, len(log)))

if __name__ == '__main__':
    main()
