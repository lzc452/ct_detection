# -*- coding: utf-8 -*-
"""
把"问题电芯图片汇总"里已标注的图片导入训练数据集
用法：标完图后，双击工程根目录的"导入新标注.bat"

功能：
  1. 扫描 问题电芯图片汇总261012 下三个分类文件夹
  2. 优先读取同名 .txt（YOLO格式）；没有txt但有 .json（X-AnyLabeling格式）
     的自动转换，不需要手动导出
  3. 按文件名自动分成 训练集 / 验证集（约 8:1，固定规则，可反复运行）
  4. 已导入过的自动跳过，不会重复
  5. 未标注的图片名单写到 问题电芯图片汇总261012\未标注清单.txt
"""

import json
import os
import shutil
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, '问题电芯图片汇总')
DST = os.path.join(ROOT, 'datasets', 'CT_DATASET')

# 文件夹名 -> (类别编号, 导入时用的英文名前缀)
CLASSES = [
    ('0-极片损伤', 0, 'damage'),
    ('1-暗斑', 1, 'dark'),
    ('2-亮斑', 2, 'light'),
]
IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp')


def read_txt_lines(txt_path):
    """读取YOLO txt，返回有效行列表（每项5个字段），失败返回None"""
    lines = []
    try:
        with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                p = line.strip().split()
                if len(p) != 5:
                    continue
                try:
                    int(float(p[0]))
                    [float(x) for x in p[1:]]
                except ValueError:
                    continue
                lines.append(p)
    except OSError:
        return None
    return lines if lines else None


def read_json_lines(json_path, fallback_cid):
    """读取X-AnyLabeling/labelme的json，转成YOLO行列表，失败返回None"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        W = float(data.get('imageWidth') or 0)
        H = float(data.get('imageHeight') or 0)
        if W <= 0 or H <= 0:
            return None
        lines = []
        for sh in data.get('shapes', []):
            if sh.get('shape_type') != 'rectangle':
                continue
            pts = sh.get('points') or []
            if len(pts) < 2:
                continue
            # X-AnyLabeling存4个角点且顺序不定，取所有点的最大/最小值
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            xc = (x1 + x2) / 2.0 / W
            yc = (y1 + y2) / 2.0 / H
            bw = abs(x2 - x1) / W
            bh = abs(y2 - y1) / H
            if bw <= 0 or bh <= 0:
                continue
            label = str(sh.get('label', '')).strip()
            try:
                cid = int(float(label))
            except ValueError:
                cid = fallback_cid  # 类别名不是数字时，按所在文件夹的类别算
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            bw = min(bw, 1.0)
            bh = min(bh, 1.0)
            lines.append([str(cid), '%.6f' % xc, '%.6f' % yc, '%.6f' % bw, '%.6f' % bh])
    except (OSError, ValueError, KeyError):
        return None
    return lines if lines else None


def main():
    print('=' * 50)
    print('  导入新标注到训练数据集')
    print('=' * 50)

    if not os.path.isdir(SRC):
        print('\n[错误] 找不到文件夹：')
        print(SRC)
        print('请确认工程里有这个文件夹。')
        pause_exit()

    total_import = 0
    total_skip = 0
    missing = []       # 没标注的图片
    warnings = []      # 标注内容异常

    for folder, cid, cname in CLASSES:
        fdir = os.path.join(SRC, folder)
        if not os.path.isdir(fdir):
            print('\n[跳过] 没找到文件夹：%s' % folder)
            continue

        imgs = sorted(f for f in os.listdir(fdir)
                      if f.lower().endswith(IMG_EXT))
        n_import = 0
        n_skip = 0

        for img_name in imgs:
            base = os.path.splitext(img_name)[0]
            txt_path = os.path.join(fdir, base + '.txt')
            json_path = os.path.join(fdir, base + '.json')

            # 优先txt，其次json（自动转换），都没有算未标注
            lines = None
            if os.path.exists(txt_path):
                lines = read_txt_lines(txt_path)
            elif os.path.exists(json_path):
                lines = read_json_lines(json_path, cid)
            if lines is None:
                if os.path.exists(txt_path) or os.path.exists(json_path):
                    warnings.append('%s ：标注文件是空的或格式不对'
                                    % os.path.join(folder, img_name))
                else:
                    missing.append(os.path.join(folder, img_name))
                continue

            ids = set(int(float(p[0])) for p in lines)

            # 同一图里标了多个不同类别时保留原编号并提示；
            # 单一类别时统一改写为该文件夹对应的类别编号（防止标注软件编号不一致）
            remap = len(ids) == 1
            if not remap:
                warnings.append('%s ：一张图标了多个类别(%s)，按原编号导入'
                                % (os.path.join(folder, img_name), sorted(ids)))

            # 固定规则分训练/验证：文件名散列取模，结果永远不变
            h = zlib.crc32(img_name.encode('utf-8'))
            subset = 'val' if h % 9 == 0 else 'train'

            new_name = '%s_%s%s' % (cname, base, os.path.splitext(img_name)[1])
            dst_img_dir = os.path.join(DST, subset, 'images')
            dst_lbl_dir = os.path.join(DST, subset, 'labels')
            os.makedirs(dst_img_dir, exist_ok=True)
            os.makedirs(dst_lbl_dir, exist_ok=True)

            dst_img = os.path.join(dst_img_dir, new_name)
            dst_lbl = os.path.join(dst_lbl_dir, os.path.splitext(new_name)[0] + '.txt')

            if os.path.exists(dst_img):
                n_skip += 1
                continue

            shutil.copy2(os.path.join(fdir, img_name), dst_img)
            out_lines = []
            for p in lines:
                if remap:
                    p = [str(cid)] + p[1:]
                out_lines.append(' '.join(p))
            with open(dst_lbl, 'w', encoding='utf-8') as f:
                f.write('\n'.join(out_lines) + '\n')
            n_import += 1

        total_import += n_import
        total_skip += n_skip
        print('\n[%s] 图片 %d 张，新导入 %d 张，已存在跳过 %d 张'
              % (folder, len(imgs), n_import, n_skip))

    print('\n' + '=' * 50)
    print('  汇总：新导入 %d 张，跳过 %d 张' % (total_import, total_skip))
    print('  未标注 %d 张' % len(missing))
    if warnings:
        print('  有 %d 条标注提醒（见下方）' % len(warnings))

    if warnings:
        print('\n----- 标注提醒 -----')
        for w in warnings[:30]:
            print('  ! ' + w)
        if len(warnings) > 30:
            print('  ! ... 其余 %d 条从略' % (len(warnings) - 30))

    # 未标注清单写到汇总文件夹里，方便对着继续标
    log_path = os.path.join(SRC, '未标注清单.txt')
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('还没标注的图片（标完一张就会自动从这里消失）：\n\n')
        for m in missing:
            f.write(m + '\n')
    print('\n未标注清单已写到：')
    print('  ' + log_path)

    if total_import == 0 and not missing:
        print('\n没有新东西需要导入。')
    elif total_import > 0:
        print('\n下一步：双击"重新增强数据集.bat"，然后"开始训练.bat"。')

    pause_exit()


def pause_exit():
    try:
        input('\n按回车键关闭窗口...')
    except EOFError:
        pass
    sys.exit(0)


if __name__ == '__main__':
    main()
