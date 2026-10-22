# -*- coding: utf-8 -*-
"""按冻结清单创建独立派生版本；坏输入即失败，无复制/删除母本回退。"""
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import yaml
import dataset_split as ds
import roi as R
from eval_cache import file_sha256, validate_annotation
from dataset_contract import SCHEMA, NAMES, check_yolo, inspect_dataset
from path_safety import ROOT, dataset_output, safe_name


def selected_rows(manifest, splits):
    rows = ds.load_manifest(manifest)
    wants = [s.strip() for s in splits.split(',') if s.strip()]
    if not wants or len(set(wants)) != len(wants) or not set(wants) <= {'train', 'val', 'test'}:
        raise ValueError('split必须是train,val,test的不重复子集')
    names, groups, hashes = set(), {}, {}
    for row in rows:
        safe_name(row['name'])
        if row['name'].casefold() in names:
            raise ValueError('清单图片名重复（Windows不区分大小写）')
        names.add(row['name'].casefold())
        split, group, md5 = row['split'], row['group'], row['md5']
        if split not in {'train', 'val', 'test'} or not group or len(md5) != 32:
            raise ValueError('清单缺少split/group/MD5')
        for key, mapping in [(group, groups), (md5, hashes)]:
            if mapping.setdefault(key, split) != split:
                raise ValueError('同组或相同图片跨split泄漏: ' + row['name'])
    chosen = [r for r in rows if r['split'] in wants]
    if any(not any(r['split'] == s for r in chosen) for s in wants):
        raise ValueError('请求的split为空')
    return chosen, wants


def build_dataset(manifest, root, name, splits='train,val', roi=False, roi_k=0.,
                  quality=95, verify_only=False, force=False):
    """verify_only只读；force保留兼容参数，但绝不删除/覆盖已有目录。"""
    out = dataset_output(root, name)
    if not verify_only and out.exists():
        raise FileExistsError('输出已存在；--force不再删除目录，请使用新版本名: ' + str(out))
    if not math.isfinite(roi_k) or roi_k < 0 or not 1 <= quality <= 100:
        raise ValueError('ROI余量或JPEG质量非法')
    if not roi and roi_k != 0:
        raise ValueError('全图数据不接受ROI余量')
    rows, wants = selected_rows(manifest, splits)
    R.set_margin_k(roi_k)  # 裁图和GT变换继续共用模块单源，物化器为单线程。
    if not verify_only:
        out.mkdir(parents=True, exist_ok=False)
    outputs, sources, stats = {}, [], {s: 0 for s in wants}
    for row in rows:
        validate_annotation(row)
        n_boxes = int(row['n_boxes'])
        if n_boxes < 0 or (not row.get('label_json') and (n_boxes or row.get('is_neg') != '1')):
            raise ValueError('缺标不能自动当负样本，须有显式is_neg=1且n_boxes=0: ' + row['name'])
        src = ROOT / row['src_path']
        if ds.md5_of(src) != row['md5']:
            raise ValueError('原图偏离冻结MD5: ' + row['name'])
        buf = np.fromfile(src, dtype=np.uint8)
        image = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
        if image is None or '%dx%d' % (image.shape[1], image.shape[0]) != row['size']:
            raise ValueError('原图无法解码或尺寸不符: ' + row['name'])
        if roi:
            image, box = R.crop_roi(image)
            if not R.roi_looks_valid(image):
                raise ValueError('ROI有效性失败: ' + row['name'])
            lines, kept, dropped = R.labelme_boxes_in_roi(str(ROOT / row['label_json']), box) \
                if row.get('label_json') else ([], 0, 0)
            if dropped or kept != n_boxes:
                raise ValueError('ROI丢GT或框数变化: ' + row['name'])
        else:
            boxes = ds.json_to_yolo(str(ROOT / row['label_json']))[2] if row.get('label_json') else []
            lines = ['%d %.6f %.6f %.6f %.6f' % tuple(b) for b in boxes]
        text = '\n'.join(lines)
        if len(check_yolo(text, row['name'])) != n_boxes:
            raise ValueError('转换后框数不符: ' + row['name'])
        source = dict(name=row['name'], split=row['split'], group=row['group'],
                      image_md5=row['md5'], label_sha256=file_sha256(ROOT / row['label_json'])
                      if row.get('label_json') else None)
        sources.append(source)
        if not verify_only:
            image_rel = f"{row['split']}/images/{row['name']}.jpg"
            label_rel = f"{row['split']}/labels/{row['name']}.txt"
            image_path, label_path = out / image_rel, out / label_rel
            image_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.parent.mkdir(parents=True, exist_ok=True)
            if roi:
                ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if not ok:
                    raise ValueError('ROI编码失败: ' + row['name'])
                with image_path.open('xb') as stream:
                    stream.write(encoded.tobytes())
            else:
                # 跨卷/不支持硬链接直接失败，绝不退回复制生产母本。
                os.link(src, image_path)
                if ds.md5_of(image_path) != row['md5']:
                    raise ValueError('物化期间原图发生变化: ' + row['name'])
            with label_path.open('x', encoding='utf-8') as stream:
                stream.write(text)
            for rel in (image_rel, label_rel):
                outputs[rel] = file_sha256(out / rel)
        stats[row['split']] += 1
    contract = dict(schema=SCHEMA, state='complete', roi=bool(roi), roi_k=float(roi_k),
                    splits=wants, jpeg_quality=quality if roi else None,
                    manifest_sha256=file_sha256(manifest), sources=sources, outputs=outputs,
                    converter_sha256=file_sha256(Path(__file__)),
                    label_converter_sha256=file_sha256(ds.__file__), cv2_version=cv2.__version__,
                    roi_code_sha256=file_sha256(R.__file__) if roi else None,
                    roi_box=[R.ROI_X0, R.ROI_Y0, R.ROI_X1, R.ROI_Y1] if roi else None, classes=NAMES)
    if not verify_only:
        config = dict(path=str(out), nc=3, names=NAMES, **{s: f'{s}/images' for s in wants})
        with (out / 'data.yaml').open('x', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, allow_unicode=True, sort_keys=False)
        # 完成标记最后写；半成品不会通过训练preflight。失败目录留存排查，不自动删除。
        with (out / 'data_contract.json').open('x', encoding='utf-8') as stream:
            json.dump(contract, stream, ensure_ascii=False, indent=2)
        inspect_dataset(out / 'data.yaml', require_training=False)
    return dict(output=str(out), stats=stats, verify_only=verify_only, roi=roi, roi_k=roi_k)
