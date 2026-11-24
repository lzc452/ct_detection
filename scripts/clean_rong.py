# -*- coding: utf-8 -*-
"""清理 E:\荣：每组只保留"正面"子文件夹，其余（上面/侧面/侧面头部/散文件）删除"""
import os, sys, shutil

root = r'E:\荣'
KEEP = '正面'
start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
end = int(sys.argv[2]) if len(sys.argv) > 2 else 10**9

groups = sorted([g for g in os.listdir(root) if os.path.isdir(os.path.join(root, g))])
todo = groups[start:end]
print(f'本批处理第 {start}~{min(end, len(groups))} 组（共 {len(groups)} 组）', flush=True)

done_dirs = done_files = 0
for g in todo:
    gp = os.path.join(root, g)
    for sub in os.listdir(gp):
        sp = os.path.join(gp, sub)
        if os.path.isdir(sp):
            if sub != KEEP:
                shutil.rmtree(sp, ignore_errors=True)
                done_dirs += 1
        else:
            try:
                os.remove(sp); done_files += 1
            except OSError:
                pass
    print(f'  [OK] {g}', flush=True)
print(f'本批完成：删目录 {done_dirs} 个、散文件 {done_files} 个', flush=True)
