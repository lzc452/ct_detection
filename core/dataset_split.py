# -*- coding: utf-8 -*-
"""
数据集分组与划分核心库（#10 修复的基础设施）

设计要点
--------
1. 原始数据只读：本模块只扫描 `问题电芯图片汇总`，绝不移动/改写原始文件。
2. 分组键（group_key）三级回退，优先级从强到弱：
   a) 电芯号（provenance）：从 `重命名对照表.csv` + `收割结果_积攒_V4` 文件名重建，如 `100-10`
   b) 内容近重复簇（duplicate cluster）：同一张切片的不同副本 / 重编码副本
   c) 单图自成一组
   同一 group 内的图片在 train/val/test 中只会出现在一侧。
3. 所有划分结果落 `split_manifest.csv`（可复核、可冻结、可做 CI 白名单校验）。
4. 断言：组键在 split 间不相交；MD5 不相交；近重复对不相交。

用法（作为库）
-------------
    from core.dataset_split import scan_master, build_groups, assign_splits, write_manifest
"""
import os
import re
import csv
import sys
import json
import glob
import time
import hashlib
import random
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASTER = os.path.join(ROOT, "问题电芯图片汇总")
POS_DIRS = ["0-极片损伤", "1-暗斑", "2-亮斑"]
NEG_DIR = "负样本"
RENAME_CSV = os.path.join(MASTER, "重命名对照表.csv")
HARVEST_SRC = os.path.join(ROOT, "收割结果_积攒_V4", "极片损伤")
CLS_NAME = {0: "damage", 1: "dark", 2: "light"}

_DESC_SIZE = 64          # 第一级粗筛描述子边长
_MAE_SIZE = 128          # 第二级精筛（MAE）边长


# ============================ 中文路径 IO ============================
def imread_cn(path, flags=None):
    """OpenCV 原生 imread 遇中文路径静默失败，必须走 imdecode（项目硬性约定）"""
    import cv2
    if flags is None:
        flags = cv2.IMREAD_GRAYSCALE
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def resize_gray(img_gray, size):
    import cv2
    return cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA)


def desc_of(img_gray, size=_DESC_SIZE):
    """z-score 后的向量，用于余弦粗筛（对整体亮度/对比度不敏感）"""
    v = resize_gray(img_gray, size).astype(np.float32).reshape(-1)
    v -= v.mean()
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-8 else v


def mae_vec_of(img_gray, size=_MAE_SIZE):
    """MAE 精筛用向量（uint8，不做归一化）"""
    return resize_gray(img_gray, size).astype(np.uint8).reshape(-1)


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


# ============================ 母本扫描 ============================
def scan_master(with_desc=True, verbose=True):
    """扫描母本，返回 items 列表。

    每项：{name, path, json, cls_main, md5, desc, mae_vec, n_boxes, cls_boxes}
    cls_main: -1 表示负样本
    """
    items = []
    for d in POS_DIRS:
        for jf in sorted(glob.glob(os.path.join(MASTER, d, "*.json"))):
            stem = os.path.splitext(os.path.basename(jf))[0]
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".bmp"):
                p = os.path.join(MASTER, d, stem + ext)
                if os.path.exists(p):
                    img = p
                    break
            if img:
                items.append({"name": stem, "path": img, "json": jf, "cls_main": int(d[0])})
    for f in sorted(os.listdir(os.path.join(MASTER, NEG_DIR))):
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            stem = os.path.splitext(f)[0]
            items.append({"name": stem, "path": os.path.join(MASTER, NEG_DIR, f),
                          "json": None, "cls_main": -1})

    if verbose:
        print("母本图片 %d 张（正样本 %d + 负样本 %d）" % (
            len(items), sum(1 for i in items if i["cls_main"] >= 0),
            sum(1 for i in items if i["cls_main"] < 0)))

    t0 = time.time()
    ok = []
    for it in items:
        it["md5"] = md5_of(it["path"])
        W, H, boxes = json_to_yolo(it["json"]) if it["json"] else (0, 0, [])
        it["n_boxes"] = len(boxes)
        cb = [0, 0, 0]
        for cls, *_ in boxes:
            if 0 <= cls < 3:
                cb[cls] += 1
        it["cls_boxes"] = cb
        if with_desc:
            g = imread_cn(it["path"])
            if g is None:
                if verbose:
                    print("  [跳过] 读取失败:", it["path"])
                continue
            it["desc"] = desc_of(g)
            it["mae_vec"] = mae_vec_of(g)
            it["h"], it["w"] = g.shape[:2]
        ok.append(it)
    if verbose:
        print("描述子/标注完成 %d 张，耗时 %.1fs" % (len(ok), time.time() - t0))
    return ok


def json_to_yolo(jpath):
    """X-AnyLabeling JSON -> (W, H, [(cls, xc, yc, w, h), ...])"""
    with open(jpath, encoding="utf-8") as stream:
        d = json.load(stream)
    W, H = d["imageWidth"], d["imageHeight"]
    boxes = []
    for s in d.get("shapes", []):
        try:
            cls = int(s["label"])
        except (ValueError, KeyError):
            continue
        pts = s.get("points", [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0.0, min(xs)), min(float(W), max(xs))
        y1, y2 = max(0.0, min(ys)), min(float(H), max(ys))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        boxes.append((cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
    return W, H, boxes


# ============================ 电芯号重建 ============================
def cell_from_stem(stem):
    """'100-10_10073' -> '100-10'；'9_1165' -> None（无电芯信息）"""
    if "_" not in stem:
        return None
    head = stem.rsplit("_", 1)[0]
    if "-" in head and head.replace("-", "").isdigit():
        return head
    return None


def load_provenance(items, verbose=True):
    """返回 {name: cell_id}。来源：重命名对照表 + 收割结果目录原名（按 MD5 对齐）"""
    prov = {}
    name_set = {it["name"] for it in items}
    trio = re.compile(r"([^,]+),([^,]+\.(?:jpg|jpeg|png|bmp)),([^,]+\.(?:jpg|jpeg|png|bmp))")

    if os.path.exists(RENAME_CSV):
        with open(RENAME_CSV, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                for m in trio.finditer(line.strip().replace("\ufeff", "")):
                    new_stem = os.path.splitext(m.group(3))[0]
                    cell = cell_from_stem(os.path.splitext(m.group(2))[0])
                    if cell and new_stem in name_set:
                        prov[new_stem] = cell

    n_csv = len(prov)
    if os.path.isdir(HARVEST_SRC):
        by_md5 = {it["md5"]: it["name"] for it in items}
        for p in glob.glob(os.path.join(HARVEST_SRC, "*.jpg")):
            cell = cell_from_stem(os.path.splitext(os.path.basename(p))[0])
            if not cell:
                continue
            nm = by_md5.get(md5_of(p))
            if nm:
                prov.setdefault(nm, cell)
    if verbose:
        print("电芯号重建：对照表 %d 张 → 合计 %d/%d (%.1f%%)" % (
            n_csv, len(prov), len(items), 100.0 * len(prov) / max(1, len(items))))
    return prov


# ============================ 分组 ============================
class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def build_groups(items, cos_th=0.999, mae_th=1.0, verbose=True):
    """三级分组：电芯号 → 内容近重复簇 → 单图。

    近重复判定必须余弦与 MAE 双阈值同时满足：
    - 余弦单用会把"大片空白/均匀背景"的不同电芯误并成一组（实测存在 243 张的巨簇）
    - MAE 保证只在像素级几乎相同时才合并（评审口径：MAE<1~2 等价同一张切片）

    返回 (group_of: list[str], info: dict)
    """
    n = len(items)
    desc = np.stack([it["desc"] for it in items])
    vecs = np.stack([it["mae_vec"] for it in items])

    uf = _UF(n)
    step = 512
    n_cand = 0
    for i0 in range(0, n, step):
        sim = desc[i0:i0 + step] @ desc.T
        for r, c in zip(*np.where(sim >= cos_th)):
            i = i0 + int(r)
            j = int(c)
            if i >= j:
                continue
            if uf.find(i) == uf.find(j):
                continue
            n_cand += 1
            if items[i]["h"] != items[j]["h"] or items[i]["w"] != items[j]["w"]:
                continue                      # 尺寸不同不可能是同一张切片的重编码
            mae = float(np.abs(vecs[i].astype(np.int16) - vecs[j].astype(np.int16)).mean())
            if mae <= mae_th:
                uf.union(i, j)

    prov = load_provenance(items, verbose=verbose)
    # 先把同电芯号的图并到同一组
    by_cell = defaultdict(list)
    for i, it in enumerate(items):
        c = prov.get(it["name"])
        if c:
            by_cell[c].append(i)
    for c, idxs in by_cell.items():
        for k in idxs[1:]:
            uf.union(idxs[0], k)

    group_of = ["g%05d" % uf.find(i) for i in range(n)]
    sizes = defaultdict(int)
    for g in group_of:
        sizes[g] += 1
    multi = {g: s for g, s in sizes.items() if s > 1}
    info = {
        "n_groups": len(sizes),
        "n_multi_groups": len(multi),
        "n_imgs_in_multi": sum(multi.values()),
        "max_group": max(sizes.values()) if sizes else 0,
        "top10_group_sizes": sorted(sizes.values(), reverse=True)[:10],
        "candidate_pairs": n_cand,
        "provenance_groups": len(by_cell),
        "cos_th": cos_th,
        "mae_th": mae_th,
    }
    if verbose:
        print("分组完成：%d 组（多图组 %d，覆盖 %d 张，最大组 %d 张）" % (
            info["n_groups"], info["n_multi_groups"], info["n_imgs_in_multi"], info["max_group"]))
    return group_of, info


# ============================ 划分 ============================
def assign_splits(items, group_of, ratios=(0.70, 0.15, 0.15), seed=42, verbose=True):
    """按组划分，按类别分层：每个 split 的三类框数与负样本数尽量按 ratios 配比。

    贪心：组按"信息量"（框数+图片数）降序，依次放进当前缺口最大的 split。
    """
    names = ("train", "val", "test")
    groups = defaultdict(list)
    for i, g in enumerate(group_of):
        groups[g].append(i)

    # 每组的三类框数 + 负样本数
    gw = {}
    for g, idxs in groups.items():
        cb = [0, 0, 0]
        for i in idxs:
            for c in range(3):
                cb[c] += items[i]["cls_boxes"][c]
        neg = sum(1 for i in idxs if items[i]["cls_main"] < 0)
        gw[g] = {"n": len(idxs), "cb": cb, "neg": neg, "key": sum(cb) + len(idxs)}

    total = {names[k]: {"cb": [0, 0, 0], "neg": 0, "n": 0} for k in range(3)}
    target_cb = [sum(gw[g]["cb"][c] for g in gw) * r for c, r in
                 zip(range(3), [1, 1, 1])]  # 占位，下面重算
    tot_cb = [sum(gw[g]["cb"][c] for g in gw) for c in range(3)]
    tot_neg = sum(gw[g]["neg"] for g in gw)
    target_cb = [[tot_cb[c] * r for c in range(3)] for r in ratios]
    target_neg = [tot_neg * r for r in ratios]

    rng = random.Random(seed)
    order = sorted(gw.keys(), key=lambda g: (-gw[g]["key"], g))
    rng.shuffle(order)                    # 打散同尺寸组的顺序，避免系统性偏向 train
    order.sort(key=lambda g: -gw[g]["key"])

    split_of = {}
    for g in order:
        w = gw[g]
        best, best_score = None, None
        for k, nm in enumerate(names):
            # 缺口 = 目标 - 已分配（正数表示还缺）
            def gap(c):
                return (target_cb[k][c] - total[nm]["cb"][c]) / max(1.0, target_cb[k][c])
            score = sum(w["cb"][c] * gap(c) for c in range(3))
            score += w["neg"] * (target_neg[k] - total[nm]["neg"]) / max(1.0, target_neg[k])
            if best_score is None or score > best_score:
                best, best_score = nm, score
        split_of[g] = best
        total[best]["n"] += w["n"]
        total[best]["neg"] += w["neg"]
        for c in range(3):
            total[best]["cb"][c] += w["cb"][c]

    if verbose:
        print("\n===== 划分结果（按组，按类别配比） =====")
        for nm in names:
            t = total[nm]
            print("  %-5s 图 %4d 张 | 负样本 %4d | 框 损伤%4d 暗斑%4d 亮斑%4d" % (
                nm, t["n"], t["neg"], *t["cb"]))
    return split_of, total


def build_manifest(items, group_of, split_of, out_csv, verbose=True):
    """写出划分清单（可冻结、可做 CI 白名单校验）"""
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["name", "src_path", "label_json", "group", "split", "md5", "n_boxes",
                    "n_damage", "n_dark", "n_light", "is_neg", "cell_id", "size"])
        cells = load_provenance(items, verbose=False)
        for i, it in enumerate(items):
            g = group_of[i]
            w.writerow([it["name"], os.path.relpath(it["path"], ROOT).replace("\\", "/"),
                        os.path.relpath(it["json"], ROOT).replace("\\", "/") if it["json"] else "",
                        g, split_of[g], it["md5"], it["n_boxes"], *it["cls_boxes"],
                        1 if it["cls_main"] < 0 else 0, cells.get(it["name"], ""),
                        "%dx%d" % (it.get("w", 0), it.get("h", 0))])
    if verbose:
        print("清单已写出：%s" % out_csv)
    return out_csv


# ============================ 泄漏断言 ============================
def assert_no_leak(items, group_of, split_of, mae_th=1.0, cos_th=0.999, verbose=True):
    """三方校验：组键不相交 / MD5 不相交 / 近重复对不相交。返回问题列表（空=通过）"""
    problems = []
    by_split = defaultdict(list)
    for i, it in enumerate(items):
        by_split[split_of[group_of[i]]].append(i)

    # 1) 组键不相交
    seen_group_split = {}
    for nm, idxs in by_split.items():
        for i in idxs:
            g = group_of[i]
            if g in seen_group_split and seen_group_split[g] != nm:
                problems.append("组 %s 同时出现在 %s 与 %s" % (g, seen_group_split[g], nm))
            seen_group_split[g] = nm

    # 2) MD5 不相交
    md5_split = {}
    for nm, idxs in by_split.items():
        for i in idxs:
            h = items[i]["md5"]
            if h in md5_split and md5_split[h] != nm:
                problems.append("MD5 %s 跨 %s/%s（%s, %s）" % (
                    h[:8], md5_split[h], nm, items[i]["name"], items[i]["name"]))
            md5_split[h] = nm

    # 3) 近重复对不相交：对跨集高余弦对做 MAE 精筛，只有"像素级几乎相同"才算泄漏
    desc = np.stack([it["desc"] for it in items])
    vecs = np.stack([it["mae_vec"] for it in items])
    pair_stat = {}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ia, ib = by_split.get(a, []), by_split.get(b, [])
        if not ia or not ib:
            continue
        sim = desc[ia] @ desc[ib].T
        ri, ci = np.where(sim >= cos_th)
        n_dupe = 0
        max_mae = 0.0
        for r, c in zip(ri, ci):
            i, j = ia[int(r)], ib[int(c)]
            if items[i]["h"] != items[j]["h"] or items[i]["w"] != items[j]["w"]:
                continue
            mae = float(np.abs(vecs[i].astype(np.int16) - vecs[j].astype(np.int16)).mean())
            max_mae = max(max_mae, mae)
            if mae <= mae_th:
                n_dupe += 1
                if n_dupe <= 5:
                    problems.append("近重复跨集 %s×%s: %s <-> %s (cos=%.6f MAE=%.3f)" % (
                        a, b, items[i]["name"], items[j]["name"], sim[r, c], mae))
        pair_stat["%s×%s" % (a, b)] = {
            "n_high_cos": int(len(ri)), "n_pixel_dupe": n_dupe, "max_mae": round(max_mae, 3),
        }
        if n_dupe:
            problems.append("%s×%s 存在 %d 对像素级近重复（cos>=%.4f 且 MAE<=%.2f）" % (
                a, b, n_dupe, cos_th, mae_th))

    if verbose:
        for k, v in pair_stat.items():
            print("  跨集对照 %-11s 高余弦对 %-4d 其中像素级重复 %-3d 最大MAE %.3f" % (
                k, v["n_high_cos"], v["n_pixel_dupe"], v["max_mae"]))
        if problems:
            print("\n[泄漏断言] 未通过，共 %d 项：" % len(problems))
            for p in problems[:20]:
                print("  -", p)
        else:
            print("\n[泄漏断言] 通过：组键 / MD5 / 像素级近重复 三方均无跨集")
    return problems


def load_manifest(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
