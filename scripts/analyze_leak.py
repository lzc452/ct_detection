# -*- coding: utf-8 -*-
"""
泄漏与分组覆盖度测量（只读，不改动任何原始数据）
- 目标：为 #10（按电芯/批次分组）确定可用的分组键与兜底策略
- 测量项：
  1) 母本 MD5 重复分布（字节级同图）
  2) V4 val -> train 最近邻相似度（复现评审 82.7% 结论）
  3) 母本内容近重复聚类的簇规模分布（决定能否用"内容簇"代替"电芯组"）
  4) 电芯号（重命名对照表 / 收割结果文件名）覆盖度
用法：
  runtime\\python.exe scripts\\analyze_leak.py
  runtime\\python.exe scripts\\analyze_leak.py --limit 800   # 快速抽样
输出：reports/leak_analysis_<日期>.json  + 控制台表格
"""
import os, sys, csv, json, glob, hashlib, time, argparse
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

MASTER = "问题电芯图片汇总"
POS_DIRS = ["0-极片损伤", "1-暗斑", "2-亮斑"]
NEG_DIR = "负样本"
V4 = os.path.join("datasets", "CT_DATASET_FINAL_V4")
RENAME_CSV = os.path.join(MASTER, "重命名对照表.csv")
HARVEST_SRC = os.path.join("收割结果_积攒_V4", "极片损伤")


# ---------------- 中文路径 IO（项目硬性约定） ----------------
def imread_cn(path, flags=None):
    import cv2
    if flags is None:
        flags = cv2.IMREAD_GRAYSCALE
    buf = np.fromfile(path, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


# ---------------- 描述子 ----------------
def desc_of(img_gray, size=64):
    """缩放到 size×size 后 z-score，返回 float32 向量（用余弦比较）"""
    import cv2
    small = cv2.resize(img_gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    small = small.reshape(-1)
    small -= small.mean()
    n = np.linalg.norm(small)
    return small / n if n > 1e-8 else small


def md5_of(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ---------------- 母本扫描 ----------------
def scan_master():
    """返回 items: [{name, path, json, cls_main}]，json 为 None 表示负样本"""
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
    return items


# ---------------- 电芯号重建 ----------------
def load_provenance(items):
    """从重命名对照表 + 收割结果文件名重建电芯号；返回 {name: cell_id}"""
    prov = {}
    name_set = {it["name"] for it in items}

    # 1) 重命名对照表（格式：类别目录,原文件名,新文件名），容错处理粘连行
    if os.path.exists(RENAME_CSV):
        with open(RENAME_CSV, encoding="utf-8-sig", errors="ignore") as f:
            for line in f:
                line = line.strip().replace("\ufeff", "")
                if not line:
                    continue
                # 用正则抓 "任意,原名,新名" 三元组，允许一行里粘连多组
                import re
                for m in re.finditer(r"([^,]+),([^,]+\.(?:jpg|jpeg|png|bmp)),([^,]+\.(?:jpg|jpeg|png|bmp))", line):
                    new_stem = os.path.splitext(m.group(3))[0]
                    old_stem = os.path.splitext(m.group(2))[0]
                    cell = cell_from_stem(old_stem)
                    if cell and new_stem in name_set:
                        prov[new_stem] = cell

    # 2) 收割结果目录里仍保留原名（电芯_切片.jpg），用字节哈希对齐母本
    if os.path.isdir(HARVEST_SRC):
        by_md5 = {}
        for it in items:
            try:
                by_md5.setdefault(md5_of(it["path"]), it["name"])
            except OSError:
                pass
        for p in glob.glob(os.path.join(HARVEST_SRC, "*.jpg")):
            cell = cell_from_stem(os.path.splitext(os.path.basename(p))[0])
            if not cell:
                continue
            h = md5_of(p)
            nm = by_md5.get(h)
            if nm:
                prov.setdefault(nm, cell)
    return prov


def cell_from_stem(stem):
    """'100-10_10073' -> '100-10'；'9_1165' -> None（无电芯信息）"""
    if "_" not in stem:
        return None
    head = stem.rsplit("_", 1)[0]
    if "-" in head and head.replace("-", "").isdigit():
        return head
    return None


# ---------------- 聚类 ----------------
class UF:
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


def cluster(desc, th):
    """按余弦阈值做连通分量聚类（分块避免爆内存）"""
    n = len(desc)
    uf = UF(n)
    step = 512
    for i0 in range(0, n, step):
        blk = desc[i0:i0 + step] @ desc.T          # (step, n)
        idx = np.where(blk >= th)
        for r, c in zip(*idx):
            if i0 + r < c:
                uf.union(i0 + r, c)
    sizes = {}
    for i in range(n):
        sizes.setdefault(uf.find(i), 0)
        sizes[uf.find(i)] += 1
    return uf, sizes


# ---------------- V4 val -> train 最近邻 ----------------
def nearest_stats(ds_dir, train_desc, train_meta, val_desc, val_meta, fullres_sample=40):
    """返回 val 每张图对 train 的最大余弦、以及全分辨率 MAE 分布"""
    sim = val_desc @ train_desc.T                 # (nv, nt)
    best = sim.argmax(axis=1)
    maxsim = sim[np.arange(len(val_desc)), best]
    return best, maxsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="抽样图片数（0=全量）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    t0 = time.time()
    items = scan_master()
    if args.limit:
        items = items[:args.limit]
    print("母本图片 %d 张" % len(items))

    # 1) MD5
    md5s = {}
    for it in items:
        md5s.setdefault(md5_of(it["path"]), []).append(it["name"])
    dup_groups = {k: v for k, v in md5s.items() if len(v) > 1}
    print("MD5 重复组 %d 个，涉及 %d 张" % (len(dup_groups), sum(len(v) for v in dup_groups.values())))

    # 2) 描述子
    descs, ok = [], []
    for it in items:
        g = imread_cn(it["path"])
        if g is None:
            print("读取失败:", it["path"])
            continue
        descs.append(desc_of(g, 64))
        ok.append(it)
    items = ok
    desc = np.stack(descs)
    print("描述子完成 %d 张，耗时 %.1fs" % (len(items), time.time() - t0))

    # 3) 聚类簇规模分布
    cluster_stat = {}
    for th in (0.9999, 0.9995, 0.999, 0.995, 0.99):
        uf, sizes = cluster(desc, th)
        labels = np.array([uf.find(i) for i in range(len(items))])
        multi = sum(1 for v in sizes.values() if v > 1)
        cluster_stat[str(th)] = {
            "n_clusters": len(sizes),
            "n_multi_clusters": multi,
            "n_imgs_in_multi": int(sum(v for v in sizes.values() if v > 1)),
            "max_cluster": int(max(sizes.values())),
            "top5": sorted(sizes.values(), reverse=True)[:5],
        }
        print("余弦>=%.4f -> 簇 %d 个（多图簇 %d，最大 %d，前5 %s）" % (
            th, len(sizes), multi, max(sizes.values()), sorted(sizes.values(), reverse=True)[:5]))

    # 4) V4 val->train 复现
    v4_stat = {}
    for split_a, split_b in (("val", "train"),):
        pa = os.path.join(V4, split_a, "images")
        pb = os.path.join(V4, split_b, "images")
        fa = sorted(glob.glob(os.path.join(pa, "*.jpg")))
        fb = sorted(glob.glob(os.path.join(pb, "*.jpg")))
        if not fa or not fb:
            print("V4 目录为空，跳过"); break
        da, db = [], []
        for f in fa:
            g = imread_cn(f)
            da.append(desc_of(g, 64) if g is not None else np.zeros(64 * 64, np.float32))
        for f in fb:
            g = imread_cn(f)
            db.append(desc_of(g, 64) if g is not None else np.zeros(64 * 64, np.float32))
        da, db = np.stack(da), np.stack(db)
        sim = da @ db.T
        maxsim = sim.max(axis=1)
        v4_stat["val_to_train_maxcos"] = {
            "p50": float(np.percentile(maxsim, 50)),
            "p90": float(np.percentile(maxsim, 90)),
            "p99": float(np.percentile(maxsim, 99)),
            "ge_0.9999": int((maxsim >= 0.9999).sum()),
            "ge_0.999": int((maxsim >= 0.999).sum()),
            "ge_0.995": int((maxsim >= 0.995).sum()),
            "n_val": len(fa),
        }
        print("V4 val->train 最大余弦：p50=%.6f p90=%.6f | >=0.9999: %d/%d | >=0.999: %d" % (
            v4_stat["val_to_train_maxcos"]["p50"], v4_stat["val_to_train_maxcos"]["p90"],
            v4_stat["val_to_train_maxcos"]["ge_0.9999"], len(fa),
            v4_stat["val_to_train_maxcos"]["ge_0.999"]))

    # 5) 覆盖度
    prov = load_provenance(items)
    cover = len(prov)
    print("电芯号可重建：%d / %d (%.1f%%)" % (cover, len(items), 100.0 * cover / max(1, len(items))))

    out = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_master_images": len(items),
        "md5_dup_groups": len(dup_groups),
        "md5_dup_images": sum(len(v) for v in dup_groups.values()),
        "cluster": cluster_stat,
        "v4_leak": v4_stat,
        "provenance": {"covered": cover, "total": len(items),
                       "ratio": round(cover / max(1, len(items)), 4)},
        "elapsed_s": round(time.time() - t0, 1),
    }
    os.makedirs("reports", exist_ok=True)
    outp = args.out or os.path.join("reports", "leak_analysis_%s.json" % time.strftime("%Y%m%d"))
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("已写出", outp)


if __name__ == "__main__":
    main()
