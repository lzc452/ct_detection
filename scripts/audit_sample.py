#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P0-1 损伤标注一致性审计：抽样清单、盲审作业单、预览裁剪图、IoU 评分

背景：V5.2 与 V5 的损伤 AP 上限（混淆矩阵对角，conf=0.001）分别只有 0.45 / 0.49，
      但该数值建立在 test 的 73 个损伤框上。若这几个框人工重标都画不到一起，
      "0.49 是不是模型的天花板"就无法回答，后续所有针对损伤的优化方向都会站错地基。

抽样设计（不是简单抽 100 个，理由见 docs/audit_protocol_20261023.md）：
  批 A  test 全量 73 框（27 组）  —— 判据主样本，决定评测可信度
  批 B  train 分层 100 框        —— 条件触发，决定是否清洗训练标签
  分层维度：框长边三分位（S/M/L）。全局切点由 484 个损伤框算出。
  组内上限 2 框：同组图为同一电芯连拍，标 4 张相似图不增加信息量。

用法:
  python scripts/audit_sample.py --build
  python scripts/audit_sample.py --score --pred-a reports/audit/labelme_A --pred-b reports/audit/labelme_B
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import random
import shutil
import sys
from statistics import median

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MANIFEST = os.path.join(ROOT, "reports", "splits", "v5_split_manifest.csv")
OUTDIR = os.path.join(ROOT, "reports", "audit")
DAMAGE = 0

# 判据（预注册，写在协议文档里，评分只引用不改）
GATE_PASS = 0.75      # IoU(A,B) 中位 >= 0.75 → 标注可用于当前精度区间
GATE_STOP = 0.65      # IoU(A,B) 中位 < 0.65  → 停止一切模型侧优化，先修标注规范


# ---------------------------------------------------------------- 数据收集

def collect_damage_boxes(manifest_path: str):
    """从划分清单 + labelme json 取全部损伤框（原图坐标）。"""
    rows = list(csv.DictReader(io.open(manifest_path, encoding="utf-8-sig")))
    recs = []
    for r in rows:
        if not r.get("label_json"):
            continue
        if int(float(r.get("n_damage") or 0)) == 0:
            continue
        jp = r["label_json"]
        if not os.path.isabs(jp):
            jp = os.path.join(ROOT, jp)
        if not os.path.exists(jp):
            continue
        try:
            d = json.load(io.open(jp, encoding="utf-8"))
        except Exception:
            continue
        W = d.get("imageWidth") or 0
        H = d.get("imageHeight") or 0
        if not W or not H:
            try:
                w_, h_ = r["size"].split("x")
                W, H = int(w_), int(h_)
            except Exception:
                continue
        for idx, sh in enumerate(d.get("shapes", [])):
            try:
                cls = int(sh["label"])
            except Exception:
                continue
            if cls != DAMAGE:
                continue
            pts = sh.get("points") or []
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            img = r["src_path"]
            if not os.path.isabs(img):
                img = os.path.join(ROOT, img)
            recs.append(dict(
                split=r["split"], group=r["group"], name=r["name"],
                image=img, label_json=jp, shape_idx=idx,
                x1=round(x1, 1), y1=round(y1, 1), x2=round(x2, 1), y2=round(y2, 1),
                w=round(w, 1), h=round(h, 1),
                ls=round(max(w, h), 1), ss=round(min(w, h), 1),
                cx=round((x1 + x2) / 2.0, 1), cy=round((y1 + y2) / 2.0, 1),
                img_w=W, img_h=H,
                n_damage_img=int(float(r.get("n_damage") or 0)),
                n_boxes_img=int(float(r.get("n_boxes") or 0)),
            ))
    return recs


def size_bins(recs):
    """按长边三分位切点（由全部损伤框决定，保证跨 split 可比）。"""
    v = sorted(x["ls"] for x in recs)
    n = len(v)
    q1 = v[int(n * 0.333)]
    q2 = v[int(n * 0.667)]

    def tag(ls: float) -> str:
        if ls <= q1:
            return "S"
        if ls <= q2:
            return "M"
        return "L"
    return tag, q1, q2


# ---------------------------------------------------------------- 抽样

def sample_stratified(pool, want_by_bin, rng, cap_per_group=2):
    """层内按 group 轮转抽样：先每组取 1，取完再取第 2，最大化覆盖的电芯数。

    cap_per_group 跨层共享（用全局计数），否则同一颗电芯可能在 S/M/L 各取 1 个、
    合计 3 框，违背"每组最多 2 框"的约束。
    """
    picked = []
    detail = {}
    gcnt = collections.Counter()
    for b in ("S", "M", "L"):
        want = int(want_by_bin.get(b, 0))
        cands = [c for c in pool if c["bin"] == b]
        groups = collections.defaultdict(list)
        for c in cands:
            groups[c["group"]].append(c)
        order = sorted(groups)
        rng.shuffle(order)
        for g in order:
            rng.shuffle(groups[g])
        got = 0
        layer_groups = set()
        while got < want:
            progressed = False
            for g in order:
                if got >= want:
                    break
                if gcnt[g] >= cap_per_group or not groups[g]:
                    continue
                picked.append(groups[g].pop(0))
                gcnt[g] += 1
                got += 1
                layer_groups.add(g)
                progressed = True
            if not progressed:
                break
        detail[b] = dict(want=want, got=got, pool=len(cands),
                         groups_used=len(layer_groups))
    return picked, detail


# ---------------------------------------------------------------- 预览图

def crop_geometry(it, scale=2, min_w=120, max_w=560, min_h=150, max_h=900,
                  w_mult=8.0, h_mult=1.6):
    """各向异性视野：宽按短边给余量，高按长边给余量。放大倍数固定，便于跨框可比。

    为什么固定 2× 而不按短边放大到足够大：
      裁剪图放大 m 倍后，标注员在屏幕上的手抖 ±e 像素折算回原图只有 ±e/m。
      若按短边放大到 3~7×，IoU 会被系统性抬高，审计变成自我安慰。
      现役 V5 输入是原图 0.90×、V5.2 是 ROI 1.55×，故 2× 接近真实成像条件，
      得到的一致性仍属乐观上界（标注员看到的结构比模型更清楚），报告须披露。

    视野做成各向异性，因为损伤长宽比中位 4.13、p90 17.05：
      方形视野会把细长划痕压成 1~2% 宽的线（首版 4× 方形视野的实测问题）。
    """
    W, H = it["img_w"], it["img_h"]
    w, h = max(1.0, it["w"]), max(1.0, it["h"])
    vw = min(max_w, max(min_w, w * w_mult))
    vh = min(max_h, max(min_h, h * h_mult))
    vw = min(vw, W)
    vh = min(vh, H)
    scale = max(1, int(scale))
    while scale > 1 and max(vw, vh) * scale > 1600:
        scale -= 1
    return int(round(vw)), int(round(vh)), scale


def make_crops(items, crop_dir: str, scale=2):
    """裁剪视野并放大保存；图上不留任何标注痕迹（盲审）。"""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    os.makedirs(crop_dir, exist_ok=True)
    cache = {}
    made = 0
    for it in items:
        p = it["image"]
        if p not in cache:
            try:
                cache[p] = Image.open(p).convert("RGB")
            except Exception as e:
                print("  [warn] 打不开 %s: %s" % (p, e))
                cache[p] = None
        im = cache[p]
        if im is None:
            continue
        W, H = im.size
        it["img_w"], it["img_h"] = W, H
        vw, vh, sc = crop_geometry(it, scale=scale)
        x0 = int(round(it["cx"] - vw / 2.0))
        y0 = int(round(it["cy"] - vh / 2.0))
        x0 = max(0, min(x0, W - vw))
        y0 = max(0, min(y0, H - vh))
        crop = im.crop((x0, y0, x0 + vw, y0 + vh))
        if sc > 1:
            crop = crop.resize((vw * sc, vh * sc), Image.LANCZOS)
        dst = os.path.join(crop_dir, "%s.jpg" % it["audit_id"])
        crop.save(dst, quality=95)
        it["crop_path"] = os.path.relpath(dst, ROOT).replace("\\", "/")
        it["crop_x0"], it["crop_y0"] = x0, y0
        it["crop_w"], it["crop_h"], it["crop_scale"] = vw, vh, sc
        made += 1
    cache.clear()
    return made


def make_contact_sheet(items, dst, cols=7, thumb=230, pad=6):
    """拼图索引（含原框，供主审人核对裁剪是否真的对上了目标；勿发标注员）。"""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None
    if not items:
        return None
    rows = (len(items) + cols - 1) // cols
    cell_w = thumb + pad * 2
    cell_h = thumb + pad * 2 + 16
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (245, 245, 245))
    dr = ImageDraw.Draw(sheet)
    for i, it in enumerate(items):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        cp = os.path.join(ROOT, it["crop_path"]) if it.get("crop_path") else ""
        if not cp or not os.path.exists(cp):
            continue
        try:
            t = Image.open(cp).convert("RGB")
        except Exception:
            continue
        sx = thumb / float(t.width)
        sy = thumb / float(t.height)
        s = min(sx, sy)
        tw, th = int(t.width * s), int(t.height * s)
        t = t.resize((tw, th), Image.LANCZOS)
        ox = x + pad + (thumb - tw) // 2
        oy = y + pad + (thumb - th) // 2
        sheet.paste(t, (ox, oy))
        # 原框映射到缩略图坐标
        sc = float(it.get("crop_scale") or 1)
        bx1 = (float(it["x1"]) - float(it["crop_x0"])) * sc * s
        by1 = (float(it["y1"]) - float(it["crop_y0"])) * sc * s
        bx2 = (float(it["x2"]) - float(it["crop_x0"])) * sc * s
        by2 = (float(it["y2"]) - float(it["crop_y0"])) * sc * s
        dr.rectangle([ox + bx1, oy + by1, ox + bx2, oy + by2], outline=(255, 40, 40))
        dr.text((x + pad, y + pad + thumb + 3), "%s %s" % (it["audit_id"], it["bin"]),
                fill=(20, 20, 20))
    sheet.save(dst, quality=88)
    return dst


# ---------------------------------------------------------------- 输出

def write_manifest_csv(items, path):
    cols = ["audit_id", "batch", "split", "group", "name", "image", "label_json",
            "shape_idx", "x1", "y1", "x2", "y2", "w", "h", "ls", "ss",
            "cx", "cy", "bin", "img_w", "img_h",
            "n_damage_img", "n_boxes_img",
            "crop_path", "crop_x0", "crop_y0", "crop_w", "crop_h", "crop_scale"]
    with io.open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for it in items:
            w.writerow(it)


def write_fill_template(items, path):
    """标注员的填写模板（labelme 为主路径，此表仅作兜底/补备注）。"""
    with io.open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "annotator", "x1", "y1", "x2", "y2", "note"])
        for it in items:
            w.writerow([it["audit_id"], "", "", "", "", "", ""])


def write_worksheet(items, path, title, batch_note):
    """盲审作业单：只给编号、裁剪图、裁剪偏移。不给原框坐标/尺寸/类别。"""
    L = []
    L.append("# %s" % title)
    L.append("")
    L.append("> 本单为**盲审**：不提供原标注的坐标、尺寸、数量。请独立判断。")
    L.append("")
    L.append("## 一、要做什么")
    L.append("")
    L.append("对每张裁剪图，标注出图中**所有**你认为属于「极片损伤 / 划痕」的区域，"
             "按你判断的**完整边界**画矩形框。")
    L.append("")
    L.append("**不要**参考任何已有标注，**不要**与他人讨论。两个人各自独立完成。")
    L.append("")
    L.append("## 二、怎么操作")
    L.append("")
    L.append("1. `labelme` 打开裁剪图目录（见下表 `裁剪图` 列所在的文件夹）；"
             "图幅偏小属**正常**（刻意只放大 2×，避免把标注精度人为抬高），"
             "用滚轮放大查看细节即可")
    L.append("2. 逐张画框，**标签名按下表规范填**（这一步比画框本身更重要）")
    L.append("3. 若某张图确实看不到任何损伤，**不要画框**，直接保存空 json")
    L.append("4. 全部做完后把整个 json 目录交回")
    L.append("")
    L.append("### 标签名规范（用来区分「边界模糊」和「标注马虎」）")
    L.append("")
    L.append("| 标签名 | 什么时候用 |")
    L.append("|---|---|")
    L.append("| `damage` | 痕迹清楚，边界能确定，换个角度看也是这条 |")
    L.append("| `damage_soft` | 能看到痕迹，但边界只能靠判断，差几个像素很正常 |")
    L.append("| `damage_unsure` | 几乎看不出，凭经验认为这里有问题 |")
    L.append("")
    L.append("> 千万不要为了让框\"好看\"而把边界画得比实际更整齐。"
             "这份数据要测的正是边界有多不确定，画得越像标准答案，结果越没用。")
    L.append("")
    L.append("> 坐标换算由脚本完成（裁剪偏移见下表），你**不需要**换算或填写坐标。")
    L.append("")
    L.append("## 三、这份数据用来判什么")
    L.append("")
    L.append("| 指标 | 含义 | 门槛 |")
    L.append("|---|---|---|")
    L.append("| IoU(A,B) 中位 | 两个人对同一划痕边界的一致程度 | ≥%.2f 通过 |" % GATE_PASS)
    L.append("| | | <%.2f 停止模型侧优化，先修标注规范 |" % GATE_STOP)
    L.append("")
    L.append("判读逻辑：如果人工彼此都画不到 %.2f，那么模型 mAP 的天花板里有一部分"
             "属于「标注定义本身模糊」，不是模型缺陷。" % GATE_STOP)
    L.append("")
    L.append(batch_note)
    L.append("")
    L.append("## 五、逐框清单（%d 框）" % len(items))
    L.append("")
    L.append("| 编号 | 裁剪图 | 裁剪偏移(原图 x0, y0) | 放大 | 画好了 |")
    L.append("|---|---|---|---|---|")
    for it in items:
        L.append("| %s | `%s` | %d, %d | %d× | ☐ |" % (
            it["audit_id"], it["crop_path"], it["crop_x0"], it["crop_y0"],
            int(it.get("crop_scale") or 1)))
    L.append("")
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L))


# ---------------------------------------------------------------- 评分

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw = max(0.0, x2 - x1); ih = max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def geom_pair(a, b):
    """返回 (IoU, 中心距 px, 宽度比, 高度比)。

    损伤长宽比中位 4.13，IoU 对短边极敏感：短边 17px 时单边 3px 抖动就能把 IoU
    从 0.85 拉到 0.6。只报 IoU 会混淆两种完全不同的失败——
      中心距小 + IoU 低 = 看到的是同一处，只是边界画法不同（尺寸分歧）
      中心距大          = 根本没对准同一处（匹配/存在性分歧）
    """
    xa = (a[0] + a[2]) / 2.0; ya = (a[1] + a[3]) / 2.0
    xb = (b[0] + b[2]) / 2.0; yb = (b[1] + b[3]) / 2.0
    cd = ((xa - xb) ** 2 + (ya - yb) ** 2) ** 0.5
    wa = max(1e-6, a[2] - a[0]); ha = max(1e-6, a[3] - a[1])
    wb = max(1e-6, b[2] - b[0]); hb = max(1e-6, b[3] - b[1])
    return iou_xyxy(a, b), cd, wa / wb, ha / hb


def read_labelme_dir(d):
    """返回 {audit_id: [(label, [x1,y1,x2,y2]), ...]}（裁剪图坐标）。"""
    out = collections.defaultdict(list)
    if not d or not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        if not fn.lower().endswith(".json"):
            continue
        aid = os.path.splitext(fn)[0]
        try:
            j = json.load(io.open(os.path.join(d, fn), encoding="utf-8"))
        except Exception:
            continue
        for sh in j.get("shapes", []):
            pts = sh.get("points") or []
            if len(pts) < 2:
                continue
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            out[aid].append((str(sh.get("label", "")),
                             [min(xs), min(ys), max(xs), max(ys)]))
    return out


def read_fill_csv(p, annotator):
    out = collections.defaultdict(list)
    if not p or not os.path.exists(p):
        return out
    for r in csv.DictReader(io.open(p, encoding="utf-8-sig")):
        if (r.get("annotator") or "").strip() != annotator:
            continue
        try:
            box = [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])]
        except Exception:
            continue
        out[r["audit_id"].strip()].append((str(r.get("note") or ""), box))
    return out


def pct(v, q):
    if not v:
        return None
    s = sorted(v)
    return s[min(len(s) - 1, int(len(s) * q))]


def describe(v):
    if not v:
        return dict(n=0)
    return dict(n=len(v), median=round(median(v), 4),
                p10=round(pct(v, .10), 4), p25=round(pct(v, .25), 4),
                p75=round(pct(v, .75), 4), p90=round(pct(v, .90), 4),
                mean=round(sum(v) / len(v), 4))


def score(items, boxA, boxB):
    """boxA/boxB 为裁剪图坐标。换算到原图坐标后比较。

    保留 existence 标记：双方都画了框才计入"边界一致性"，只有一方画了框属于
    "存在性分歧"（这条到底算不算损伤），两者混在一算会把结论糊掉。
    """
    rows = []
    for it in items:
        aid = it["audit_id"]
        ox, oy = float(it["crop_x0"]), float(it["crop_y0"])
        sc = float(it.get("crop_scale") or 1)
        ref = [float(it["x1"]), float(it["y1"]), float(it["x2"]), float(it["y2"])]

        def conv(lst):
            return [(lb, [b[0] / sc + ox, b[1] / sc + oy,
                          b[2] / sc + ox, b[3] / sc + oy]) for lb, b in lst]

        A = conv(boxA.get(aid, []))
        B = conv(boxB.get(aid, []))
        ia, la = max([(iou_xyxy(b, ref), lb) for lb, b in A], default=(None, ""))
        ib, lb_ = max([(iou_xyxy(b, ref), lb) for lb, b in B], default=(None, ""))
        ab, lab_ab, cd, wr, hr = None, "", None, None, None
        if A and B:
            best = None
            for la_, ba in A:
                for lb2, bb in B:
                    v, d, rw, rh = geom_pair(ba, bb)
                    if best is None or v > best[0]:
                        best = (v, d, rw, rh, la_)
            ab, cd, wr, hr, lab_ab = best
        if A and B:
            status = "both"
        elif A:
            status = "onlyA"
        elif B:
            status = "onlyB"
        else:
            status = "none"
        rows.append(dict(audit_id=aid, batch=it["batch"], split=it["split"],
                         bin=it["bin"], ls=it["ls"], ss=it["ss"], group=it["group"],
                         nA=len(A), nB=len(B), status=status,
                         label_a=la, label_b=lb_,
                         iou_ab=ab, center_dist=cd, w_ratio=wr, h_ratio=hr,
                         iou_a_ref=ia, iou_b_ref=ib))
    return rows


def score_report(rows, out_md, out_json):
    both = [r for r in rows if r["status"] == "both"]
    ab = [r["iou_ab"] for r in both]
    a_ref = [r["iou_a_ref"] for r in rows if r["iou_a_ref"] is not None]
    b_ref = [r["iou_b_ref"] for r in rows if r["iou_b_ref"] is not None]
    exist = collections.Counter(r["status"] for r in rows)

    def sym_dev(r):
        return 2.0 * abs(r - 1.0) / (r + 1.0)

    geo = dict(
        center_dist=describe([r["center_dist"] for r in both
                              if r["center_dist"] is not None]),
        w_dev=describe([sym_dev(r["w_ratio"]) for r in both if r["w_ratio"]]),
        h_dev=describe([sym_dev(r["h_ratio"]) for r in both if r["h_ratio"]]),
    )

    res = dict(schema="damage_audit/v1", n_items=len(rows),
               gate_pass=GATE_PASS, gate_stop=GATE_STOP,
               existence=dict(exist),
               existence_disagree=round(
                   (exist["onlyA"] + exist["onlyB"]) / float(len(rows)), 4) if rows else None,
               iou_AB=describe(ab), iou_A_ref=describe(a_ref), iou_B_ref=describe(b_ref),
               geometry=geo,
               by_batch={}, by_bin={}, by_label={},
               empty=dict(A=sum(1 for r in rows if r["nA"] == 0),
                          B=sum(1 for r in rows if r["nB"] == 0)))
    for k in ("A", "B"):
        res["by_batch"][k] = describe([r["iou_ab"] for r in both if r["batch"] == k])
    for k in ("S", "M", "L"):
        res["by_bin"][k] = describe([r["iou_ab"] for r in both if r["bin"] == k])
    # 按标注员自评难度档拆分：区分"目标本身没边界"和"标注马虎"
    for k in ("damage", "damage_soft", "damage_unsure"):
        res["by_label"][k] = describe([r["iou_ab"] for r in both
                                       if r["label_a"] == k and r["label_b"] == k])

    med = median(ab) if ab else None
    if med is None:
        verdict = "无有效标注"
    elif med >= GATE_PASS:
        verdict = "通过：标注可用于当前精度区间，模型侧优化可继续"
    elif med >= GATE_STOP:
        verdict = "灰区：标注有抖动但未致命，优化时需把容差算进预期"
    else:
        verdict = "不通过：先修标注规范，暂停模型侧优化"
    res["verdict"] = verdict
    res["median_AB"] = round(med, 4) if med is not None else None

    json.dump(res, io.open(out_json, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    L = ["# 损伤标注一致性审计结果", "",
         "样本 %d 框；双方都画了框的 %d 框参与主指标" % (len(rows), len(both)), ""]
    L += ["## 一、存在性分歧（先看这个）", "",
          "| 情况 | 框数 | 占比 |", "|---|---|---|"]
    for k, name in (("both", "双方都标了"), ("onlyA", "只有 A 标"),
                    ("onlyB", "只有 B 标"), ("none", "双方都没标")):
        n = exist.get(k, 0)
        L.append("| %s | %d | %.1f%% |" % (name, n, 100.0 * n / max(1, len(rows))))
    L += ["", "> 只有一方标、或双方都没标，属于「这算不算损伤」的分歧。"
          "这部分不是边界精度问题，而是标签定义问题；它若占比高，说明类别定义本身需要写清楚，"
          "任何模型都学不到一致的东西。", ""]
    L += ["## 二、主指标 IoU(A,B)（边界一致性）", "",
          "| 指标 | 值 |", "|---|---|"]
    for k, v in res["iou_AB"].items():
        L.append("| %s | %s |" % (k, v))
    L += ["", "## 三、几何分解（IoU 低到底是「没对准」还是「边界画法不同」）", "",
          "| 指标 | n | 中位 | p75 | p90 |", "|---|---|---|---|---|"]
    for name, k in (("中心点距离 (px)", "center_dist"),
                    ("宽度相对偏差", "w_dev"), ("长度相对偏差", "h_dev")):
        v = geo[k]
        L.append("| %s | %s | %s | %s | %s |" % (name, v.get("n"), v.get("median"),
                                                 v.get("p75"), v.get("p90")))
    L += ["", "读法：中心点距离小（≤10px）而 IoU 低 → 双方看到的是同一处缺陷，"
          "分歧在边界定义；中心点距离大 → 没对准同一处，属于匹配或存在性分歧。", ""]
    L += ["## 四、按尺寸档（长边 S/M/L）", "",
          "| 档 | n | 中位 | p25 | p75 |", "|---|---|---|---|---|"]
    for k, v in res["by_bin"].items():
        L.append("| %s | %s | %s | %s | %s |" % (k, v.get("n"), v.get("median"),
                                                 v.get("p25"), v.get("p75")))
    L += ["", "## 五、按标注员自评难度", "",
          "| 双方标签 | n | 中位 |", "|---|---|---|"]
    for k, v in res["by_label"].items():
        L.append("| %s | %s | %s |" % (k, v.get("n"), v.get("median")))
    L += ["", "读法：若 `damage`（双方都认为清晰）的中位仍低，说明是标注执行问题；"
              "若只有 `damage_soft` 低而 `damage` 高，说明是目标本身没边界。", ""]
    L += ["## 六、与原标注的偏离", "",
          "| 比较 | n | 中位 |", "|---|---|---|",
          "| IoU(A, 原标) | %s | %s |" % (res["iou_A_ref"].get("n"), res["iou_A_ref"].get("median")),
          "| IoU(B, 原标) | %s | %s |" % (res["iou_B_ref"].get("n"), res["iou_B_ref"].get("median")),
          "", "## 七、判定", "", "**%s**（IoU(A,B) 中位 = %s）" % (verdict, res["median_AB"]),
          "", "> 注意：本批裁剪图的视野中心往往就是目标所在，A/B 面临同样的位置暗示，",
          "> 故 IoU(A,B) 不受影响；但 IoU(原标, A/B) 可能因锚定而偏高，只作参考。"]
    io.open(out_md, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
    return res


# ---------------------------------------------------------------- main

def cmd_build(args):
    recs = collect_damage_boxes(MANIFEST)
    if not recs:
        print("未取到损伤框，检查 manifest 路径")
        return 1
    tag, q1, q2 = size_bins(recs)
    for r in recs:
        r["bin"] = tag(r["ls"])
    print("损伤框 %d 个；长边三分位切点 S<=%d, M<=%d, L>%d" % (len(recs), q1, q2, q2))

    os.makedirs(OUTDIR, exist_ok=True)
    rng = random.Random(args.seed)

    # 批 A：test 全量
    test_pool = [r for r in recs if r["split"] == "test"]
    test_pool.sort(key=lambda x: (x["group"], x["ls"]))
    items = []
    for i, it in enumerate(test_pool, 1):
        it["audit_id"] = "A%03d" % i
        it["batch"] = "A"
        items.append(it)
    print("批 A（test 全量）: %d 框 / %d 组" % (
        len(items), len(set(x["group"] for x in items))))

    # 批 B：train 分层
    if args.n_train > 0:
        train_pool = [r for r in recs if r["split"] == "train"]
        cnt = collections.Counter(x["bin"] for x in train_pool)
        tot = sum(cnt.values())
        want = {b: int(round(args.n_train * cnt[b] / tot)) for b in ("S", "M", "L")}
        diff = args.n_train - sum(want.values())
        if diff:
            want[max(want, key=lambda b: cnt[b])] += diff
        picked, detail = sample_stratified(train_pool, want, rng)
        picked.sort(key=lambda x: (x["group"], x["ls"]))
        for i, it in enumerate(picked, 1):
            it["audit_id"] = "B%03d" % i
            it["batch"] = "B"
            items.append(it)
        print("批 B（train 分层）: %d 框 / %d 组  分层明细 %s" % (
            len(picked), len(set(x["group"] for x in picked)),
            {k: (v["got"], v["pool"], v["groups_used"]) for k, v in detail.items()}))

    # 裁剪图
    print("生成预览裁剪图（各向异性视野，%d× 放大，图上不画框）…" % args.crop_scale)
    made = make_crops(items, os.path.join(OUTDIR, "crops"), scale=args.crop_scale)
    print("  已生成 %d 张" % made)

    write_manifest_csv(items, os.path.join(OUTDIR, "sample_manifest.csv"))
    write_fill_template(items, os.path.join(OUTDIR, "fill_template.csv"))

    A = [x for x in items if x["batch"] == "A"]
    B = [x for x in items if x["batch"] == "B"]
    make_contact_sheet(A, os.path.join(OUTDIR, "contact_A.jpg"))
    if B:
        make_contact_sheet(B, os.path.join(OUTDIR, "contact_B.jpg"))

    note_A = ("## 四、为什么 test 要全审而不是抽\n\n"
              "test 的损伤框共 73 个，是 mAP/召回/混淆矩阵的**唯一**样本。"
              "抽 30 个算中位数，95% 置信区间约 ±0.07，正好盖住 0.65 与 0.75 这道分界，"
              "等于问了也答不了。73 个全审的工作量约 1 小时/人，性价比远高于抽样。")
    write_worksheet(A, os.path.join(OUTDIR, "worksheet_A_test.md"),
                    "批 A · 损伤标注一致性审计作业单（test 全量 %d 框）" % len(A), note_A)

    if B:
        note_B = ("## 四、这批什么时候做\n\n"
                  "**条件触发**：先做批 A。若批 A 判定为「通过」，说明标注测量本身可信，"
                  "此批用于判断**训练标签**是否也需要清洗（决定是否花 1 天重标 338 框）；"
                  "若批 A 判定「不通过」，先修标注规范，本批顺延重做。")
        write_worksheet(B, os.path.join(OUTDIR, "worksheet_B_train.md"),
                        "批 B · 损伤标注一致性审计作业单（train 分层 %d 框）" % len(B), note_B)

    # 主审人用的定位核对表（含真实信息，不发给标注员）
    L = ["# 主审人核对表（含原标信息，勿发标注员）", "",
         "长边三分位切点：S≤%d，M≤%d，L>%d" % (q1, q2, q2), "",
         "| 编号 | split | 组 | 原标 x1,y1,x2,y2 | 长×宽 | 档 | 裁剪偏移 | 放大 | 裁剪图尺寸 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for it in items:
        L.append("| %s | %s | %s | %.0f,%.0f,%.0f,%.0f | %.0f×%.0f | %s | %d,%d | %d× | %d×%d |" % (
            it["audit_id"], it["split"], it["group"],
            it["x1"], it["y1"], it["x2"], it["y2"], it["ls"], it["ss"],
            it["bin"], it["crop_x0"], it["crop_y0"], int(it.get("crop_scale") or 1),
            it["crop_w"] * int(it.get("crop_scale") or 1),
            it["crop_h"] * int(it.get("crop_scale") or 1)))
    io.open(os.path.join(OUTDIR, "checklist_reviewer.md"), "w",
            encoding="utf-8", newline="\n").write("\n".join(L) + "\n")

    summary = dict(
        schema="damage_audit_sample/v1", seed=args.seed,
        total_damage_boxes=len(recs),
        bins=dict(q33=q1, q67=q2),
        batch_A=dict(n=len(A), groups=len(set(x["group"] for x in A)),
                     rule="test 全量"),
        batch_B=dict(n=len(B), groups=len(set(x["group"] for x in B)),
                     rule="train 按长边三分位分层，组内上限 2 框"),
        crops=made,
    )
    json.dump(summary, io.open(os.path.join(OUTDIR, "sample_summary.json"), "w",
                               encoding="utf-8"), ensure_ascii=False, indent=2)

    print("\n输出目录: %s" % OUTDIR)
    for f in ("sample_manifest.csv", "fill_template.csv", "worksheet_A_test.md",
              "worksheet_B_train.md", "checklist_reviewer.md", "contact_A.jpg",
              "contact_B.jpg", "sample_summary.json"):
        p = os.path.join(OUTDIR, f)
        print("  %-26s %s" % (f, "OK" if os.path.exists(p) else "MISSING"))
    return 0


def cmd_score(args):
    mp = os.path.join(OUTDIR, "sample_manifest.csv")
    if not os.path.exists(mp):
        print("缺少 %s，先跑 --build" % mp)
        return 1
    items = list(csv.DictReader(io.open(mp, encoding="utf-8-sig")))
    boxA = read_labelme_dir(args.pred_a) if args.pred_a else read_fill_csv(args.fill, "A")
    boxB = read_labelme_dir(args.pred_b) if args.pred_b else read_fill_csv(args.fill, "B")
    if not boxA or not boxB:
        print("A 或 B 的标注为空：A=%d 框, B=%d 框" % (
            sum(len(v) for v in boxA.values()), sum(len(v) for v in boxB.values())))
        return 1
    rows = score(items, boxA, boxB)
    res = score_report(rows, os.path.join(OUTDIR, "score_report.md"),
                       os.path.join(OUTDIR, "score_report.json"))
    print(json.dumps({k: res[k] for k in ("n_items", "median_AB", "iou_AB",
                                          "by_bin", "verdict")},
                     ensure_ascii=False, indent=2))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--seed", type=int, default=20261023)
    ap.add_argument("--crop-scale", type=int, default=2, dest="crop_scale",
                    help="裁剪图放大倍数，默认 2（接近 V5 的 0.90× / V5.2 的 1.55×）")
    ap.add_argument("--n-train", type=int, default=100, dest="n_train")
    ap.add_argument("--pred-a", default=None, dest="pred_a")
    ap.add_argument("--pred-b", default=None, dest="pred_b")
    ap.add_argument("--fill", default=os.path.join(OUTDIR, "fill_template.csv"))
    args = ap.parse_args()
    if args.build:
        return cmd_build(args)
    if args.score:
        return cmd_score(args)
    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
