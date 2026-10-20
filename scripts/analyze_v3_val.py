# -*- coding: utf-8 -*-
# V3 验证集错题本：漏检(FN) / 误报(FP) 逐图导出 + 损伤类标注审计
import os, sys, csv, json, math
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(r"D:\CT Detection")
DATA = ROOT / "datasets" / "CT_DATASET_FINAL_V3_AUG"
VAL_IMG = DATA / "val" / "images"
VAL_LAB = DATA / "val" / "labels"
WEIGHTS = ROOT / "runs" / "detect" / "CT_pro" / "V3" / "weights" / "best.pt"
OUT = ROOT / "复核_V3"
CONF = 0.25
IOU = 0.5
CLS = {0: "damage", 1: "dark", 2: "light"}
CLS_ZH = {0: "损伤", 1: "暗斑", 2: "亮斑"}

def imread(p):
    return cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_COLOR)

def imwrite(p, img):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", img)[1].tofile(str(p))

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = (a[2]-a[0]) * (a[3]-a[1]); bb = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (aa + bb - inter + 1e-9)

def read_gt(lab_path, w, h):
    boxes = []
    if lab_path.exists():
        for line in lab_path.read_text(encoding="utf-8").splitlines():
            t = line.split()
            if len(t) < 5: continue
            c = int(t[0]); cx, cy, bw, bh = map(float, t[1:5])
            boxes.append({"cls": c,
                          "xyxy": [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h],
                          "wh": [bw*w, bh*h]})
    return boxes

def draw(img, box, color, text):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(img, (x1, max(0, y1-th-10)), (x1+tw+6, y1), color, -1)
    cv2.putText(img, text, (x1+3, y1-6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

def main():
    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    imgs = sorted(p for p in VAL_IMG.iterdir() if p.suffix.lower() == ".jpg")
    done_file = OUT / "_done.txt"
    OUT.mkdir(exist_ok=True)
    done = set(done_file.read_text(encoding="utf-8").splitlines()) if done_file.exists() else set()
    rows_f = open(OUT / "错题清单.csv", "a", newline="", encoding="utf-8-sig")
    writer = csv.writer(rows_f)
    if not done:
        writer.writerow(["类型", "图片", "类别", "置信度", "x1", "y1", "x2", "y2", "宽px", "高px"])
    stats = {}  # cls -> [tp, fn, fp]
    audit_damage = []  # 损伤GT框统计

    for p in imgs:
        if p.name in done: continue
        img = imread(p)
        if img is None:
            done.add(p.name); continue
        h, w = img.shape[:2]
        gt = read_gt(VAL_LAB / (p.stem + ".txt"), w, h)
        r = model.predict(str(p), imgsz=1920, conf=CONF, verbose=False, device=0)[0]
        pred = []
        if r.boxes is not None and len(r.boxes):
            for b in r.boxes:
                pred.append({"cls": int(b.cls), "conf": float(b.conf),
                             "xyxy": [float(v) for v in b.xyxy[0]]})
        matched_gt, matched_pr = set(), set()
        for pi, pb in enumerate(pred):
            best_i, best_g = 0, -1
            for gi, gb in enumerate(gt):
                if gi in matched_gt or gb["cls"] != pb["cls"]: continue
                v = iou(pb["xyxy"], gb["xyxy"])
                if v > best_i: best_i, best_g = v, gi
            if best_i >= IOU and best_g >= 0:
                matched_gt.add(best_g); matched_pr.add(pi)
        fn = [g for gi, g in enumerate(gt) if gi not in matched_gt]
        fp = [pb for pi, pb in enumerate(pred) if pi not in matched_pr]
        for g in gt:
            s = stats.setdefault(g["cls"], [0, 0, 0])
            if g in fn: s[1] += 1
            else: s[0] += 1
            if g["cls"] == 0:
                audit_damage.append([p.name, round(g["wh"][0],1), round(g["wh"][1],1),
                                     round(g["wh"][0]*g["wh"][1]/(w*h)*100, 3)])
        for pb in fp:
            stats.setdefault(pb["cls"], [0,0,0])[2] += 1
        if fn:
            vis = img.copy()
            for g in fn:
                draw(vis, g["xyxy"], (0,0,255), f"MISS {CLS[g['cls']]}")
                writer.writerow(["漏检", p.name, CLS_ZH[g["cls"]], "", *[round(v) for v in g["xyxy"]],
                                 round(g["wh"][0]), round(g["wh"][1])])
                out = OUT / f"漏检_{CLS_ZH[g['cls']]}" / p.name
                if not out.exists(): imwrite(out, vis)
        if fp:
            vis = img.copy()
            for pb in fp:
                draw(vis, pb["xyxy"], (255,128,0), f"FP {CLS[pb['cls']]} {pb['conf']:.2f}")
                bw, bh = pb["xyxy"][2]-pb["xyxy"][0], pb["xyxy"][3]-pb["xyxy"][1]
                writer.writerow(["误报", p.name, CLS_ZH[pb["cls"]], round(pb["conf"],3),
                                 *[round(v) for v in pb["xyxy"]], round(bw), round(bh)])
                out = OUT / f"误报_{CLS_ZH[pb['cls']]}" / p.name
                if not out.exists(): imwrite(out, vis)
        done.add(p.name)
        if len(done) % 50 == 0:
            done_file.write_text("\n".join(done), encoding="utf-8")
            rows_f.flush()
            print(f"progress {len(done)}/{len(imgs)}", flush=True)
    rows_f.close()
    done_file.write_text("\n".join(done), encoding="utf-8")

    print("\n===== 验证集成绩单（conf=%.2f, IoU=%.2f）=====" % (CONF, IOU))
    for c in sorted(stats):
        tp, f_, fp_ = stats[c]
        prec = tp/(tp+fp_) if tp+fp_ else 0
        rec = tp/(tp+f_) if tp+f_ else 0
        print(f"{CLS_ZH[c]}: TP={tp} 漏检={f_} 误报={fp_} | 精确率={prec*100:.1f}% 召回={rec*100:.1f}%")
    if audit_damage:
        ws = [a[1] for a in audit_damage]; hs = [a[2] for a in audit_damage]; ar = [a[3] for a in audit_damage]
        print(f"\n===== 损伤类GT标注审计（{len(audit_damage)}个框）=====")
        print(f"框宽px: 最小{min(ws)} 中位{sorted(ws)[len(ws)//2]} 最大{max(ws)}")
        print(f"框高px: 最小{min(hs)} 中位{sorted(hs)[len(hs)//2]} 最大{max(hs)}")
        print(f"占图面积%: 最小{min(ar)} 中位{sorted(ar)[len(ar)//2]} 最大{max(ar)}")
        tiny = [a for a in audit_damage if a[1] < 30 or a[2] < 30]
        huge = [a for a in audit_damage if a[3] > 5]
        print(f"过小框(<30px边长): {len(tiny)}个; 过大框(>5%图面积): {len(huge)}个")
        if tiny: print("  过小示例:", tiny[:10])
        if huge: print("  过大示例:", huge[:10])

if __name__ == "__main__":
    main()
