# -*- coding: utf-8 -*-
"""
非交互训练入口（脚本化 / 可复现）

与 scripts/train.py 的区别：
- 全部参数走命令行，不 input()，可在后台跑
- **imgsz 只能是方形 int**（ultralytics 硬约束：trainer/validator 的 check_imgsz(max_dim=1)），
  传 '1184x2112' 会被自动取长边 2112 并打 WARNING。这里用 resolve_imgsz(..., for_what="train")
  显式做这个降级，避免"以为用了矩形画布"的误解。
- 训练结束把训练元信息写进 run 目录 `_train_meta.json`（数据集、清单 md5、imgsz、全部生效超参）
- **训练日志自动落盘**：run 目录下的 `train.log`（stdout+stderr 全量，含每轮指标行）
- 训练完不自动启用模型（项目红线：必须人工确认后再复制进 models/启用/）

V5.1 相对 V5 的关键改动（依据见 docs/preprocess_analysis_20261021.md 与 docs/v51_eval_*.md）：
  画布 1920 -> 2112（原生 1:1）；mosaic 1.0 -> 0.0（消除训练/推理 2 倍尺度错配）；
  degrees 5 -> 0；erasing 0.4(默认) -> 0；cls_pw 0 -> 0.5（damage 类加权）

用法：
  runtime\\python.exe scripts\\train_run.py --data datasets_v5\\CT_DATASET_V5_GROUPED\\data.yaml ^
      --model models\\pretrained\\yolov8n.pt --imgsz 2112 --epochs 80 --mosaic 0 ^
      --degrees 0 --erasing 0 --cls-pw 0.5 --name V51_PREPROC
"""
import os
import sys
import json
import time
import shutil
import datetime
import argparse
import hashlib
import logging

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "core"))
os.chdir(PROJECT_ROOT)

# Windows 下 PROCESSOR_ARCHITECTURE 缺失会让 py-cpuinfo 崩溃（项目硬性约定）
os.environ.setdefault("PROCESSOR_ARCHITECTURE", os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"))

from ultralytics import YOLO  # noqa: E402
from preprocess import resolve_imgsz  # noqa: E402
import roi as roi_mod  # noqa: E402


def md5_of(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def count_imgs(d):
    return len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]) \
        if os.path.isdir(d) else 0


class _Tee:
    """把 stdout/stderr 同时写进日志文件。

    注意：光把 sys.stdout/sys.stderr 换掉**捕获不到** ultralytics 的日志行——
    它的 `LOGGER` 在 `from ultralytics import YOLO` 那一刻就把 StreamHandler 绑到了
    原始的 sys.stderr 上，之后换 sys.stderr 对它无效（2026-10-22 V51_ROI_1536 实测：
    train.log 只有 8 行表头）。因此还必须额外挂一个 logging.FileHandler（见 attach_logger）。
    这里同时补齐 isatty/fileno/encoding 等方法，避免 tqdm、rich 拿到裸对象后行为异常。
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    # ---- 让 tqdm / rich 认为这是个正常的流 ----
    def isatty(self):
        return False

    def writable(self):
        return True

    def readable(self):
        return False

    def fileno(self):
        for s in self.streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise OSError("no underlying fileno")

    @property
    def encoding(self):
        return getattr(self.streams[0], "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self.streams[0], "errors", "replace")

    def __getattr__(self, item):
        # 其余属性一律委托给第一条真实流
        return getattr(self.streams[0], item)


def attach_logger(log_path):
    """把 ultralytics 的 LOGGER（以及 root）额外写一份到 log_path。

    这是捕获每轮指标行的**唯一可靠手段**：LOGGER 的 StreamHandler 在 import 时就绑定了
    原始 stderr，事后替换 sys.stderr 拿不到它的输出。
    """
    import logging
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    try:
        import ultralytics
        ultralytics.LOGGER.addHandler(fh)
    except Exception:
        pass
    logging.getLogger().addHandler(fh)
    return fh


def rebuild_epoch_log(log_path, run_dir):
    """若控制台转录缺失（历史缺陷），用 results.csv 逐轮回放一张完整 epoch 表。

    不伪造任何数字：所有值都直接来自 ultralytics 自己写的 results.csv。
    """
    import csv
    csv_path = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv_path):
        return False
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    keys = [k.strip() for k in rows[0].keys()]
    # (列标题, 列宽) —— lr 那几个是 0.000666099 这种长小数，宽度要给够
    layout = [("box", 10), ("cls", 10), ("dfl", 10),
              ("P", 10), ("R", 10), ("mAP50", 10), ("mAP50-95", 11),
              ("vbox", 10), ("vcls", 10), ("vdfl", 10),
              ("lr_pg0", 13), ("lr_pg1", 13), ("lr_pg2", 13)]
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("=" * 120 + "\n")
        f.write("[注意] 上面的控制台转录不完整（ultralytics 的 LOGGER 在 import 时就绑定了原始 stderr，\n")
        f.write("       事后替换 sys.stdout/sys.stderr 抓不到它的输出）。以下 epoch 表由 results.csv\n")
        f.write("       逐轮回放，数值未经任何加工，可作为完整训练记录。\n")
        f.write("=" * 120 + "\n")
        f.write("%-5s" % "ep" + "".join("%-*s" % (w, h) for h, w in layout) + "\n")
        for r in rows:
            f.write("%-5s" % r[keys[0]].strip())
            for k, (_h, w) in zip(keys[2:], layout):
                f.write("%-*s" % (w, r[k].strip()))
            f.write("\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=os.path.join("models", "pretrained", "yolov8n.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--imgsz", default="1920",
                    help="方形画布边长；给 '1184x2112' 会按 ultralytics 约束取长边 2112（并打印说明）")
    ap.add_argument("--batch", default="0.75", help="数字或 0~1 的显存占比")
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume-from", default="",
                    help="微调来源权重（上一轮 run 的 weights/best.pt）；替换 --model，不复用优化器状态")
    # ---- 优化器 ----
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--optimizer", default="AdamW")
    ap.add_argument("--weight-decay", type=float, default=0.0005)
    # ---- 增强（V5.1 全部显式化，不依赖 ultralytics 默认值）----
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--close-mosaic", type=int, default=10)
    ap.add_argument("--degrees", type=float, default=5.0)
    ap.add_argument("--translate", type=float, default=0.05)
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--shear", type=float, default=0.0)
    ap.add_argument("--perspective", type=float, default=0.0)
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--hsv-h", type=float, default=0.0)
    ap.add_argument("--hsv-s", type=float, default=0.0)
    ap.add_argument("--hsv-v", type=float, default=0.1)
    ap.add_argument("--erasing", type=float, default=0.4,
                    help="随机擦除概率。ultralytics 8.4 默认 0.4，对本项目 15px 亮斑有误标风险，建议 0")
    ap.add_argument("--copy-paste", type=float, default=0.0,
                    help="注意：检测任务(无 segments)下 ultralytics 的 CopyPaste 是空操作，设了也无效")
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--cutmix", type=float, default=0.0)
    # ---- 损失 ----
    ap.add_argument("--box", type=float, default=7.5)
    ap.add_argument("--cls", type=float, default=0.5)
    ap.add_argument("--dfl", type=float, default=1.5)
    ap.add_argument("--cls-pw", type=float, default=0.0,
                    help="类别权重幂次 0~1：0 关闭；1=完全逆频率。本项目 damage 框最少，建议 0.5")
    ap.add_argument("--roi", action="store_true",
                    help="标记本模型是在 ROI 裁剪数据上训练的；写入 _train_meta.json 供推理端读取。"
                         "不改训练行为（数据本身已裁好），只影响部署时的推理口径")
    args = ap.parse_args()

    if args.resume_from:
        args.model = args.resume_from

    imgsz_raw = args.imgsz
    args.imgsz = resolve_imgsz(args.imgsz, for_what="train")

    name = args.name or "V5_%s" % datetime.datetime.now().strftime("%m%d_%H%M")
    batch = float(args.batch)
    batch = int(batch) if batch >= 1 else batch
    out_root = os.path.join(PROJECT_ROOT, "runs", "detect", "CT_pro")
    run_dir = os.path.join(out_root, name)
    os.makedirs(out_root, exist_ok=True)

    # 日志先写到 run 目录外（ultralytics 要求 run_dir 不存在），结束时再挪进去
    log_path = os.path.join(out_root, "_%s.train.log" % name)
    logf = open(log_path, "w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, logf)
    sys.stderr = _Tee(sys.__stderr__, logf)
    _log_handler = attach_logger(log_path)   # LOGGER 的输出另走 FileHandler

    ds_dir = os.path.dirname(os.path.abspath(args.data))
    n_train = count_imgs(os.path.join(ds_dir, "train", "images"))
    n_val = count_imgs(os.path.join(ds_dir, "val", "images"))
    print("=" * 68)
    print("数据集 %s | train %d | val %d" % (os.path.basename(ds_dir), n_train, n_val))
    print("模型 %s" % args.model)
    print("imgsz %s（请求 %s） | epochs %d | patience %d | batch %s | lr0 %g | opt %s" % (
        args.imgsz, imgsz_raw, args.epochs, args.patience, batch, args.lr0, args.optimizer))
    print("增强 mosaic %g | close_mosaic %d | degrees %g | translate %g | scale %g | erasing %g | fliplr %g | hsv_v %g" % (
        args.mosaic, args.close_mosaic, args.degrees, args.translate, args.scale,
        args.erasing, args.fliplr, args.hsv_v))
    print("损失 box %g cls %g dfl %g | cls_pw %g" % (args.box, args.cls, args.dfl, args.cls_pw))
    print("输出 %s" % run_dir)
    print("=" * 68)

    model = YOLO(args.model)
    t0 = time.time()
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        project=out_root,
        name=name,
        exist_ok=False,
        optimizer=args.optimizer,
        lr0=args.lr0, lrf=args.lrf, momentum=0.937, weight_decay=args.weight_decay, cos_lr=True,
        close_mosaic=args.close_mosaic,
        warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
        cache="disk", workers=args.workers, val=True, plots=True, seed=args.seed,
        box=args.box, cls=args.cls, dfl=args.dfl, cls_pw=args.cls_pw,
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        shear=args.shear, perspective=args.perspective, flipud=args.flipud, fliplr=args.fliplr,
        hsv_h=args.hsv_h, hsv_s=args.hsv_s, hsv_v=args.hsv_v,
        mosaic=args.mosaic, mixup=args.mixup, copy_paste=args.copy_paste,
        erasing=args.erasing,
    )

    best = os.path.join(run_dir, "weights", "best.pt")
    hyper = {
        "imgsz": args.imgsz, "imgsz_requested": imgsz_raw,
        "epochs": args.epochs, "patience": args.patience, "batch": batch,
        "optimizer": args.optimizer, "lr0": args.lr0, "lrf": args.lrf,
        "weight_decay": args.weight_decay, "cos_lr": True, "seed": args.seed,
        "mosaic": args.mosaic, "close_mosaic": args.close_mosaic,
        "degrees": args.degrees, "translate": args.translate, "scale": args.scale,
        "shear": args.shear, "perspective": args.perspective,
        "flipud": args.flipud, "fliplr": args.fliplr,
        "hsv_h": args.hsv_h, "hsv_s": args.hsv_s, "hsv_v": args.hsv_v,
        "erasing": args.erasing, "copy_paste": args.copy_paste,
        "mixup": args.mixup, "cutmix": args.cutmix,
        "box": args.box, "cls": args.cls, "dfl": args.dfl, "cls_pw": args.cls_pw,
    }
    # 记下 ultralytics 实际生效值，方便比对"我传的"和"它用的"
    effective = {}
    try:
        for k in hyper:
            if hasattr(model.trainer.args, k):
                effective[k] = getattr(model.trainer.args, k)
    except Exception:
        pass

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_yaml": os.path.abspath(args.data),
        "dataset": os.path.basename(ds_dir),
        "n_train": n_train, "n_val": n_val,
        "base_model": args.model,
        "base_model_md5": md5_of(args.model) if os.path.exists(args.model) else None,
        "resume_from": args.resume_from or None,
        "classes": {0: "damage", 1: "dark", 2: "light"},
        "roi": bool(args.roi),
        "roi_box": None if not args.roi else {
            "x0": roi_mod.ROI_X0, "x1": roi_mod.ROI_X1,
            "y0": roi_mod.ROI_Y0, "y1": roi_mod.ROI_Y1},
        "hyperparams": hyper,
        "hyperparams_effective": effective,
        "elapsed_s": round(time.time() - t0, 1),
        "best_pt": best if os.path.exists(best) else None,
        "best_pt_md5": md5_of(best) if os.path.exists(best) else None,
    }
    with open(os.path.join(run_dir, "_train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    # 日志归位
    try:
        # Windows 上文件句柄不关就删不掉，会残留 _<name>.train.log
        try:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
        except Exception:
            pass
        try:
            logf.flush()
            logf.close()
        except Exception:
            pass
        try:
            logging.getLogger().removeHandler(_log_handler)
            _log_handler.close()
        except Exception:
            pass
        final_log = os.path.join(run_dir, "train.log")
        shutil.copy2(log_path, final_log)
        # 转录不完整时用 results.csv 回放补齐 epoch 表（数值原样，不加工）
        n_nonblank = sum(1 for ln in open(final_log, encoding="utf-8") if ln.strip())
        if n_nonblank < 20:
            if rebuild_epoch_log(final_log, run_dir):
                print("\n[日志] 控制台转录不完整（%d 行），已用 results.csv 回放补齐 epoch 表" % n_nonblank)
        os.remove(log_path)
    except Exception as e:
        print("日志归位失败：%s" % e)

    print("\n训练结束，用时 %.1f 分钟" % (meta["elapsed_s"] / 60))
    print("best.pt: %s" % best)
    print("元信息: %s" % os.path.join(run_dir, "_train_meta.json"))
    print("日志  : %s" % os.path.join(run_dir, "train.log"))
    print("→ 下一步：用 scripts/eval_report.py 在冻结 test 上评估，确认后再人工复制进 models/启用/")


if __name__ == "__main__":
    main()
