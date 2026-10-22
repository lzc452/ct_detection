# -*- coding: utf-8 -*-
"""电芯 CT 系统证据审计：只读代码/归档，最多抽 6 张母本图，不用 GPU。
用法：runtime/python.exe -B scripts/audit_system.py --out reports/system_audit/evidence.json
不启动 Web、不运行训练、不创建数据集、不改库或母本；JPEG 对照仅在内存编码。
"""
import os
import sys
import ast
import csv
import json
import hashlib
import inspect
import argparse
import subprocess
from pathlib import Path
from collections import Counter
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
sys.path.insert(0, str(ROOT / "core"))
sys.dont_write_bytecode = True


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8").strip()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_inventory():
    entries = []
    for name in git("-c", "core.quotepath=false", "ls-files").splitlines():
        if not name.startswith(("core/", "scripts/", "web/")):
            continue
        if Path(name).suffix not in (".py", ".js", ".html", ".css", ".yaml"):
            continue
        text = (ROOT / name).read_text(encoding="utf-8-sig")
        entry = dict(path=name, lines=len(text.splitlines()), sha256=digest(ROOT / name))
        if name.endswith(".py"):
            try:
                ast.parse(text, filename=name)
                entry["syntax"] = "ok"
            except SyntaxError as exc:
                entry["syntax"] = str(exc)
        entry["legacy_path_lines"] = [i for i, line in enumerate(text.splitlines(), 1)
                                      if "D:\\CT Detection" in line and "CT Detection-yolo" not in line]
        entry["native_opencv_io_lines"] = [i for i, line in enumerate(text.splitlines(), 1)
                                          if "cv2.imread(" in line or "cv2.imwrite(" in line]
        entries.append(entry)
    return entries


def config_diffs():
    import yaml
    runs = ["V5_GROUPED_1021_80ep", "V51_PREPROC_1021", "V51_ROI_1536", "V52_ROI_1536_REG", "V53_ROI_M15"]
    configs = {}
    for run in runs:
        path = ROOT / "experiments" / run / "01_train/args.yaml"
        if not path.exists():
            path = ROOT / "runs/detect/CT_pro" / run / "args.yaml"
        if path.exists():
            configs[run] = yaml.safe_load(path.read_text(encoding="utf-8"))
    pairs = [(runs[0], runs[1]), (runs[1], runs[2]), (runs[2], runs[3]), (runs[0], runs[3]), (runs[3], runs[4])]
    out = []
    for a, b in pairs:
        if a not in configs or b not in configs:
            continue
        diff = {key: [configs[a].get(key), configs[b].get(key)]
                for key in sorted(configs[a].keys() | configs[b].keys())
                if configs[a].get(key) != configs[b].get(key) and key not in ("name", "save_dir", "project")}
        out.append(dict(baseline=a, candidate=b, differences=diff))
    return out


def manifest_stats(rows):
    out = {}
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        out[split] = dict(images=len(sub), groups=len({r["group"] for r in sub}),
                          images_with_cell_id=sum(bool(r["cell_id"]) for r in sub),
                          true_cells=len({r["cell_id"] for r in sub if r["cell_id"]}),
                          boxes={c: sum(int(r["n_" + c]) for r in sub) for c in ("damage", "dark", "light")},
                          negative_images=sum(int(r["is_neg"]) for r in sub),
                          resolutions=dict(Counter(r["size"] for r in sub)))
    return out


def scaling_probe(rows):
    import cv2
    import numpy as np
    from ultralytics.data.base import BaseDataset
    from ultralytics.data.augment import LetterBox
    import roi
    samples = []
    for split in ("train", "val", "test"):
        for high in (False, True):
            pool = [r for r in rows if r["split"] == split and (int(r["size"].split("x")[0]) > 1500) == high]
            row = next((r for r in pool if int(r["n_damage"]) > 0), pool[0])
            path = ROOT / row["src_path"]
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("抽样图无法解码: " + str(path))
            # 直接调用已安装库的 load_image；借用内存数组，避免读取/清理原目录的 .npy 缓存。
            for use_roi, imgsz in ((False, 1920), (True, 1536)):
                roi.set_margin_k(0)
                source = roi.crop_roi(image)[0] if use_roi else image
                obj = BaseDataset.__new__(BaseDataset)
                obj.ims = [None]
                obj.im_files = [str(path)]
                import uuid
                obj.npy_files = [ROOT / ("__audit_no_cache_" + uuid.uuid4().hex + ".npy")]
                obj.channels = 3
                obj.cv2_flag = cv2.IMREAD_COLOR
                obj.imgsz = imgsz
                obj.augment = False
                # 复用同一公开方法，用内存原图替代解码，绝不落盘。
                import unittest.mock as mock
                with mock.patch("ultralytics.data.base.imread", return_value=source.copy()):
                    resized, original_hw, resized_hw = obj.load_image(0)
                false = LetterBox((imgsz, imgsz), scaleup=False)(image=resized)
                true = LetterBox((imgsz, imgsz), scaleup=True)(image=resized)
                samples.append(dict(name=row["name"], split=split, roi=use_roi, imgsz=imgsz,
                                    original_hw=original_hw, resized_hw=resized_hw,
                                    scale=resized_hw[0] / original_hw[0],
                                    letterbox_scaleup_pixel_equal=bool(np.array_equal(false, true))))
    return samples


def library_probe():
    import ultralytics
    import ultralytics.data.augment as aug
    import ultralytics.data.base as base
    import ultralytics.data.dataset as dataset
    # 最小数据集替身仅用于构造增强链，不获取样本，不使用 GPU。
    from ultralytics.cfg import get_cfg
    transforms = aug.v8_transforms(SimpleNamespace(data={}, use_keypoints=False, cache=False), 1920, get_cfg())
    def tree(transform):
        return dict(name=type(transform).__name__, p=getattr(transform, "p", None),
                    children=[tree(t) for t in getattr(transform, "transforms", [])])
    detection_src = inspect.getsource(aug.v8_transforms)
    return dict(ultralytics=ultralytics.__version__,
                erasing_in_detection_transforms="erasing" in detection_src,
                auto_augment_in_detection_transforms="auto_augment" in detection_src,
                transforms=tree(transforms),
                source_sha256={m.__name__: digest(m.__file__) for m in (aug, base, dataset)})


def matching_probe():
    # 忠实重现原 eval_report 的先遇先配，与 sweep 的最大 IoU 匹配规则。
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_report import iou_xyxy
    gt = [[0, 0, 10, 10], [4, 0, 14, 10]]
    pred = [[2, 0, 14, 10], [0, 0, 10, 10]]  # 已按置信度从高到低
    def match(best_iou):
        used = set()
        for p in pred:
            valid = [(i, iou_xyxy(p, g)) for i, g in enumerate(gt) if i not in used and iou_xyxy(p, g) >= 0.5]
            if valid:
                used.add(max(valid, key=lambda x: x[1])[0] if best_iou else valid[0][0])
        return len(used)
    return dict(gt=gt, predictions=pred, first_acceptable_tp=match(False), max_iou_tp=match(True),
                note="合成重叠同类框反例，不声称历史实测指标已改变")


def mosaic_probe():
    import numpy as np
    import unittest.mock as mock
    from ultralytics.data.augment import Mosaic, RandomPerspective
    from ultralytics.utils.instance import Instances
    def label():
        return dict(img=np.zeros((320, 320, 3), dtype=np.uint8),
                    resized_shape=(320, 320), ori_shape=(320, 320),
                    im_file="synthetic", cls=np.array([[0]], dtype=np.float32),
                    instances=Instances(np.array([[240, 240, 256, 256]], dtype=np.float32),
                                        segments=np.zeros((0, 1000, 2), dtype=np.float32),
                                        bbox_format="xyxy", normalized=False))
    class Dummy:
        cache = "ram"
        def __len__(self):
            return 4
        def get_image_and_label(self, index):
            return label()
    # 固定在中心拼接、恒等尺度，直接走安装库的真实像素/标签实现。
    with mock.patch("random.uniform", side_effect=lambda a, b: (a + b) / 2):
        mosaic = Mosaic(Dummy(), imgsz=320, p=1)(label())
        output = RandomPerspective(degrees=0, translate=0, scale=0, shear=0,
                                   perspective=0, size=(320, 320))(mosaic)
    boxes = output["instances"].bboxes
    return dict(original_box_side=16, mosaic_hw=[640, 640], output_hw=list(output["img"].shape[:2]),
                output_boxes=boxes.tolist(),
                retained_widths=(boxes[:, 2] - boxes[:, 0]).tolist(),
                retained_heights=(boxes[:, 3] - boxes[:, 1]).tolist(),
                note="恒等尺度合成反例，否定必然缩半；不代表完整训练随机分布")


def device_probe():
    # 不分配GPU张量：只执行Web精度表达式，并模拟其设备选择器的返回值。
    text = (ROOT / "web/app.py").read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "USE_FP16" for t in n.targets))
    expression = compile(ast.Expression(node.value), "web/app.py:USE_FP16", "eval")
    from device_utils import is_cuda_device
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    cases = []
    for device in (0, "cuda:0", "cpu"):
        cases.append(dict(device=device, half=eval(expression, {"torch": fake_torch, "DEVICE": device, "is_cuda_device": is_cuda_device, "os": os})))
    return dict(pick_device_gpu_return=0, web_half_expression_results=cases,
                note="执行当前Web精度表达式；历史恒False缺陷已修复，原始证据仍保留在evidence.json")


def metrics():
    out = []
    for name in ("V5_80ep", "V51_2112", "V51_ROI_1536", "V52_ROI_1536_REG", "V5_resume120"):
        path = ROOT / "reports/eval" / (name + "_v5test.json")
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        wp = d["layer2_3_workpoint"]
        out.append(dict(tag=name, weights_md5=d.get("weights_md5"), workpoint=wp["workpoint"],
                        per_class=wp["per_class"], fp_per_1000img=wp["fp_per_1000img"],
                        group_level=wp["group_level"],
                        layer1=d.get("layer1_coco"), elapsed_s=d.get("elapsed_s")))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="reports/system_audit/evidence.json")
    args = ap.parse_args()
    with open(ROOT / "reports/splits/v5_split_manifest.csv", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    result = dict(schema="ct-system-audit/1", git_head=git("rev-parse", "HEAD"),
                  history=git("log", "--all", "--format=%h %ad %s", "--date=iso-strict").splitlines(),
                  inventory=source_inventory(), manifest_sha256=digest(ROOT / "reports/splits/v5_split_manifest.csv"),
                  manifest=manifest_stats(rows), config_diffs=config_diffs(), library=library_probe(),
                  scaling=scaling_probe(rows), matching=matching_probe(), mosaic=mosaic_probe(), device=device_probe(), metrics=metrics())
    path = Path(args.out).resolve()
    if not path.is_relative_to(ROOT / "reports"):
        raise SystemExit("审计输出必须放在 reports/ 内，避免覆盖代码或原始数据")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("证据已写出: " + str(path))
    print("源码文件 %d；合成/抽样缩放检查 %d；训练配置对照 %d" % (len(result["inventory"]), len(result["scaling"]), len(result["config_diffs"])))
    print(json.dumps({k: result[k] for k in ("manifest", "config_diffs", "matching")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
