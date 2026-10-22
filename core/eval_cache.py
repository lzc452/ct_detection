# -*- coding: utf-8 -*-
"""评估缓存契约：冻结输入指纹、ROI变换指纹与缓存内容校验，不依赖GPU。
缓存仅复用内容完全相同的图片/标签与裁剪规格；异常标注不能被当成负样本。
"""
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "ct-eval-cache/2"
ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_annotation(row):
    label_path = row.get("label_json")
    if not label_path:
        if int(row.get("n_boxes") or 0):
            raise ValueError("有框样本缺标注路径: " + row["name"])
        return
    path = ROOT / label_path
    data = json.loads(path.read_text(encoding="utf-8"))
    width, height = (int(v) for v in row["size"].split("x"))
    if (data.get("imageWidth"), data.get("imageHeight")) != (width, height):
        raise ValueError("标注尺寸与清单不一致: " + row["name"])
    shapes = data.get("shapes")
    if not isinstance(shapes, list):
        raise ValueError("标注缺少shapes列表: " + str(path))
    for shape in shapes:
        if str(shape.get("label", "")).strip() not in {"0", "1", "2"}:
            raise ValueError("标注存在未知类别: " + str(path))
        points = shape.get("points")
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError("标注框缺少坐标: " + str(path))
        if any(not isinstance(p, list) or len(p) != 2 for p in points):
            raise ValueError("标注坐标格式错误: " + str(path))
        xs, ys = [float(p[0]) for p in points], [float(p[1]) for p in points]
        if not all(math.isfinite(v) for v in xs + ys):
            raise ValueError("标注含非有限坐标: " + str(path))
        if min(xs) < 0 or max(xs) > width or min(ys) < 0 or max(ys) > height:
            raise ValueError("标注越出原图: " + str(path))
        if max(xs) - min(xs) < 1 or max(ys) - min(ys) < 1:
            raise ValueError("标注框退化: " + str(path))
    if "n_boxes" in row and len(shapes) != int(row["n_boxes"]):
        raise ValueError("标注框数与冻结清单不一致: " + row["name"])


def build_contract(rows, roi=False):
    import roi as roi_module
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("评估清单包含重名图片")
    if any(not name or "/" in name or "\\" in name or name in {".", ".."} for name in names):
        raise ValueError("图片名必须为安全文件名")
    inputs = []
    for row in rows:
        validate_annotation(row)
        src = ROOT / row["src_path"]
        # 清单是精确访问白名单；不扫描生产目录。额外校验原图仍与冻结MD5一致。
        digest = hashlib.md5()
        with src.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        if row.get("md5") and digest.hexdigest() != row["md5"]:
            raise ValueError("原图已偏离冻结清单: " + row["name"])
        inputs.append(dict(name=row["name"], src=str(src.resolve()), image_md5=digest.hexdigest(),
                           size=row["size"], label_sha256=file_sha256(ROOT / row["label_json"])
                           if row.get("label_json") else None))
    import cv2
    contract = dict(schema=SCHEMA, inputs=inputs, roi=bool(roi), jpeg_quality=95,
                    converter_sha256=file_sha256(ROOT / "core/dataset_split.py"),
                    preparer_sha256=file_sha256(ROOT / "scripts/eval_report.py"),
                    cv2_version=cv2.__version__,
                    roi_k=roi_module.get_margin_k() if roi else None,
                    roi_code_sha256=file_sha256(roi_module.__file__) if roi else None,
                    roi_box=[roi_module.ROI_X0, roi_module.ROI_Y0, roi_module.ROI_X1, roi_module.ROI_Y1]
                    if roi else None)
    serialized = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return contract, hashlib.sha256(serialized).hexdigest()


def validate_cached_files(out, metadata):
    out = Path(out)
    expected = set(metadata["outputs"])
    actual = {str(p.relative_to(out)).replace("\\", "/")
              for folder in (out / "images", out / "labels") for p in folder.iterdir() if p.is_file()}
    actual.add("data.yaml")
    if expected != actual:
        raise ValueError("评测缓存文件清单已变化: " + str(out))
    for relative, digest in metadata["outputs"].items():
        if file_sha256(out / relative) != digest:
            raise ValueError("评测缓存内容已变化: " + relative)
