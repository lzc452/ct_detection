# -*- coding: utf-8 -*-
"""Web精度契约回归：模拟GPU编号0、CPU回退与显式FP32，真实路由接收正确half参数。
用法：runtime/python.exe -B scripts/test_web_precision.py
假模型与临时结果目录隔离生产服务；不占用GPU，不读取生产图片或修改权重。
"""
import os
import json
import hashlib
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("PROCESSOR_ARCHITECTURE", "AMD64")
import sys
import importlib.util
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(ROOT / "scripts"))
from device_utils import is_cuda_device
from test_web_ledger import FakeYOLO, install_fake_ultralytics, jpeg_bytes


class PrecisionTests(unittest.TestCase):
    def test_device_identifiers(self):
        for device in (0, 1, "0", "cuda", "cuda:0", "cuda:1"):
            with self.subTest(device=device):
                self.assertTrue(is_cuda_device(device))
        for device in ("cpu", "mps", "xpu:0", -1, None, True, False):
            with self.subTest(device=device):
                self.assertFalse(is_cuda_device(device))

    def run_route(self, device, override, expected):
        with tempfile.TemporaryDirectory(prefix="ct_precision_") as tmp:
            base = Path(tmp)
            model_dir = base / "models"
            model_dir.mkdir()
            (model_dir / "fake.pt").write_bytes(b"fake")
            (model_dir / '_train_meta.json').write_text(json.dumps(dict(
                hyperparams=dict(imgsz=1920), roi=False, roi_k=0.0,
                best_pt_sha256=hashlib.sha256(b'fake').hexdigest())), encoding='utf8')
            env = dict(CT_MODEL_DIR=str(model_dir), CT_RESULTS_DIR=str(base / "results"),
                       CT_NO_BROWSER="1", CT_FP16=override)
            install_fake_ultralytics()
            FakeYOLO.calls.clear()
            name = "precision_web_" + str(device) + "_" + override
            with mock.patch.dict(os.environ, env), mock.patch("device_utils.pick_device", return_value=device):
                spec = importlib.util.spec_from_file_location(name, ROOT / "web/app.py")
                web = importlib.util.module_from_spec(spec)
                sys.modules[name] = web
                try:
                    spec.loader.exec_module(web)
                    self.assertEqual(web.USE_FP16, expected)
                    client = web.app.test_client()
                    response = client.post("/api/detect/image", data={
                        "file": (jpeg_bytes(100), "synthetic.jpg"), "conf": "0.30"},
                        content_type="multipart/form-data")
                    self.assertEqual(response.status_code, 200, response.get_json())
                    self.assertTrue(response.get_json()["success"], response.get_json())
                    self.assertEqual(FakeYOLO.calls[-1]["half"], expected)
                    self.assertEqual(FakeYOLO.calls[-1]["device"], device)
                    self.assertEqual(FakeYOLO.calls[-1]["imgsz"], 1920)
                    # 独立HTTP实例检查页面；已有5000服务不被停止或重启。
                    from werkzeug.serving import make_server
                    server = make_server("127.0.0.1", 0, web.app)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:%d/" % server.server_port, timeout=5) as page:
                            self.assertEqual(page.status, 200)
                    finally:
                        server.shutdown()
                        server.server_close()
                        thread.join(timeout=5)
                finally:
                    sys.modules.pop(name, None)

    def test_gpu_integer_enables_half(self):
        self.run_route(0, "1", True)

    def test_cpu_disables_half(self):
        self.run_route("cpu", "1", False)

    def test_explicit_fp32(self):
        self.run_route(0, "0", False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
