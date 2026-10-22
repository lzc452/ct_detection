# -*- coding: utf-8 -*-
"""
训练实时监视页（本地只读，给"看训练过程"用）

- 读训练日志尾部（增量推送）+ 解析 results.csv 出每轮指标
- 只读文件，不干预训练；关掉监视页不影响训练进程

用法：
  runtime\\python.exe scripts\\train_watch.py --log .cache\\resume120_train.log ^
      --csv runs\\detect\\CT_pro\\V5_GROUPED_1021_resume120\\results.csv --port 5077
  浏览器打开 http://127.0.0.1:5077
"""
import os
import io
import csv
import sys
import json
import time
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>V5 续训监控</title>
<style>
  :root { --bg:#f7f8fa; --card:#fff; --line:#e3e6eb; --txt:#1f2430; --dim:#6b7280;
          --ok:#1a7f37; --warn:#b45309; --accent:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; padding:16px 18px 28px; background:var(--bg); color:var(--txt);
         font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; font-size:13px; }
  h1 { font-size:16px; margin:0 0 4px; font-weight:600; }
  .sub { color:var(--dim); font-size:12px; margin-bottom:14px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(132px,1fr)); gap:10px; margin-bottom:14px; }
  .kpi { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
  .kpi .k { color:var(--dim); font-size:11px; letter-spacing:.02em; }
  .kpi .v { font-size:19px; font-weight:600; margin-top:3px; font-variant-numeric:tabular-nums; }
  .kpi .v.best { color:var(--ok); }
  .kpi .v.warn { color:var(--warn); }
  .panel { background:var(--card); border:1px solid var(--line); border-radius:8px; margin-bottom:14px; overflow:hidden; }
  .panel > h2 { margin:0; padding:9px 12px; font-size:12px; font-weight:600; color:var(--dim);
                border-bottom:1px solid var(--line); background:#fbfcfd; letter-spacing:.03em; }
  table { width:100%; border-collapse:collapse; font-variant-numeric:tabular-nums; }
  th, td { padding:5px 10px; text-align:right; border-bottom:1px solid #f0f2f5; font-size:12px; }
  th { color:var(--dim); font-weight:500; background:#fbfcfd; position:sticky; top:0; }
  th:first-child, td:first-child { text-align:left; }
  .scroll { max-height:260px; overflow:auto; }
  pre { margin:0; padding:11px 12px; font-family:Consolas,"Cascadia Mono",monospace; font-size:11.5px;
        line-height:1.5; white-space:pre-wrap; word-break:break-all; max-height:330px; overflow:auto;
        background:#0f1420; color:#d7e0ee; border-radius:0 0 8px 8px; }
  .dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:var(--ok); margin-right:5px; }
  .dot.off { background:#c0c5cd; }
  .hint { color:var(--dim); font-size:11.5px; margin-top:10px; }
  code { background:#eef1f6; padding:1px 5px; border-radius:4px; font-size:11.5px; }
  .bar { height:6px; background:#eef1f6; border-radius:4px; overflow:hidden; margin-top:6px; }
  .bar > i { display:block; height:100%; background:var(--accent); width:0%; transition:width .4s; }
</style></head><body>
<h1>V5 续训监控 <span class="dot" id="dot"></span></h1>
<div class="sub" id="sub">连接中…</div>

<div class="grid">
  <div class="kpi"><div class="k">进度</div><div class="v" id="k_ep">-</div>
    <div class="bar"><i id="k_bar"></i></div></div>
  <div class="kpi"><div class="k">最优 mAP50</div><div class="v best" id="k_m50">-</div></div>
  <div class="kpi"><div class="k">最优 mAP50-95</div><div class="v best" id="k_m5095">-</div></div>
  <div class="kpi"><div class="k">当前 P / R</div><div class="v" id="k_pr">-</div></div>
  <div class="kpi"><div class="k">已用时</div><div class="v" id="k_time">-</div></div>
</div>

<div class="panel"><h2>每轮指标（最近 12 轮，按 mAP50-95 排序取前 5 已高亮最优）</h2>
  <div class="scroll"><table id="tbl">
    <thead><tr><th>epoch</th><th>mAP50</th><th>mAP50-95</th><th>P</th><th>R</th><th>box_loss</th></tr></thead>
    <tbody></tbody></table></div></div>

<div class="panel"><h2>训练日志（实时）</h2><pre id="log">等待日志…</pre></div>
<div class="hint">本页每 3 秒自取一次；<code>results.csv</code> 每轮写入后才出现新行。关掉本页不影响训练。</div>

<script>
let offset = 0;
function fmt(s){ if(s==null||isNaN(s)) return "-"; return (+s).toFixed(4); }
function dur(sec){ if(!sec) return "-"; const h=Math.floor(sec/3600), m=Math.round((sec%3600)/60);
  return h? (h+"h "+m+"m") : (m+"m"); }
async function tick(){
  try{
    const r = await fetch('/api?offset='+offset, {cache:'no-store'});
    const j = await r.json();
    document.getElementById('dot').className = 'dot' + (j.done ? ' off' : '');
    document.getElementById('sub').textContent = j.done ? '训练已结束（见下方日志末行）'
        : (j.running ? (j.epochs && j.epochs.length ? '训练进行中 · 每轮约 2.5 min · 日志 '+(j.log_size/1024|0)+' KB'
            : '训练已启动，正在跑第 1 轮（约 2.5 min 后出首个指标）· 日志 '+(j.log_size/1024|0)+' KB')
        : '未发现训练日志');
    // 日志增量
    if(j.text){ const el=document.getElementById('log');
      el.textContent += j.text; offset = j.offset;
      el.scrollTop = el.scrollHeight; }
    else if(j.reset){ document.getElementById('log').textContent='等待日志…'; offset=0; }
    // 指标
    const rows = j.epochs||[];
    if(rows.length){
      const last = rows[rows.length-1];
      const total = j.total_epochs||120;
      document.getElementById('k_ep').textContent = last.epoch+' / '+total;
      document.getElementById('k_bar').style.width = Math.min(100, 100*last.epoch/total)+'%';
      const b50 = rows.reduce((a,b)=> (+b.map50 > +a.map50 ? b : a));
      const b95 = rows.reduce((a,b)=> (+b.map5095 > +a.map5095 ? b : a));
      document.getElementById('k_m50').textContent = fmt(b50.map50)+' (e'+b50.epoch+')';
      document.getElementById('k_m5095').textContent = fmt(b95.map5095)+' (e'+b95.epoch+')';
      document.getElementById('k_pr').textContent = fmt(last.p)+' / '+fmt(last.r);
      document.getElementById('k_time').textContent = dur(last.time);
      const tb = document.querySelector('#tbl tbody');
      tb.innerHTML = rows.slice(-12).map(x=>
        '<tr'+(x.epoch==b95.epoch?' style="background:#f0fdf4"':'')+'><td>'+x.epoch+'</td><td>'+fmt(x.map50)+
        '</td><td>'+fmt(x.map5095)+'</td><td>'+fmt(x.p)+'</td><td>'+fmt(x.r)+'</td><td>'+
        (x.box_loss==null?'-':(+x.box_loss).toFixed(3))+'</td></tr>').join('');
    }
  }catch(e){ document.getElementById('dot').className='dot off';
    document.getElementById('sub').textContent='监视服务已断开：'+e; }
}
setInterval(tick, 3000); tick();
</script></body></html>"""


def parse_epochs(csv_path):
    if not os.path.exists(csv_path):
        return []
    try:
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return []
    out = []
    for r in rows:
        def g(k):
            v = r.get(k, "")
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        out.append({
            "epoch": int(g("epoch") or 0) + 1,
            "map50": g("metrics/mAP50(B)"),
            "map5095": g("metrics/mAP50-95(B)"),
            "p": g("metrics/precision(B)"),
            "r": g("metrics/recall(B)"),
            "box_loss": g("train/box_loss"),
            "time": g("time"),
        })
    return out


class H(BaseHTTPRequestHandler):
    log_path = None
    csv_path = None
    total_epochs = 120

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path != "/api":
            self.send_error(404)
            return

        q = parse_qs(u.query)
        try:
            offset = max(0, int(q.get("offset", ["0"])[0]))
        except ValueError:
            offset = 0

        size = os.path.getsize(self.log_path) if os.path.exists(self.log_path) else 0
        text, reset, new_off = "", False, offset
        if size and offset <= size:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                text = f.read().decode("utf-8", "replace")
                new_off = f.tell()
        elif size and offset > size:
            reset = True
            new_off = 0

        epochs = parse_epochs(self.csv_path)
        done = ("训练结束" in text) or ("training exited" in text)
        running = bool(size) and not done

        self._json({"offset": new_off, "text": text, "reset": reset,
                    "log_size": size, "epochs": epochs, "running": running,
                    "done": done, "total_epochs": self.total_epochs})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=".cache/resume120_train.log")
    ap.add_argument("--csv", default="runs/detect/CT_pro/V5_GROUPED_1021_resume120/results.csv")
    ap.add_argument("--port", type=int, default=5077)
    ap.add_argument("--total-epochs", type=int, default=120)
    args = ap.parse_args()

    H.log_path = os.path.abspath(args.log)
    H.csv_path = os.path.abspath(args.csv)
    H.total_epochs = args.total_epochs
    print("监视 %s" % H.log_path)
    print("指标 %s" % H.csv_path)
    print("打开 http://127.0.0.1:%d" % args.port)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), H)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
